import argparse
import gc
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import RobertaModel, RobertaTokenizer

from file_io import read_list_from_jsonl_file, write_single_dict_to_json_file

# ---------------------- CONFIG ------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

emotion_list = [
    "anger",
    "brain dysfunction (forget)",
    "emptiness",
    "hopelessness",
    "loneliness",
    "sadness",
    "suicide intent",
    "worthlessness",
]
N_LABELS = len(emotion_list)


# ---------------------- LABEL UTIL ------------------------------

def label_id_to_vector(label_id, n_labels=N_LABELS):
    """
    label_id is a number whose digits are bits (1/0) for 8 emotions.
    Example: ["emptiness","hopelessness"] -> 00110000 -> stored as 110000.

    We:
      - convert to string
      - keep only last n_labels digits
      - left-pad with zeros
      - map chars to ints
    """
    s = str(label_id).strip()
    s = "".join(ch for ch in s if ch in "01")
    if len(s) == 0:
        s = "0" * n_labels

    s = s[-n_labels:]
    s = s.zfill(n_labels)

    return np.array([int(ch) for ch in s], dtype=np.float32)


# ---------------------- DATASET ------------------------------

class EmotionDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for item in data_list:
            text = item.get("text", "")
            lid = item.get("label_id", "0")
            self.texts.append(str(text))
            self.labels.append(label_id_to_vector(lid))

        self.labels = np.stack(self.labels, axis=0)  # [N, 8]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vec = self.labels[idx]

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vec, dtype=torch.float),
        }


# ---------------------- MODEL ------------------------------

class CategoryClassifier(nn.Module):
    def __init__(self, n_labels=N_LABELS, pretrained_model="roberta-base", dropout=0.4):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        # 🔥 increased dropout
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS
        x = self.drop(cls_emb)
        logits = self.out(x)
        return logits  # [batch, n_labels]


# ---------------------- CLASS-BALANCED POS_WEIGHTS ------------------------------

def compute_pos_weights(train_set, n_labels=N_LABELS):
    """
    pos_weight for BCEWithLogitsLoss:
    pos_weight[j] = (num_neg_j / num_pos_j)

    This upweights rare emotions like suicide intent.
    """
    counts = np.zeros(n_labels, dtype=np.float64)
    N = len(train_set)

    for item in train_set:
        vec = label_id_to_vector(item["label_id"], n_labels)
        counts += vec

    pos = counts
    # avoid division by zero
    pos = np.clip(pos, 1.0, None)
    neg = N - pos

    pos_weight = neg / pos
    return torch.tensor(pos_weight, dtype=torch.float32)


# ---------------------- METRICS ------------------------------

def compute_global_metrics(y_true, y_pred):
    """
    y_true, y_pred: np.array [N, N_LABELS], 0/1
    """
    metrics = {}

    metrics["f1_micro"] = f1_score(
        y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0
    )
    metrics["precision_micro"] = precision_score(
        y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0
    )

    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    return metrics


def compute_per_emotion_metrics(y_true, y_pred, emotions=emotion_list):
    per_emo = {}
    for i, emo in enumerate(emotions):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        f1 = f1_score(yt, yp, average="binary", zero_division=0)
        p = precision_score(yt, yp, average="binary", zero_division=0)
        r = recall_score(yt, yp, average="binary", zero_division=0)

        per_emo[emo] = {
            "f1_macro": f1,
            "precision_macro": p,
            "recall_macro": r,
            "f1_micro": f1,
            "precision_micro": p,
            "recall_micro": r,
            "avg": (f1 + p + r) / 3.0,
        }
    return per_emo


# ---------------------- THRESHOLD HELPERS ------------------------------

def apply_thresholds(probs, thresholds):
    """
    probs: [N, L] in [0,1]
    thresholds: scalar or array of shape [L]
    """
    probs = np.asarray(probs)
    if np.isscalar(thresholds):
        return (probs >= thresholds).astype(int)
    thr = np.asarray(thresholds)[None, :]  # [1, L]
    return (probs >= thr).astype(int)


def collect_probs_and_labels(model, data_loader, device):
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            labs = labels.cpu().numpy()

            all_probs.append(probs)
            all_true.append(labs)

    all_probs = np.vstack(all_probs)
    all_true = np.vstack(all_true)
    return all_true, all_probs


def tune_labelwise_thresholds(y_true, y_probs, grid=None):
    """
    🔥 Label-wise threshold tuning.
    y_true: [N, L] 0/1
    y_probs: [N, L] in [0,1]
    grid: list/array of thresholds to scan
    """
    if grid is None:
        # fine-ish grid from 0.1 to 0.9
        grid = np.linspace(0.1, 0.9, 17)

    n_labels = y_true.shape[1]
    best_thresholds = np.zeros(n_labels, dtype=np.float32)

    for j in range(n_labels):
        yt = y_true[:, j]
        best_f1 = 0.0
        best_thr = 0.5
        for thr in grid:
            yp = (y_probs[:, j] >= thr).astype(int)
            f1 = f1_score(yt, yp, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        best_thresholds[j] = best_thr
    return best_thresholds


# ---------------------- TRAIN / EVAL EPOCHS ------------------------------

def train_epoch(model, data_loader, loss_fn, optimizer, device, thresholds=0.5):
    model.train()
    losses = []
    all_true = []
    all_pred = []

    for i, batch in enumerate(data_loader):
        sys.stdout.write(f"\rTraining batch: {i+1}/{len(data_loader)}")
        sys.stdout.flush()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = apply_thresholds(probs, thresholds)
        true = labels.detach().cpu().numpy().astype(int)

        all_true.append(true)
        all_pred.append(preds)

    print()
    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)

    metrics = compute_global_metrics(all_true, all_pred)
    avg_loss = float(np.mean(losses))
    return metrics, avg_loss


def eval_epoch(model, data_loader, loss_fn, device, thresholds=0.5):
    model.eval()
    losses = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = apply_thresholds(probs, thresholds)
            true = labels.cpu().numpy().astype(int)

            all_true.append(true)
            all_pred.append(preds)

    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)

    metrics = compute_global_metrics(all_true, all_pred)
    avg_loss = float(np.mean(losses))
    return metrics, avg_loss


# ---------------------- TRAIN WRAPPER ------------------------------

def train_model(
    train_set,
    val_set,
    pretrained_model="roberta-base",
    saved_model_file="best_roberta_multilabel.bin",
    saved_history_file="history_roberta_multilabel.json",
    thresholds_file="best_label_thresholds.npy",
    epochs=15,
    max_len=256,
    batch_size=8,
    lr=2e-5,
    weight_decay=0.01,
    base_threshold=0.5,
    patience=3,
):

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)

    train_loader = DataLoader(
        EmotionDataset(train_set, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        EmotionDataset(val_set, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = CategoryClassifier(N_LABELS, pretrained_model=pretrained_model, dropout=0.4).to(device)

    # 🔥 class-balanced BCE
    pos_weight = compute_pos_weights(train_set, N_LABELS).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 🔥 LR schedule: reduce after epoch 4
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.5
    )

    best_val_f1_mac = 0.0
    best_epoch = 0
    write_single_dict_to_json_file(saved_history_file, {}, "w")
    epochs_since_improve = 0

    # we train/validate with a global threshold 0.5;
    # label-wise thresholds will be tuned AFTER training using val probs.
    train_thr = base_threshold
    val_thr = base_threshold

    for epoch in range(1, epochs + 1):
        print("-" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_metrics, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, thresholds=train_thr
        )
        val_metrics, val_loss = eval_epoch(
            model, val_loader, loss_fn, device, thresholds=val_thr
        )

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Train F1-Mac: {train_metrics['f1_macro']:.4f}, "
            f"Train F1-Mic: {train_metrics['f1_micro']:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f}, "
            f"Val   F1-Mac: {val_metrics['f1_macro']:.4f}, "
            f"Val   F1-Mic: {val_metrics['f1_micro']:.4f}"
        )

        val_f1_mac = val_metrics["f1_macro"]

        if val_f1_mac > best_val_f1_mac:
            best_val_f1_mac = val_f1_mac
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), saved_model_file)
            print(
                f"  ✅ New best model saved (epoch {epoch}, "
                f"Val F1-Mac={val_f1_mac:.4f})"
            )
        else:
            epochs_since_improve += 1
            print(f"  No improvement for {epochs_since_improve} epoch(s).")

        history_dict = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_val_f1_mac,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        write_single_dict_to_json_file(saved_history_file, history_dict)

        torch.cuda.empty_cache()
        gc.collect()

        # 🔥 LR step after each epoch (so after epoch 4 LR is halved)
        lr_scheduler.step()

        if epochs_since_improve >= patience:
            print(f"\n⏹ Early stopping triggered (patience={patience}).")
            break

    print(f"\nBest Val F1-Macro = {best_val_f1_mac:.4f} at epoch {best_epoch}")

    # 🔥 After training: load best model and tune label-wise thresholds on val set
    print("\nTuning label-wise thresholds on validation set...")
    best_model = CategoryClassifier(N_LABELS, pretrained_model=pretrained_model, dropout=0.4)
    best_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    best_model.to(device)

    y_val_true, y_val_probs = collect_probs_and_labels(best_model, val_loader, device)
    best_thresholds = tune_labelwise_thresholds(y_val_true, y_val_probs)
    np.save(thresholds_file, best_thresholds)
    print("Saved label-wise thresholds to:", thresholds_file)
    print("Thresholds:", best_thresholds)


# ---------------------- TEST / FULL EVAL ------------------------------

def test_dataset(
    test_set,
    pretrained_model="roberta-base",
    saved_model_file="best_roberta_multilabel.bin",
    thresholds_file="best_label_thresholds.npy",
    max_len=256,
    batch_size=16,
    fallback_threshold=0.5,
):

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    test_loader = DataLoader(
        EmotionDataset(test_set, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = CategoryClassifier(N_LABELS, pretrained_model=pretrained_model, dropout=0.4)
    model.load_state_dict(torch.load(saved_model_file, map_location=device))
    model.to(device)

    # class-balanced BCE just for loss reporting
    pos_weight = compute_pos_weights(test_set, N_LABELS).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # 🔥 load label-wise thresholds if available
    if os.path.exists(thresholds_file):
        thresholds = np.load(thresholds_file)
        print("Using label-wise thresholds from:", thresholds_file)
    else:
        print("Threshold file not found, using global threshold:", fallback_threshold)
        thresholds = fallback_threshold

    model.eval()
    losses = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = apply_thresholds(probs, thresholds)
            true = labels.cpu().numpy().astype(int)

            all_true.append(true)
            all_pred.append(preds)

    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)

    metrics = compute_global_metrics(all_true, all_pred)
    avg_loss = float(np.mean(losses))

    print("\n=========== OVERALL TEST METRICS ===========")
    print(f"Loss:      {avg_loss:.4f}")
    print(f"F1-Macro:  {metrics['f1_macro']:.4f}")
    print(f"P-Macro:   {metrics['precision_macro']:.4f}")
    print(f"R-Macro:   {metrics['recall_macro']:.4f}")
    print(f"F1-Micro:  {metrics['f1_micro']:.4f}")
    print(f"P-Micro:   {metrics['precision_micro']:.4f}")
    print(f"R-Micro:   {metrics['recall_micro']:.4f}")

    per_emo = compute_per_emotion_metrics(all_true, all_pred, emotion_list)

    print("\n=========== PER-EMOTION METRICS ===========")
    print("Emotion                 | F1-Mac | P-Mac | R-Mac | F1-Mic | P-Mic | R-Mic | Avg")
    print("-" * 80)
    for emo in emotion_list:
        m = per_emo[emo]
        print(
            f"{emo:23} | "
            f"{m['f1_macro']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} | "
            f"{m['f1_micro']:.4f} | {m['precision_micro']:.4f} | {m['recall_micro']:.4f} | "
            f"{m['avg']:.4f}"
        )

    return metrics, per_emo


# ---------------------- MAIN ------------------------------

def main(args):
    train_set = read_list_from_jsonl_file(args.train_path)
    val_set = read_list_from_jsonl_file(args.val_path)
    test_set = read_list_from_jsonl_file(args.test_path)

    if args.mode == "train":
        train_model(
            train_set,
            val_set,
            pretrained_model=args.model_name,
            saved_model_file=args.saved_model_file,
            saved_history_file=args.history_file,
            thresholds_file=args.thresholds_file,
            epochs=args.epochs,
            max_len=args.max_length,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            base_threshold=args.base_threshold,
            patience=args.patience,
        )
    else:  # test
        test_dataset(
            test_set,
            pretrained_model=args.model_name,
            saved_model_file=args.saved_model_file,
            thresholds_file=args.thresholds_file,
            max_len=args.max_length,
            batch_size=args.test_batch_size,
            fallback_threshold=args.base_threshold,
        )


# ---------------------- ENTRY ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label RoBERTa Emotion Training with Threshold Tuning")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model_name", type=str, default="roberta-base")
    # For roberta-large just pass --model_name roberta-large (if VRAM allows)

    parser.add_argument("--train_path", type=str, default="/kaggle/working/DepressionEmo/Dataset/train.json")
    parser.add_argument("--val_path", type=str, default="/kaggle/working/DepressionEmo/Dataset/val.json")
    parser.add_argument("--test_path", type=str, default="/kaggle/working/DepressionEmo/Dataset/test.json")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--base_threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--saved_model_file", type=str, default="best_roberta_multilabel.bin")
    parser.add_argument("--history_file", type=str, default="history_roberta_multilabel.json")
    parser.add_argument("--thresholds_file", type=str, default="best_label_thresholds.npy")

    args, unknown = parser.parse_known_args()
    main(args)
