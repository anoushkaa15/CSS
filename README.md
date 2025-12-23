# Improving Depression Emotion Classification Using RoBERTa

This repository contains the implementation for **multilabel depression emotion classification** using a **RoBERTa-based transformer model**, as presented in our paper *“Improving Depression Emotion Classification Using RoBERTa”*.
The work focuses on capturing **fine-grained, co-occurring emotional states** expressed in long-form social media text, rather than relying on coarse binary depression detection .

---

## 📌 Overview

Depression manifests through multiple overlapping emotions such as hopelessness, loneliness, guilt, and suicidal ideation. Traditional binary classification approaches fail to capture this complexity.

This project:

* Formulates depression detection as a **multilabel emotion classification task**
* Uses **RoBERTa-base** for rich contextual representation
* Addresses **severe class imbalance** using class-balanced loss
* Applies **per-label threshold optimization** to improve precision–recall trade-offs
* Evaluates performance using **F1-Macro, F1-Micro, ROC-AUC, and PR-AUC**

---

## 🧠 Emotion Labels

The model predicts the following **8 depression-related emotions**:

1. Hopelessness
2. Worthlessness
3. Loneliness
4. Sadness
5. Anger
6. Suicide Intent
7. Cognitive Dysfunction
8. Emptiness

Each input text can be assigned **multiple labels simultaneously**.

---

## 🏗️ Model Architecture

* **Encoder:** RoBERTa-base (12 layers, 768 hidden size, 12 attention heads)
* **Pooling:** `[CLS]` token representation
* **Regularization:** Dropout (p = 0.4)
* **Classifier:** Fully connected linear layer (8 outputs)
* **Loss Function:** Binary Cross-Entropy with Logits + label-specific positive weights

---

## ⚖️ Handling Class Imbalance

To mitigate skewed label distributions:

* Label-wise **positive class weights** are computed
* Rare but critical emotions (e.g., suicide intent) are penalized more heavily during training
* Improves recall for underrepresented emotions without sacrificing overall performance

---

## 🎯 Threshold Optimization

Instead of a fixed 0.5 threshold:

* Thresholds are optimized **per emotion label**
* Values from 0.1 to 0.9 are evaluated on a validation set
* Thresholds maximizing **label-wise F1-score** are selected
* Enables emotion-specific precision–recall balancing

---

## 📊 Results

| Metric                   | Score  |
| ------------------------ | ------ |
| **F1-Macro**             | 0.80   |
| **F1-Micro**             | 0.84   |
| **ROC-AUC (all labels)** | > 0.87 |

* Strong performance on frequent emotions (sadness, hopelessness, loneliness)
* Improved recall on rare emotions compared to transformer baselines
* Predicted emotion correlations align with psychological theory

---

## 📚 Dataset

This project uses the **DepressionEmo** dataset:

* 6,037 long-form Reddit posts
* Annotated using a **majority vote over zero-shot models**
* Validated by human annotators and ChatGPT
* Designed specifically for multilabel depression emotion analysis 

---

## ⚠️ Ethical Considerations

This work is intended **strictly for research purposes**.

* Outputs should **not** be used as clinical diagnoses
* Human-in-the-loop systems are essential for real-world deployment
* Privacy, consent, and misuse risks must be carefully managed

---
