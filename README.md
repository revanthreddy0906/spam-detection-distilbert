# Spam Detection using DistilBERT

This project implements a **spam detection system** using a transformer-based
model (`distilbert-base-uncased`) fine-tuned on SMS spam data.

The trained model is published on **Hugging Face Hub** and can be loaded
directly for inference.

---

## 🔗 Trained Model (Hugging Face)

👉 https://huggingface.co/WhiteDevilOP/spam-detection-distilbert

---

## 📌 Task
Binary text classification:
- `0` → Ham (not spam)
- `1` → Spam

---

## 🧠 Model Details
- Base model: `distilbert-base-uncased`
- Architecture: Transformer (DistilBERT)
- Framework: PyTorch + Hugging Face Transformers
- Training epochs: 3
- Evaluation metric: F1-score (~0.97)

---

## 🚀 Usage (Inference)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "WhiteDevilOP/spam-detection-distilbert"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "WhiteDevilOP/spam-detection-distilbert"
)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    return "SPAM" if pred == 1 else "HAM"

print(predict("You won a free iPhone!"))

---

##🏋️ Training

```bash
pip install -r requirements.txt
python train.py
```
---

## 📈 Results
- Accuracy : ~99%
- F1-Score:~0.97
