# predict.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

# ─── Labels ───────────────────────────────────────────────────────────────────
LABELS = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

# ─── Load tokenizer and model ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("hate_model")
model = BertForSequenceClassification.from_pretrained(
    "hate_model",
    ignore_mismatched_sizes=True,
)
model.eval()

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict_comment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return {
        "label": LABELS[pred_idx],
        "probability": round(probs[0][pred_idx].item(), 4)
    }
