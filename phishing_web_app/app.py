from flask import Flask, request, render_template
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np
import os

# Folder containing: config.json, model.safetensors, tokenizer files, etc.
MODEL_DIR = "final_model"

# Load tokenizer & model (BertForSequenceClassification as per config.json)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ---- LABEL HANDLING ----
# We assume a binary classifier:
#   0 -> Legitimate Email
#   1 -> Phishing Email
#
# If your labels are reversed, just swap the strings below.
label_map = {
    0: "Legitimate Email",
    1: "Phishing Email",
}

app = Flask(__name__)

def predict(text: str):
    """Run the BERT classifier on a single email text."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    friendly_label = label_map.get(pred_idx, f"Class {pred_idx}")
    confidence = float(probs[pred_idx]) * 100.0

    return friendly_label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result_label = None
    confidence = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if input_text:
            result_label, confidence = predict(input_text)

    return render_template(
        "index.html",
        result_label=result_label,
        confidence=confidence,
        input_text=input_text
    )

if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
