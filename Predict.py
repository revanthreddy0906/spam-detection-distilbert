from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "./results/checkpoint-837"

# Load trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "SPAM" if prediction == 1 else "HAM"


# Test
if __name__ == "__main__":
    while True:
        msg = input("Enter message (or 'exit'): ")
        if msg.lower() == "exit":
            break
        print("Prediction:", predict(msg))