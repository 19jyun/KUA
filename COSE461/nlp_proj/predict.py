from transformers import RobertaTokenizer
from model import get_model
import torch

def make_prediction(texts):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = get_model()
    model.eval()  # Set the model to evaluation mode
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_input)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

if __name__ == "__main__":
    texts = ["Example of a sarcastic comment.", "Just a normal statement."]
    predictions = make_prediction(texts)
    print(predictions)
