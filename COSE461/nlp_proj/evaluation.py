from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import os
import csv

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_directory = 'trained_model'
model_files = [f for f in os.listdir(models_directory) if os.path.isdir(os.path.join(models_directory, f))]

# Display the list of trained models
for i, model_file in enumerate(model_files):
    print(f"{i+1}. {model_file}")

# Get user input
model_number = int(input("Enter the number of the model you want to load: "))

# Load the selected model
selected_model_file = model_files[model_number - 1]
print(f"Loading model: {selected_model_file}")
model_path = os.path.join(models_directory, selected_model_file)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

test_data = pd.read_csv('test_datasets/test_sarcasm_dataset_0.csv')
test_data = test_data[test_data['text'].notna()]

inputs = tokenizer(test_data['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
labels = test_data['label'].tolist()

dataset = TextDataset(inputs, labels)

batch_size = 16
data_loader = DataLoader(
    dataset, 
    sampler=SequentialSampler(dataset), 
    batch_size=batch_size
)

model.eval()
total_eval_accuracy = 0
for batch in data_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():        
        outputs = model(**batch)
        
    logits = outputs.logits
    _, preds = torch.max(logits, dim=1)
    total_eval_accuracy += accuracy_score(batch['labels'].cpu(), preds.cpu())

avg_val_accuracy = total_eval_accuracy / len(data_loader)
print(f'Accuracy: {avg_val_accuracy}')

if not os.path.exists('results/accuracy.csv'):
    with open('results/accuracy.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Accuracy"])

with open('results/accuracy.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    if any(selected_model_file in row for row in reader):
        print(f"The accuracy of {selected_model_file} is already saved in results/accuracy.csv.")
    else:
        with open('results/accuracy.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([selected_model_file, avg_val_accuracy])
            print(f"The accuracy of {selected_model_file} has been saved to results/accuracy.csv.")
