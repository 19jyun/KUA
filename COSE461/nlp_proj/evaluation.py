from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def write_evaluation_result(model_name, dataset_name, accuracy, precision, recall, f1_score):
    accuracy_file = 'results/accuracy.csv'
    record_exists = False

    # Check if the accuracy file exists
    if os.path.exists(accuracy_file):
        # Load the accuracy file
        df = pd.read_csv(accuracy_file)

        # Check if the model has been evaluated with the specific dataset
        record_exists = ((df['Model'] == model_name) & (df['Test set'] == dataset_name)).any()

    # If the record does not exist, write the new record
    if not record_exists:
        with open(accuracy_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, dataset_name, accuracy, precision, recall, f1_score])
            print(f"The evaluation results of {model_name} on {dataset_name} have been saved to {accuracy_file}.")

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

# Display the list of test datasets
test_datasets_directory = 'test_datasets'
test_data_files = [f for f in os.listdir(test_datasets_directory) if f.endswith('.csv')]

for i, test_data_file in enumerate(test_data_files):
    print(f"{i+1}. {test_data_file}")

# Get user input
test_data_number = int(input("Enter the number of the test dataset you want to load: "))

# Load the selected test dataset
selected_test_data_file = test_data_files[test_data_number - 1]
print(f"Loading test dataset: {selected_test_data_file}")
test_data_path = os.path.join(test_datasets_directory, selected_test_data_file)
test_data = pd.read_csv(test_data_path)

test_data = test_data[test_data['text'].notna()]

if 'context' in test_data.columns:
    inputs = tokenizer(test_data['text'].tolist(), text_pair=test_data['context'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
else:
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
all_preds = []
all_labels = []
for batch in data_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():        
        outputs = model(**batch)
        
    logits = outputs.logits
    _, preds = torch.max(logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch['labels'].cpu().numpy())
    total_eval_accuracy += accuracy_score(batch['labels'].cpu(), preds.cpu())

avg_val_accuracy = total_eval_accuracy / len(data_loader)
precision = precision_score(all_labels, all_preds, zero_division=1)
recall = recall_score(all_labels, all_preds, zero_division=1)
f1 = f1_score(all_labels, all_preds, zero_division=1)
print(f'Accuracy: {avg_val_accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

write_evaluation_result(selected_model_file, selected_test_data_file, avg_val_accuracy, precision, recall, f1)
