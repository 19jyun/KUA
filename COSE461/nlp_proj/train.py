import os
import glob
import pandas as pd
import torch
import logging
import warnings
from transformers import Trainer, TrainingArguments, RobertaTokenizer, AutoModelForSequenceClassification, default_data_collator, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import re

# Initialize logger
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
logging.disable(logging.WARNING)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="overflowing tokens are not returned for the setting you have chosen")
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.")

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, contexts=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.contexts = contexts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        context = str(self.contexts[idx]) if self.contexts is not None else None

        # Tokenize input
        inputs = self.tokenizer(
            text,
            text_pair=context,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_data_with_context(file_path):
    df = pd.read_csv(file_path)
    if 'context' in df.columns:
        texts = df['text'].tolist()
        contexts = df['context'].tolist()
        labels = df['label'].tolist()
        return texts, contexts, labels
    else:
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, None, labels

def get_latest_trained_model_path():
    # Read the status file to get the last trained dataset
    status_file = 'trained_model/train_status.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            last_trained = f.read().strip()
        # Construct the path to the last trained model
        model_path = os.path.join('trained_model', last_trained)
        if os.path.exists(model_path):
            return model_path
    return None

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = [total_samples / (2 * count) for count in class_counts]
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(tokenizer, model, train_dataset, eval_dataset, model_save_name, dataset_name, num_epochs=5, batch_size=16):  # Increased epochs
    class_weights = compute_class_weights(train_dataset.labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to('cuda'))

    training_args = TrainingArguments(
        output_dir=f'./results/{dataset_name}',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_dir=f'./logs/{dataset_name}',
        logging_steps=100,
        evaluation_strategy='steps',
        save_steps=500,  # Save checkpoint every 500 steps
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,  # Ensure consistent tensor sizes in the batch
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Added early stopping callback
        #criterion=loss_fn  # Use the custom loss function with class weights
    )
    trainer.train()
    
    # Save the trained model with the user-provided name
    model.save_pretrained(f"trained_model/{model_save_name}")
    with open(f"trained_model/{model_save_name}_trained", 'w') as f:
        f.write('This dataset has been trained.')

def clear_stale_marker_files():
    model_dirs = [d for d in os.listdir('trained_model') if os.path.isdir(os.path.join('trained_model', d))]
    for dir in model_dirs:
        if not os.path.exists(os.path.join('trained_model', dir, 'config.json')) and os.path.exists(os.path.join('trained_model', f"{dir}_trained")):
            os.remove(os.path.join('trained_model', f"{dir}_trained"))
            print(f"Removed stale marker file for {dir}")

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def select_model():
    models_directory = 'trained_model'
    model_files = [f for f in os.listdir(models_directory) if os.path.isdir(os.path.join(models_directory, f))]

    # Display the list of trained models
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")

    # Get user input
    model_number = int(input("Enter the number of the model you want to load (or 0 for roberta-based model): "))

    if model_number == 0:
        return 'roberta-base', 'roberta-base'
    else:
        selected_model_file = model_files[model_number - 1]
        return selected_model_file, os.path.join(models_directory, selected_model_file)

def select_dataset():
    # Get all dataset files in sorted order
    dataset_files = sorted(glob.glob('datasets/*.csv'), key=natural_sort_key)

    # Display the list of datasets
    for i, dataset_file in enumerate(dataset_files):
        print(f"{i+1}. {dataset_file}")

    # Get user input
    dataset_number = int(input("Enter the number of the dataset you want to train: "))
    selected_dataset_file = dataset_files[dataset_number - 1]
    dataset_name = os.path.basename(selected_dataset_file).split('.')[0]

    return selected_dataset_file, dataset_name

def update_status_file(dataset_name):
    with open('trained_model/train_status.txt', 'w') as f:
        f.write(dataset_name)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    f1 = f1_score(labels, preds, zero_division=1)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Allow user to select the model and dataset
    model_name, model_path = select_model()
    next_dataset, dataset_name = select_dataset()

    # Load the selected model or a base model
    if model_path == 'roberta-base':
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Clear any stale marker files
    clear_stale_marker_files()

    print(f"Processing dataset: {next_dataset}")
    print(f"Training model: {model_path if model_path else 'roberta-base'}")

    texts, contexts, labels = load_data_with_context(next_dataset)
    
    # Debugging: print out a few examples
    for idx in range(5):
        print(f"Text: {texts[idx]}")
        if contexts:
            print(f"Context: {contexts[idx]}")
        print(f"Label: {labels[idx]}")
        print("Tokenized Input: ", tokenizer(texts[idx], text_pair=contexts[idx] if contexts else None))
        print("\n")

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)
    train_contexts, eval_contexts = None, None
    if contexts is not None:
        train_contexts, eval_contexts = train_test_split(contexts, test_size=0.2, stratify=labels)

    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, max_len=128, contexts=train_contexts)
    eval_dataset = SarcasmDataset(eval_texts, eval_labels, tokenizer, max_len=128, contexts=eval_contexts)

    # Ask user for the model save name
    model_save_name = input("Enter the name to save the trained model: ")
    
    train_model(tokenizer, model, train_dataset, eval_dataset, model_save_name, dataset_name)

    # Update the status file
    update_status_file(dataset_name)

if __name__ == "__main__":
    main()
