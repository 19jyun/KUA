import os
import glob
import pandas as pd
import torch
import logging
import warnings
from transformers import Trainer, TrainingArguments, RobertaTokenizer, AutoModelForSequenceClassification, default_data_collator, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
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
            truncation='longest_first',
            return_tensors='pt',
            return_overflowing_tokens=False  # Ensure no overflowing tokens are returned
        )

        # Remove batch dimension
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

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

def train_model(tokenizer, model, train_dataset, eval_dataset, dataset_name, num_epochs=5, batch_size=16):  # Increased epochs
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Added early stopping callback
    )
    trainer.train()
    model.save_pretrained(f"trained_model/{dataset_name}")
    with open(f"trained_model/{dataset_name}_trained", 'w') as f:
        f.write('This dataset has been trained.')

def clear_stale_marker_files():
    model_dirs = [d for d in os.listdir('trained_model') if os.path.isdir(os.path.join('trained_model', d))]
    for dir in model_dirs:
        if not os.path.exists(os.path.join('trained_model', dir, 'config.json')) and os.path.exists(os.path.join('trained_model', f"{dir}_trained")):
            os.remove(os.path.join('trained_model', f"{dir}_trained"))
            print(f"Removed stale marker file for {dir}")

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def get_next_dataset_to_train():
    # Read the status file
    status_file = 'trained_model/train_status.txt'
    last_trained = None
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            last_trained = f.read().strip()

    # Get all dataset files in sorted order
    dataset_files = sorted(glob.glob('datasets/*.csv'), key=natural_sort_key)

    # Check for next dataset to train
    found_last_trained = False
    for filename in dataset_files:
        dataset_name = os.path.basename(filename).split('.')[0]
        if found_last_trained:
            marker_path = f"trained_model/{dataset_name}_trained"
            if not os.path.exists(marker_path):
                return filename, dataset_name
        if dataset_name == last_trained:
            found_last_trained = True

    # If last_trained is None, return the first dataset
    if last_trained is None and dataset_files:
        first_dataset = dataset_files[0]
        return first_dataset, os.path.basename(first_dataset).split('.')[0]

    return None, None

def update_status_file(dataset_name):
    with open('trained_model/train_status.txt', 'w') as f:
        f.write(dataset_name)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Get the next dataset to train
    next_dataset, dataset_name = get_next_dataset_to_train()
    if not next_dataset:
        print("All datasets have been trained.")
        return

    # Load the most recently trained model or a base model
    latest_model_path = get_latest_trained_model_path()
    if (latest_model_path is not None) and os.path.exists(latest_model_path):
        model = AutoModelForSequenceClassification.from_pretrained(latest_model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Clear any stale marker files
    clear_stale_marker_files()

    print(f"Processing dataset: {next_dataset}")
    print(f"Training model: {latest_model_path if latest_model_path else 'roberta-base'}")

    df = load_data(next_dataset)
    train_df, eval_df = train_test_split(df, test_size=0.2)

    contexts = train_df['context'].to_numpy() if 'context' in train_df.columns else None
    train_dataset = SarcasmDataset(
        train_df['text'].to_numpy(), 
        train_df['label'].to_numpy(), 
        tokenizer, 
        max_len=128,  # Use 128 to match previous training settings
        contexts=contexts
    )

    contexts = eval_df['context'].to_numpy() if 'context' in eval_df.columns else None
    eval_dataset = SarcasmDataset(
        eval_df['text'].to_numpy(), 
        eval_df['label'].to_numpy(), 
        tokenizer, 
        max_len=128,  # Use 128 to match previous training settings
        contexts=contexts
    )

    train_model(tokenizer, model, train_dataset, eval_dataset, dataset_name)

    # Update the status file
    update_status_file(dataset_name)

if __name__ == "__main__":
    main()
