import pandas as pd
import os
import kaggle
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import RobertaTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_data():
    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load datasets from Hugging Face
    huggingface_datasets = [
        'AgamP/sarcasm-detection',
        'FlorianKibler/sarcasm_dataset_en'
    ]

    for i, dataset_name in enumerate(huggingface_datasets):
        # Load dataset
        dataset = load_dataset(dataset_name)

        # Split data
        texts = dataset['train']['headline']
        labels = dataset['train']['is_sarcastic']
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Convert data to DataFrame
        df = pd.DataFrame({
            'text': train_texts + test_texts,
            'label': train_labels + test_labels
        })

        # Save data to CSV file
        df.to_csv(f'datasets/sarcasm_dataset_{i}.csv', index=False)

    # Download datasets from Kaggle
    kaggle_datasets = ['nikhiljohnk/tweets-with-sarcasm-and-irony']

    name_of_sarc = ['class']
    name_of_head = ['tweets']
    det_of_sarc = ['sarcasm']
    name_of_train = ['train.csv']
    name_of_test = ['']

    for i, dataset in enumerate(kaggle_datasets):
        kaggle.api.dataset_download_files(dataset, path='./', unzip=True)
        
        if name_of_train[i]:
            train_df = pd.read_csv(name_of_train[i])
            train_df.rename(columns={name_of_head[i]: 'headline'}, inplace=True)
            train_df['is_sarcastic'] = train_df[name_of_sarc[i]].apply(lambda x: 1 if str(x) == det_of_sarc[i] else 0)
            train_df = train_df[['headline', 'is_sarcastic']]
            
            os.makedirs('datasets', exist_ok=True)
            
            existing_datasets = len([name for name in os.listdir('datasets') if os.path.isfile(os.path.join('datasets', name))])
            start_index = existing_datasets
            
            train_texts = train_df['headline'].tolist()
            train_labels = train_df['is_sarcastic'].tolist()

            train_df_i = pd.DataFrame({'text': train_texts, 'label': train_labels})
            train_df_i.to_csv(f'datasets/sarcasm_dataset_{start_index}.csv', index=False)

        if name_of_test[i]:
            test_df = pd.read_csv(name_of_test[i])
            test_df.rename(columns={name_of_head[i]: 'headline'}, inplace=True)
            test_df['is_sarcastic'] = test_df[name_of_sarc[i]].apply(lambda x: 1 if str(x) == det_of_sarc[i] else 0)
            test_df = test_df[['headline', 'is_sarcastic']]

            os.makedirs('test_datasets', exist_ok=True)

            existing_test_datasets = len([name for name in os.listdir('test_datasets') if os.path.isfile(os.path.join('test_datasets', name))])
            
            start_test_index = existing_test_datasets

            test_texts = test_df['headline'].tolist()
            test_labels = test_df['is_sarcastic'].tolist()

            test_df_i = pd.DataFrame({'text': test_texts, 'label': test_labels})
            test_df_i.to_csv(f'test_datasets/test_sarcasm_dataset_{start_test_index}.csv', index=False)

def prepare_data_with_context(dataset_name):
    # Read file
    df = pd.read_csv(dataset_name)

    global_index_for_context_dataset = 0

    if 'parent_comment' in df.columns:
        # Select necessary columns
        df = df[['comment', 'parent_comment', 'label']]

        # Rename columns
        df.columns = ['text', 'context', 'label']

        # Split data into groups of 50,000 rows
        groups = df.groupby(np.arange(len(df)) // 50000)

        # Save each group to a separate file
        for _, group in groups:
            group.to_csv(f'datasets/sarcasm_dataset_with_context_{global_index_for_context_dataset}.csv', index=False)
            global_index_for_context_dataset += 1
    else:
        # Handle datasets without context
        df = df[['comment', 'label']]
        df.columns = ['text', 'label']

        groups = df.groupby(np.arange(len(df)) // 50000)

        for _, group in groups:
            group.to_csv(f'datasets/sarcasm_dataset_without_context_{global_index_for_context_dataset}.csv', index=False)
            global_index_for_context_dataset += 1

if __name__ == "__main__":
    prepare_data()
    prepare_data_with_context('train-balanced-sarcasm.csv')
