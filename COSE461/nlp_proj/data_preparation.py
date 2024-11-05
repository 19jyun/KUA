import pandas as pd
import os
import kaggle
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from datasets import load_dataset
from transformers import RobertaTokenizer
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

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

def oversample_data(X, y, context=None):
    ros = RandomOverSampler(random_state=0)
    if context:
        X_resampled, y_resampled = ros.fit_resample(np.array(X + context).reshape(-1, 1), y)
        X_resampled = X_resampled[:, 0].tolist()
        print(f"After oversampling - Label 0: {y_resampled.count(0)} samples ({y_resampled.count(0) / len(y_resampled) * 100:.2f}%)")
        print(f"After oversampling - Label 1: {y_resampled.count(1)} samples ({y_resampled.count(1) / len(y_resampled) * 100:.2f}%)")
        return X_resampled[:len(X_resampled)//2], X_resampled[len(X_resampled)//2:], y_resampled
    else:
        X_resampled, y_resampled = ros.fit_resample(np.array(X).reshape(-1, 1), y)
        X_resampled = X_resampled[:, 0].tolist()
        print(f"After oversampling - Label 0: {y_resampled.count(0)} samples ({y_resampled.count(0) / len(y_resampled) * 100:.2f}%)")
        print(f"After oversampling - Label 1: {y_resampled.count(1)} samples ({y_resampled.count(1) / len(y_resampled) * 100:.2f}%)")
        return X_resampled, y_resampled

def prepare_data():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    huggingface_datasets = [
        'AgamP/sarcasm-detection',
        'FlorianKibler/sarcasm_dataset_en'
    ]

    for i, dataset_name in enumerate(huggingface_datasets):
        dataset = load_dataset(dataset_name)

        texts = dataset['train']['headline']
        labels = dataset['train']['is_sarcastic']
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

        print(f"{dataset_name} (Train) - Label 0: {train_labels.count(0)} samples ({train_labels.count(0) / len(train_labels) * 100:.2f}%)")
        print(f"{dataset_name} (Train) - Label 1: {train_labels.count(1)} samples ({train_labels.count(1) / len(train_labels) * 100:.2f}%)")

        if train_labels.count(0) / len(train_labels) * 100 < 40 or train_labels.count(0) / len(train_labels) * 100 > 60:
            print("Applying oversampling to handle class imbalance.")
            texts, labels = oversample_data(texts, labels)

        df = pd.DataFrame({
            'text': train_texts + test_texts,
            'label': train_labels + test_labels
        })

        df.to_csv(f'datasets/sarcasm_dataset_{i}.csv', index=False)

    kaggle_datasets = ['nikhiljohnk/tweets-with-sarcasm-and-irony']

    name_of_sarc = ['class']
    name_of_head = ['tweets']
    det_of_sarc = ['sarcasm']
    name_of_train = ['train.csv']
    name_of_test = ['test.csv']

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

            print(f"Kaggle Dataset {i} (Train) - Label 0: {train_labels.count(0)} samples ({train_labels.count(0) / len(train_labels) * 100:.2f}%)")
            print(f"Kaggle Dataset {i} (Train) - Label 1: {train_labels.count(1)} samples ({train_labels.count(1) / len(train_labels) * 100:.2f}%)")

            if train_labels.count(0) / len(train_labels) * 100 < 40 or train_labels.count(0) / len(train_labels) * 100 > 60:
                print("Applying overfitting to handle class imbalance.")
                train_texts, train_labels = oversample_data(train_texts, train_labels)

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
    
    print(f"\nPreparing dataset with context for {dataset_name}.")
    
    df = pd.read_csv(dataset_name)

    global_index_for_context_dataset = 0

    if 'parent_comment' in df.columns:
        df = df[['comment', 'parent_comment', 'label']]
        df.columns = ['text', 'context', 'label']

        texts = df['text'].tolist()
        contexts = df['context'].tolist()
        labels = df['label'].tolist()

        groups = list(df.groupby(np.arange(len(df)) // 50000))

        for group_index, (_, group) in enumerate(groups):
            group_texts = group['text'].tolist()
            group_contexts = group['context'].tolist()
            group_labels = group['label'].tolist()

            print(f"Group {group_index} (Train) - Label 0: {group_labels.count(0)} samples ({group_labels.count(0) / len(group_labels) * 100:.2f}%)")
            print(f"Group {group_index} (Train) - Label 1: {group_labels.count(1)} samples ({group_labels.count(1) / len(group_labels) * 100:.2f}%)")

            if group_labels.count(0) / len(group_labels) * 100 < 40 or group_labels.count(0) / len(group_labels) * 100 > 60:
                print(f"Applying oversampling to handle class imbalance in group {group_index}.")
                group_texts, group_contexts, group_labels = oversample_data(group_texts, group_labels, group_contexts)
                assert len(group_texts) == len(group_contexts) == len(group_labels), "All arrays must be of the same length after oversampling"

            df_balanced = pd.DataFrame({'text': group_texts, 'context': group_contexts, 'label': group_labels})
            df_balanced.to_csv(f'datasets/sarcasm_dataset_with_context_{global_index_for_context_dataset}.csv', index=False)
            global_index_for_context_dataset += 1
            
    else:
        

if __name__ == "__main__":
    prepare_data()
    prepare_data_with_context('train-balanced-sarcasm.csv')
