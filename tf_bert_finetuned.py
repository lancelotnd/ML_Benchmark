import os
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features

def load_dataset(file_path):
    # Assuming the file is in TSV format with columns [label, sentence1, sentence2]
    df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', error_bad_lines=False)
    return df

def encode_examples(df, tokenizer, max_length=128):
    input_ids, token_type_ids, attention_masks, labels = [], [], [], []
    
    for _, row in df.iterrows():
        sentence1, sentence2, label = row['sentence1'], row['sentence2'], row['label']
        encoded_dict = tokenizer.encode_plus(sentence1, sentence2, max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='tf')
        
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append([label])
    
    return tf.data.Dataset.from_tensor_slices(({"input_ids": tf.concat(input_ids, 0),
                                                "token_type_ids": tf.concat(token_type_ids, 0),
                                                "attention_mask": tf.concat(attention_masks, 0)},
                                               tf.constant(labels)))

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Paths to your local files
train_file_path = '/data/MSRP_train.tsv'
dev_file_path = '/data/MSRP_test.tsv'

# Load and encode the datasets
train_df = load_dataset(train_file_path)
dev_df = load_dataset(dev_file_path)
train_dataset = encode_examples(train_df, tokenizer)
valid_dataset = encode_examples(dev_df, tokenizer)

# Prepare the datasets for training
train_dataset = train_dataset.shuffle(100).batch(32).repeat(-1)
valid_dataset = valid_dataset.batch(64)

# (The rest of the training code remains the same)
