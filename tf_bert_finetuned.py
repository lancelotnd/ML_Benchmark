import os
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features

def load_dataset(file_path):
    # Assuming the file is in TSV format with columns [label, sentence1, sentence2]
    df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', names=['label', 'id1', 'id2', 'sentence1', 'sentence2'], skiprows=1 ,on_bad_lines='skip')
    print(df) 
    df = df.dropna(subset=['sentence1', 'sentence2'])
    print(df)
    return df

def encode_examples(df, tokenizer, max_length=512):
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
# Load a pre-trained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Training
epochs = 3
train_steps = len(train_dataset) // 32
valid_steps = len(valid_dataset) // 64

model.fit(train_dataset, epochs=epochs, steps_per_epoch=train_steps, validation_data=valid_dataset, validation_steps=valid_steps)