import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from datasets import load_dataset

# Load and prepare the CoNaLa dataset
dataset = load_dataset("neulab/conala", "curated")

# Print out the first example (for verification)
print(dataset['train'][0])

# Load the tokenizer and add a padding token if it's missing
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['intent'], examples['snippet'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
train_size = 0.8
train_dataset, val_dataset = tokenized_datasets['train'].train_test_split(test_size=1-train_size).values()

# Create DataLoader with DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)  # Reduced batch size
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)  # Reduced batch size

# Define the model
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# Set up training arguments
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Note the change from evaluation_strategy to eval_strategy
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,  # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
    fp16=True,  # Enable mixed precision training
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
