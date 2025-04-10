import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import glob
import os
from transformers import DataCollatorWithPadding

# Force CPU
device = torch.device("cpu")
print("Forcing CPU. MPS will NOT be used.")

# Load the most recent dataset
def load_most_recent_dataset():
    list_of_files = glob.glob('synthetic_dataset_*.json')
    if not list_of_files:
        raise FileNotFoundError("No synthetic dataset files found.")
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load and unpack dataset
dataset = load_most_recent_dataset()
html_texts, labels, summaries = zip(*dataset)
label_map = {"informational": 0, "navigational": 1, "commercial": 2, "transactional": 3}
encoded_labels = [label_map[label] for label in labels]

# Create Hugging Face Dataset
hf_dataset = Dataset.from_dict({
    "text": html_texts,
    "summary": summaries,
    "label": encoded_labels
})

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    combined_text = [f"{html}\n\nSummary: {summary}" for html, summary in zip(examples["text"], examples["summary"])]
    tokenized = tokenizer(combined_text, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=hf_dataset.column_names)

# Split into train/val
train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]

# Load model and force to CPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_cpu",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_cpu",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    no_cuda=True,  # 🔥 THIS IS IMPORTANT: Force no GPU/MPS use
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer (NO device arg!)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./fine_tuned_bert_page_intent")
tokenizer.save_pretrained("./fine_tuned_bert_page_intent")

# Load the model and tokenizer for inference
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert_page_intent")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert_page_intent")

