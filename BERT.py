import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate-Email-Classification\emails.xlsx")  # Ensure your dataset has 'Subject', 'Body', 'Category'
print(df.head())
df["text"] = df["Subject"] + " " + df["Body"]

# departments = {"Medical": 0, "Automobile": 1, "Housing": 2}
# df["label"] = df["Category"].map(departments)

label2id = {'Health': 0, 'Automobile': 1, 'Building': 2}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['Category'].map(label2id)

df = df[["text", "label"]]

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)

# Convert to Hugging Face Dataset
def convert_to_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "label": labels})

train_dataset = convert_to_hf_dataset(train_texts.tolist(), train_labels.tolist())
test_dataset = convert_to_hf_dataset(test_texts.tolist(), test_labels.tolist())

# Define models to train
models = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "albert-base-v2": "albert-base-v2",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "google/electra-base-discriminator": "google/electra-base-discriminator"
}

results = {}



# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define the evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy ": acc}


# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
)


# Initialize the model and trainer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# Get accuracy on the test set
eval_result = trainer.evaluate()
print("Test Accuracy : ", eval_result["eval_accuracy "])
