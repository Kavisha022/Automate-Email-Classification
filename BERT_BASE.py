import pandas as pd
import torch
import re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Dataset
file_path = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate-Email-Classification\emails.xlsx"
df = pd.read_excel(file_path)

# Clean the text in the 'Subject' and 'Body'
def clean_text(text):
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text.strip()

df["Subject"] = df["Subject"].apply(clean_text)
df["Body"] = df["Body"].apply(clean_text)

# Encode Labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Category"])
df["label"] = df["label"].astype(int) 

print(df[["Category", "label"]].head())   
print(df["label"].dtype) 

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset Class
class EmailDataset(Dataset):
    def __init__(self, subjects, bodies, labels, tokenizer, max_len=256):
        self.subjects = subjects
        self.bodies = bodies
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        text = str(self.subjects[idx]) + " " + str(self.bodies[idx])  # Concatenating Subject and Body
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(label), dtype=torch.long),
        }
    
# Split dataset
train_subjects, val_subjects, train_bodies, val_bodies, train_labels, val_labels = train_test_split(
    df["Subject"].tolist(), df["Body"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Create Dataset & DataLoader
train_dataset = EmailDataset(train_subjects, train_bodies, train_labels, tokenizer)
val_dataset = EmailDataset(val_subjects, val_bodies, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize BERT model from scratch 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df["label"].unique()), config=None)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

# Save model and tokenizer locally
model.save_pretrained("email_classifier_model")
tokenizer.save_pretrained("email_classifier_tokenizer")

# Evaluation Function
def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = total_correct / total_samples

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues")

    # Confusion matrix calculation
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return accuracy



# Calculate and print the validation accuracy
val_accuracy = evaluate(model, val_loader)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

print("Training and evaluation completed successfully!")
