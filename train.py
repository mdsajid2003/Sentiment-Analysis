import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from model.preprocess import clean_tweet
from utils.data_loader import TweetDataset
from model.bert_model import SentimentModel
import os

# Hyperparameters and settings
BATCH_SIZE = 16
EPOCHS = 5  # Number of epochs for training
MAX_LEN = 128
LEARNING_RATE = 5e-5
MODEL_SAVE_PATH = "models/sentiment_model.pt"  # Path to save the trained model
TRAINING_FILE = r"C:\Users\mdsaj\Downloads\twitter_training_with_columns.csv"
VALIDATION_FILE = r"C:\Users\mdsaj\Downloads\twitter_validation.csv"

# Load and preprocess data
train_df, val_df = prepare_datasets(TRAINING_FILE, VALIDATION_FILE, clean_tweet)

# Prepare datasets for PyTorch
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TweetDataset(train_df, tokenizer, max_len=MAX_LEN)
val_dataset = TweetDataset(val_df, tokenizer, max_len=MAX_LEN)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentModel(num_labels=3)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train_model():
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_train_loss/len(train_loader):.4f}")
        validate_model(epoch)

    # Save the model after training
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# Validation
def validate_model(epoch):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    print(f"Validation Loss: {total_val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    train_model()
