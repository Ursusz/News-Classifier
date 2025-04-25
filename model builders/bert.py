import pandas as pd
import torch
import torch.nn.functional as F
from nltk import word_tokenize
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_scheduler)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from nltk.corpus import stopwords
import re

EPOCHS = 2
BATCH_SIZE = 32
MODEL_NAME = "google-bert/bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stop_words = set(stopwords.words('english'))

def preproccesstext(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    text = " ".join(filtered_words)

    text = text.lower()

    return text

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe.fillna("")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['text']

        text = preproccesstext(text)

        label = 0 if row['label'] == 'FAKE' else 1
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

def evaluate_model(model, dataloader, label_key='label'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['input_ids'].to(DEVICE),
                            attention_mask=batch['attention_mask'].to(DEVICE))
            preds = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch[label_key].tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    return accuracy, f1

def train_model(model, dataloader, optimizer, scheduler, label_key='label', epochs=EPOCHS, val_loader=None):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        for batch in loop:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'].to(DEVICE),
                            attention_mask=batch['attention_mask'].to(DEVICE))
            loss = loss_fn(outputs.logits, batch[label_key].to(DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=loss.item())
        if val_loader:
            val_accuracy, val_f1 = evaluate_model(model, val_loader, label_key)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"{label_key}_best_model.pth")
                print(f"Saved best model with F1: {best_val_f1:.4f}")
    if val_loader:
        model.load_state_dict(torch.load(f"{label_key}_best_model.pth"))
    return model

def predict_single_text(text, model, tokenizer, max_len=256):
    model.eval()
    processed_text = preproccesstext(text)
    encoding = tokenizer(processed_text, truncation=True, padding='max_length',
                         max_length=max_len, return_tensors='pt')
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.tolist()[0]

def run_full_pipeline(train_data_path, model):
    df_train = pd.read_csv(train_data_path)
    news_dataset = NewsDataset(df_train, tokenizer)
    train_size = int(0.8 * len(df_train))
    train_dataset, val_dataset = random_split(news_dataset, [train_size, len(df_train) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=len(train_loader) * EPOCHS)
    print("Training classification model...")
    trained_model = train_model(model, train_loader, optimizer, scheduler, 'label', EPOCHS, val_loader)
    print("Training complete.")
    return trained_model

if __name__ == "__main__":
    train_data_path = "../dataset/DataSet.csv"
    trained_model = run_full_pipeline(train_data_path, model)

    input_text = "This is a sample text to be classified."
    predicted_class, probabilities = predict_single_text(input_text, trained_model, tokenizer)

    print(f"Input text: {input_text}")
    print(f"Predicted class: {predicted_class} (0: FAKE, 1: REAL)")
    print(f"Probabilities: {probabilities}")