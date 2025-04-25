import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from nltk import word_tokenize
from nltk.corpus import stopwords

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAPPING = {0: 'FAKE', 1: 'REAL'}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

path_to_saved_model = "app/nlps/bert_base_uncased.pth"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
model.load_state_dict(torch.load(path_to_saved_model, map_location=torch.device(DEVICE)))
model.eval()

stop_words = set(stopwords.words('english'))
def preproccesstext(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
    text = " ".join(filtered_words)
    text = text.lower()
    return text

def predict_bert(text, title, model=model, tokenizer=tokenizer, max_len=MAX_LEN, device=DEVICE, label_mapping=LABEL_MAPPING):
    try:
        combined_text = f"{title} {text}" if title else text
        processed_text = preproccesstext(combined_text)

        encoding = tokenizer(processed_text, truncation=True, padding='max_length',
                             max_length=max_len, return_tensors='pt')

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).tolist()[0]
            predicted_class_index = torch.argmax(logits, dim=1).item()
            predicted_label = label_mapping[predicted_class_index]

        # print(f"Bert: {predicted_label}")
        return predicted_label
    except Exception as e:
        print(f"Error in predict_bert: {e}")
        return None