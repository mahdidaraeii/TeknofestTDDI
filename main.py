import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

app = FastAPI()

# Load NER model and tokenizer
ner_model = AutoModelForTokenClassification.from_pretrained("./ner_model")
ner_tokenizer = AutoTokenizer.from_pretrained("./ner_model")

# Load Sentiment Analysis model and tokenizer
sentiment_model = AutoModelForSequenceClassification.from_pretrained("./sen_model")
sentiment_tokenizer = AutoTokenizer.from_pretrained("./sen_model")

# Load the MultiLabelBinarizer object
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Define label map for NER
label_map = {"LABEL_0": "O", "LABEL_1": "B-COMPANY"}

# Map for Turkish sentiment labels
sentiment_mapping = {"positive": "olumlu", "negative": "olumsuz", "neutral": "nötr"}

def ner_predict(text):
    tokens = ner_tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = ner_model(**tokens)
    logits = outputs.logits[0]
    predictions = torch.argmax(logits, dim=-1)

    results = []
    for i, prediction in enumerate(predictions):
        if label_map[f'LABEL_{prediction.item()}'] != 'O':
            results.append({
                'word': ner_tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][i].item()),
                'entity': label_map[f'LABEL_{prediction.item()}'],
                'score': torch.sigmoid(logits[i]).max().item()
            })

    return results

# Function to merge subword tokens
def merge_subword_tokens(results):
    merged_results = []
    current_entity = None
    current_score = 0
    current_word = ""

    for result in results:
        if result['entity'] == 'B-COMPANY':
            if current_entity is None:
                current_entity = 'B-COMPANY'
                current_word = result['word']
                current_score = result['score']
            else:
                if result['word'].startswith("##"):
                    current_word += result['word'][2:]
                else:
                    merged_results.append({
                        'word': current_word,
                        'entity': current_entity,
                        'score': current_score
                    })
                    current_word = result['word']
                    current_score = result['score']
        else:
            if current_entity is not None:
                merged_results.append({
                    'word': current_word,
                    'entity': current_entity,
                    'score': current_score
                })
                current_entity = None
                current_score = 0
                current_word = ""

    if current_entity is not None:
        merged_results.append({
            'word': current_word,
            'entity': current_entity,
            'score': current_score
        })

    return merged_results

# Function to locate original entities
def locate_original_entities(entities, original_text):
    original_entities = []
    for entity in entities:
        start_idx = original_text.lower().find(entity.lower())
        if start_idx != -1:
            original_entities.append(original_text[start_idx:start_idx + len(entity)])
        else:
            original_entities.append(entity)  # Fallback to the original if not found
    return original_entities

# Function to merge consecutive entities
def merge_consecutive_entities(entities, original_text):
    merged_entities = []
    i = 0
    while i < len(entities):
        current_entity = entities[i]
        if i < len(entities) - 1:  # Check if there's a next entity
            next_entity = entities[i + 1]
            combined_entity = f"{current_entity} {next_entity}"
            # Check if combined entity exists in the original text
            if combined_entity in original_text:
                merged_entities.append(combined_entity)
                i += 2  # Skip the next entity since it has been merged
                continue
        merged_entities.append(current_entity)
        i += 1
    return merged_entities

class Item(BaseModel):
    text: str = Field(...,
                      example="""Tefal kettle dış yüzeyi 5 ay sonra çillendi, servise gönderdim ve geri gönderdiler, Hepsiburada ve Trendyol ile iletişime geçtim.""")

def evaluate_sentiment(model, tokenizer, sentence, entities, mlb):
    model.eval()
    results = []

    for entity in entities:
        if entity in ["[CLS]", "[PAD]", "[SEP]"]:  # Skip special tokens
            continue

        inputs = tokenizer(sentence, entity, return_tensors="pt", truncation=True, padding=False)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        predicted_labels = (predictions > 0.5).astype(int)
        sentiments = mlb.inverse_transform(predicted_labels)[0]

        # Map the sentiment to Turkish
        sentiment_turkish = sentiment_mapping[sentiments[0]]

        results.append({"entity": entity, "sentiment": sentiment_turkish})

    return results

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    ner_results = ner_predict(item.text)
    merged_entities = merge_subword_tokens(ner_results)

    # Exclude special tokens from entity list
    entities = [result['word'] for result in merged_entities if
                result['entity'] == 'B-COMPANY' and result['word'] not in ["[CLS]", "[PAD]", "[SEP]"]]

    # Locate original entities in the original text
    original_entities = locate_original_entities(entities, item.text)

    # Merge consecutive entities based on the original text
    final_entities = merge_consecutive_entities(original_entities, item.text)

    sentiment_results = evaluate_sentiment(sentiment_model, sentiment_tokenizer, item.text, final_entities, mlb)

    result = {
        "entity_list": final_entities,
        "results": sentiment_results
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
