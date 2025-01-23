from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model_and_tokenizer(model_name="bert-base-uncased"):
    """
    Load the pre-trained BERT model and tokenizer.

    Parameters:
    - model_name (str): Name of the pre-trained BERT model (e.g., "bert-base-uncased").

    Returns:
    - model (BertForSequenceClassification): Pre-trained BERT model for sequence classification.
    - tokenizer (BertTokenizer): Pre-trained tokenizer for the BERT model.
    """
    # Load pre-trained tokenizer and model from Hugging Face Model Hub
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3  # Assuming 3 labels: negative, neutral, positive
    )
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, max_length=128):
    """
    Predict the sentiment of a given text using the pre-trained BERT model.

    Parameters:
    - text (str): The input text for sentiment analysis.
    - model (BertForSequenceClassification): Pre-trained BERT model.
    - tokenizer (BertTokenizer): Tokenizer for the BERT model.
    - max_length (int): Maximum token length for padding/truncation.

    Returns:
    - sentiment (str): Predicted sentiment (negative, neutral, positive).
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # Run the model and get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map class index to sentiment label
    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_mapping.get(predicted_class, "unknown")

    return sentiment
