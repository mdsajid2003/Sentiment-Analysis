from flask import Flask, request, render_template
from model.bert_model import predict_sentiment, load_model_and_tokenizer
from twilio.rest import Client

# Initialize Flask app
app = Flask(__name__)

# Load the BERT model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Twilio credentials
ACCOUNT_SID = 'AC1affde58af20a651a4de869474219306'
AUTH_TOKEN = '2095f91cac9585cf4cc4965a66898a51'  # Replace with your actual Auth Token
TWILIO_WHATSAPP_FROM = 'whatsapp:+14155238886'
TWILIO_WHATSAPP_TO = 'whatsapp:+918250538974'  # Default recipient for notifications (can be changed as needed)

@app.route('/')
def home():
    """
    Renders the home page where users can enter text for sentiment analysis.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze the sentiment of a single input text and send the result via WhatsApp if the sentiment is negative.
    """
    text = request.form.get('text', '')  # Get the text from the form

    if not text:
        return render_template('index.html', error="Please enter some text.")

    # Predict sentiment using the pre-trained model
    sentiment = predict_sentiment(text, model, tokenizer)

    # Send the result via WhatsApp only if the sentiment is negative
    whatsapp_status = ""
    if sentiment == 'negative':
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        try:
            message = client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=TWILIO_WHATSAPP_TO,
                body=f"The sentiment of your text is negative.\n\nText: {text}"
            )
            whatsapp_status = f"Message sent successfully! SID: {message.sid}"
        except Exception as e:
            whatsapp_status = f"Failed to send message. Error: {str(e)}"

    # Render the result page with the sentiment and WhatsApp status (if any)
    return render_template(
        'result.html',
        text=text,
        sentiment=sentiment,
        whatsapp_status=whatsapp_status
    )

if __name__ == '__main__':
    app.run(debug=True)
