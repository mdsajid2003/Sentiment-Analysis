import re

def clean_tweet(tweet):
    """
    Clean the tweet text by removing URLs, special characters, and extra spaces.
    """
    if not isinstance(tweet, str):  # Check if the input is not a string
        tweet = str(tweet)
    
    # Remove URLs, special characters, and trim extra spaces
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Remove non-alphabetic characters
    tweet = tweet.strip()  # Remove leading/trailing spaces
    return tweet
