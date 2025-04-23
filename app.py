import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.title("Twitter Sentiment Analysis")

    # Load stopwords, model, vectorizer, and scraper only once
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    # User input: either text input or Twitter username
    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            try:
                tweets_data = scraper.get_tweets(username, mode='user', number=5)
                if 'tweets' in tweets_data and tweets_data['tweets']:  # Check if tweets exist
                    for tweet in tweets_data['tweets']:
                        tweet_text = tweet['text']
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.error("No tweets found for this user. Please try another username.")
            except Exception as e:
                st.error("Unable to fetch tweets at the moment. This could be due to service limitations. Please try again later or use the 'Input text' option instead.")
                st.info("Error details: " + str(e))

if __name__ == "__main__":
    main()
