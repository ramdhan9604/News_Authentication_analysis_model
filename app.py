import streamlit as st
from joblib import load
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load the pre-trained pipeline model
pipeline = load('text_classification_pipeline.pkl')

# Initialize the stemmer and stopwords list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocessing(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove text within square brackets
    text = re.sub('\\[.*?\\]', '', text)
    
    # Replace non-word characters (excluding whitespace) with spaces
    text = re.sub("\\\\W", " ", text)
    
    # Remove URLs
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    
    # Remove HTML/XML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove newline characters
    text = re.sub('\\n', '', text)
    
    # Remove words containing digits
    text = re.sub('\\w*\\d\\w*', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Join the stemmed words back into a single string
    text = ' '.join(stemmed_words)
    
    return text

# Streamlit app
st.title("News Classification App")
st.write("This app classifies news articles as **Real** or **Fake** based on their content.")

# Text input for the news article
news_text = st.text_area("Enter the news article here:", height=200)

# Button to classify the news
if st.button("Classify"):
    if news_text.strip() == "":
        st.warning("Please enter a news article to classify.")
    else:
        # Preprocess the text
        preprocessed_text = preprocessing(news_text)
        
        # Predict using the pipeline
        prediction = pipeline.predict([preprocessed_text])
        
        # Display the result
        if prediction[0] == 1:
            st.success("The news is: **REAL**")
        else:
            st.error("The news is: **FAKE**")