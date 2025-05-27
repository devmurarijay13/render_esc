import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

ps = PorterStemmer()

# text preprocessing 
def transform_text(text):
    text = text.lower()                          
    text = nltk.word_tokenize(text)              

    # Remove punctuation and non-alphanumeric
    words = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming
    stemmed_words = [ps.stem(word) for word in words]

    return " ".join(stemmed_words)

# Load pre-trained TF-IDF vectorizer and ML model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📧 Email/SMS Spam Classifier")

# Text input
input_sms = st.text_area("Enter your message here:")

# Predict
if st.button("Predict"):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])

    # 3. Make prediction
    result = model.predict(vector_input)[0]

    # 4. Display result
    st.subheader("Prediction:")
    if result == 1:
        st.error("🚫 This message is **Spam**.")
    else:
        st.success("✅ This message is **Not Spam**.")
