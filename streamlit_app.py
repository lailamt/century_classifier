import re
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

def preprocess(text):
    stop_words = set(stopwords.words("portuguese"))
    
    text = text.lower()
    text = re.sub(r'ſ', 's', str(text))
    text = re.sub(r'&', 'e', str(text))
    text = re.sub(r'-[ ]{2,}.*\n[ ]{2,}', '', str(text))
    text = re.sub(r'[ ]{2,}.*\n[ ]{2,}', ' ', str(text))
    text = re.sub(r'[àáâãäå]', 'a', str(text))
    text = re.sub(r'[èéêë]', 'e', str(text))
    text = re.sub(r'[ìíîï]', 'i', str(text))
    text = re.sub(r'[òóôõöø]', 'o', str(text))
    text = re.sub(r'[ùúûü]', 'u', str(text))

    text = re.sub(r'\[.*?\]', '', str(text))
    text = re.sub(r'\(.*?\)', '', str(text))

    text = re.sub(r'\d+', '', str(text))

    pattern = re.compile(r'\s{2,}')
    text = pattern.sub(' ', text).strip()

    text = re.sub(r'[^\w\' ]', '', str(text))
    text = re.sub(r'\n', '', str(text))

    text = [w for w in word_tokenize(text) if w not in stop_words]

    return ' '.join(text)


# Function to predict the class using your pre-trained model
def predict_text_class(text):
    # Add your code here to use the pre-trained model and predict the class of the text
    # Replace the return statement with your actual prediction
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    vectorizer = TfidfVectorizer()

    return loaded_model.predict(vectorizer.transform([text]))[0]

def main():
    st.title("Text Classification")
    text_input = st.text_input("Enter some text")
    submit_button = st.button("Submit")

    if submit_button:
        if text_input:
            text_input = preprocess(text_input)
            prediction = predict_text_class(text_input)
            st.write("Text:", text_input)
            st.write("Prediction:", prediction)
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()