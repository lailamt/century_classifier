import re
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

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
    vecname = 'vectorizer.pickle'
    vectorizer = pickle.load(open(vecname, 'rb'))

    return loaded_model.predict(vectorizer.transform([text]))[0]

def main():
    print("Century Classifier")
    #text_input = st.text_input("Enter some text")
    text_input = input("Insira o texto: ")

    if text_input:
        text_input = preprocess(text_input)
        prediction = predict_text_class(text_input)
        print("Predição: séc.", prediction)
    else:
        print("Favor inserir algum texto.")

if __name__ == "__main__":
    main()