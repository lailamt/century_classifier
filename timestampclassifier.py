import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

"""###Open df"""

df = pd.read_csv('corpus_preprocessado.csv')

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

def train_predict(text):
    X = df['Text']
    y = df['Period']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    svc_model_tfidf = LinearSVC()
    svc_model_tfidf.fit(X_train_vectorized, y_train)

    return svc_model_tfidf.predict(vectorizer.transform([text]))[0]

if __name__ == "__main__":
    while True:
        text = input()
        text = preprocess(text)
        print()