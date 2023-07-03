import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

"""###Open df"""

df = pd.read_csv('corpus_preprocessado.csv')

"""###Tf-idf / LinearSVC

####Train/test split
"""

X = df['Text']
y = df['Period']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

"""####Vectorize"""

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the LinearSVC model
svc_model_tfidf = LinearSVC()
svc_model_tfidf.fit(X_train_vectorized, y_train)

if __name__ == "__main__":
    while True:
        print(svc_model_tfidf.predict(vectorizer.transform([input()])))