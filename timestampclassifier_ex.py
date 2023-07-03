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
    print('sec XV - Garcia de Resende')
    print(svc_model_tfidf.predict(vectorizer.transform([
    '''
    Senhoras não hajais medo
    não receeis fazer bem
    tende o coração mui quedo
    e vossas mercês verão cedo
    quão grandes bens do bem vem.
    Não torvem vosso sentido
    as cousas qu'haveis ouvido
    porqu'é lei de deos d'amor
    bem, vertude nem primor
    nunca jamais ser perdido.
    Por verdes o galardão
    que do amor recebeu
    porque por ele morreu
    nestas trovas saberão
    o que ganhou ou perdeu.
    Não perdeu senão a vida
    que pudera ser perdida
    sem na ninguém conhecer
    e ganhou por bem querer
    ser sua morte tão sentida.
    '''])))

    print('sec XVI - Duarte Dias')
    print(svc_model_tfidf.predict(vectorizer.transform([
    '''
    Logrando estou, senhora, um brando riso
    Daquela doce boca e vista pura
    Ua rara beleza, ua figura
    Que me faz ver na terra o paraiso.
    Logrando estou o delicado aviso,
    A luz que torna dia a noite escura
    O claro sol, a nova fermosura
    Que abrasa o pensamento e perde o siso.
    Logrando estou a graça peregrina,
    A trança dos cabelos de ouro fino
    Que em diferentes laços me arremata:
    E quem logra tisouro tão divino
    Claramente delira e desatina,
    Se dele se apartar por ouro ou prata.
    '''])))

    print('julho de 1843 - Gonçalves Dias')
    print(svc_model_tfidf.predict(vectorizer.transform([
    '''
    Minha terra tem palmeiras
    Onde canta o Sabiá,
    As aves, que aqui gorjeiam,
    Não gorjeiam como lá.
    Nosso céu tem mais estrelas,
    Nossas várzeas têm mais flores,
    Nossos bosques têm mais vida,
    Nossa vida mais amores.
    Em cismar, sozinho, à noite,
    Mais prazer encontro eu lá;
    Minha terra tem palmeiras,
    Onde canta o Sabiá.
    Minha terra tem primores,
    Que tais não encontro eu cá;
    Em cismar – sozinho, à noite –
    Mais prazer encontro eu lá;
    Minha terra tem palmeiras,
    Onde canta o Sabiá.
    Não permita Deus que eu morra,
    Sem que eu volte para lá;
    Sem que desfrute os primores
    Que não encontro por cá;
    Sem qu’inda aviste as palmeiras,
    Onde canta o Sabiá.
    '''])))

    print('julho de 1928 - Carlos Drummond de Andrade')
    print(svc_model_tfidf.predict(vectorizer.transform([
    '''
    No meio do caminho tinha uma pedra
    tinha uma pedra no meio do caminho
    tinha uma pedra
    no meio do caminho tinha uma pedra.
    Nunca me esquecerei desse acontecimento
    na vida de minhas retinas tão fatigadas.
    Nunca me esquecerei que no meio do caminho
    tinha uma pedra
    tinha uma pedra no meio do caminho
    no meio do caminho tinha uma pedra.
    '''])))