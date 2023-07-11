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
    print("Timestamp Classifier")
    #text_input = st.text_input("Enter some text")
    print('sec XV - Garcia de Resende')
    print(predict_text_class("Senhoras não hajais medo não receeis fazer bem tende o coração mui quedo e vossas mercês verão cedo quão grandes bens do bem vem. Não torvem vosso sentido as cousas qu'haveis ouvido porqu'é lei de deos d'amor bem, vertude nem primor nunca jamais ser perdido. Por verdes o galardão que do amor recebeu porque por ele morreu nestas trovas saberão o que ganhou ou perdeu. Não perdeu senão a vida que pudera ser perdida sem na ninguém conhecer e ganhou por bem querer ser sua morte tão sentida."))

    print('sec XVI - Duarte Dias')
    print(predict_text_class("Logrando estou, senhora, um brando riso Daquela doce boca e vista pura Ua rara beleza, ua figura Que me faz ver na terra o paraiso. Logrando estou o delicado aviso, A luz que torna dia a noite escura O claro sol, a nova fermosura Que abrasa o pensamento e perde o siso. Logrando estou a graça peregrina, A trança dos cabelos de ouro fino Que em diferentes laços me arremata: E quem logra tisouro tão divino Claramente delira e desatina, Se dele se apartar por ouro ou prata."))

    print('sec XIX - Gonçalves Dias')
    print(predict_text_class("Minha terra tem palmeiras Onde canta o Sabiá, As aves, que aqui gorjeiam, Não gorjeiam como lá. Nosso céu tem mais estrelas, Nossas várzeas têm mais flores, Nossos bosques têm mais vida, Nossa vida mais amores. Em cismar, sozinho, à noite, Mais prazer encontro eu lá; Minha terra tem palmeiras, Onde canta o Sabiá. Minha terra tem primores, Que tais não encontro eu cá; Em cismar – sozinho, à noite – Mais prazer encontro eu lá; Minha terra tem palmeiras, Onde canta o Sabiá. Não permita Deus que eu morra, Sem que eu volte para lá; Sem que desfrute os primores Que não encontro por cá; Sem qu’inda aviste as palmeiras, Onde canta o Sabiá."))

    print('sec XX - Carlos Drummond de Andrade')
    print(predict_text_class("No meio do caminho tinha uma pedra tinha uma pedra no meio do caminho tinha uma pedra no meio do caminho tinha uma pedra. Nunca me esquecerei desse acontecimento na vida de minhas retinas tão fatigadas. Nunca me esquecerei que no meio do caminho tinha uma pedra tinha uma pedra no meio do caminho no meio do caminho tinha uma pedra."))

if __name__ == "__main__":
    main()
