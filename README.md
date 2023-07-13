# Century Classifier

Century Classifier é um classificador multiclasse de texto que determina a qual período um determinado texto pertence com base em suas características. 

O Century Classifier é limitado a classificação de textos entre os séculos XV e XIX.

Foi construído utilizando representação TF-IDF e LinearSVC. Para o treinamento do classificador foi utilizado o Corpus Histórico do Português Tycho Brahe. Utilize apenas com textos em Língua Portuguesa.

A documentação do corpus pode ser encontrada neste repositório no arquivo [Documentação do corpus](https://github.com/lailamt/century_classifier/blob/main/Documenta.txt)

Este produto é parte da avaliação da disciplina Tópicos em Banco de Dados do programa de Pós-graduação da Universidade Federal da Bahia.

## Streamlit

O classificador também está disponível online através da plataforma Streamlit.io no endereço [Century Classifier](https://centuryclassifier-ic007.streamlit.app), com uma interface bem mais amigável para utilização. Experimente

## Uso
O repositório possui um codespace configurado, caso queira testar o classificador por esse método.

Para utilizar, abra o codespace (ou em sua máquina, caso tenha clonado o repositório para executar localmente) e no terminal digite a seguinte linha para construção do docker:

```terminal
docker build -t century_classifier .
```

Depois de construído, você pode passar um texto por linha de comando. Para isso digite a seguinte linha no terminal e em seguida insira seu texto:

```terminal
docker run --rm -it century_classifier python3 centuryclassifier.py
```

O repositório possui um script que testa o classificador em alguns textos de exemplo de diversos períodos. Para executar insira o seguinte no terminal:

```terminal
docker run --rm -it century_classifier python3 centuryclassifier_ex.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Créditos

Desenvolvido por: Laila Mota.
