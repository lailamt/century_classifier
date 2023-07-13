FROM python:3.10.12

WORKDIR /century_classifier

RUN python3 -m pip install -U scikit-learn
RUN python3 -m pip install pandas
RUN python3 -m pip install nltk

COPY . /century_classifier

CMD [ "python3", "centuryclassifier.py", "centuryclassifier_ex.py" ]