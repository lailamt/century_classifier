FROM python:3.10.12

WORKDIR /timestamp_classifier

RUN python3 -m pip install -U scikit-learn
RUN python3 -m pip install pandas

COPY . /timestamp_classifier

CMD [ "python3", "timestampclassifier.py", "timestampclassifier_ex.py" ]