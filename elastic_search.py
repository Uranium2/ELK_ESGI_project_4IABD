import csv
import sys
import os
import subprocess
import time
from elasticsearch import Elasticsearch, RequestsHttpConnection
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
from io import StringIO

from logger import logger_init

project_folder = "E:\\ELK_ESGI_project_4IABD\\"
logstash_folder = "C:\\Users\\taver\\Downloads\\logstash-7.7.0\\"

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu, 
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

def create_dataframe_from_ESjson(elastic_docs):
    docs = pd.DataFrame()
    for num, doc in enumerate(elastic_docs):
        source_data = doc["_source"]
        _id = doc["_id"]
        doc_data = pd.Series(source_data, name = _id)
        docs = docs.append(doc_data)

    logger.info("Python - Remove meta data column in Dataframe")
    docs = docs[['SEX', 'LEN', 'DIA', 'HEI', 'W1', 'W2', 'W3', 'W4', 'RIN']]
    docs = docs.reset_index(drop=True)
    return docs

logger = logger_init()

# configure elasticsearch
config = {
    "host": "127.0.0.1",
    "port": "9200" 
}

logger.info("Elasticsearch - Creating connexion")
logger.info("Elasticsearch - {}".format(config))
es = Elasticsearch([config,], timeout=300)

# ['SEX', 'LEN', 'DIA', 'HEI', 'W1', 'W2', 'W3', 'W4', 'RIN']
request_body = {
    "settings" : {
        "number_of_shards": 5,
        "number_of_replicas": 1
    },

    "mappings": {
        "properties": {
            "SEX": {"type": "keyword"},
            "LEN": {"type": "float"},
            "DIA": {"type": "float"},
            "HEI": {"type": "float"},
            "W1": {"type": "float"},
            "W2": {"type": "float"},
            "W3": {"type": "float"},
            "W4": {"type": "float"},
            "RIN": {"type": "integer"}
        }
    }
}

index="abalone"

logger.info("Elasticsearch - Deleting `{}` index".format(index))
es.indices.delete(index=index, ignore=[400, 404])

logger.info("Elasticsearch - Creating `{}` index".format(index))
logger.info("Elasticsearch - request_body: {}".format(request_body))
es.indices.create(index=index, body=request_body)

logger.info("Logstash - Deleting `NULL` file")
null_path = "./NULL"
if os.path.exists(null_path):
    os.remove(null_path)

logger.info("Logstash - Running csv import")
proc = subprocess.Popen([logstash_folder+ "bin\\logstash.bat", "-f", project_folder + "logstash.conf"], shell=True)

data_size_file = os.path.getsize("./data/{}.csv".format(index))

logger.info("Logstash - Checking if import done")
while True:
    if not os.path.exists(null_path):
        logger.info("Logstash - Waiting for NULL file to be created")
        time.sleep(1)
        continue

    with open(null_path) as f:
        logger.info("Logstash - Waiting for EOF")
        content = f.readlines()
        if not content:
            logger.warning("Logstash - `NULL` file created but empty")
            time.sleep(1)
            continue
        if str(data_size_file) in content[0]:
            break
        time.sleep(1)
        continue

logger.info("Logstash - Reached EOF import")
logger.info("Logstash - Shutting down Logstash")
proc.kill()

index="abalone"
request_body_search = {
    'size' : 10000,
    "query": {
        "match_all": {}
    }
}
logger.info("Elasticsearch - Getting data from {}".format(index))
result = es.search(index=index, body=request_body_search)

logger.info("Python - Transforming {} JSON result to Dataframe".format(index))
elastic_docs = result["hits"]["hits"]
data = create_dataframe_from_ESjson(elastic_docs)

logger.info("Python - Changing dtypes Dataframe")
data = data.astype({'LEN': float, 'DIA': float, 'HEI': float, 'W1': float, 'W2': float, 'W3': float, 'W4': float, 'RIN': int})

logger.info("Machine Learning - Starting")
logger.info("Machine Learning - Convert SEX into One Hot Encoding")
data = pd.get_dummies(data)

logger.info("Machine Learning - data content head 10 :\n{}".format(data.head(10)))

logger.info("Machine Learning - Get Labbel Y")
labels = data["RIN"]
data = data.drop(['RIN'], axis=1)

logger.info("Machine Learning - Split Train and Test")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

logger.info("Machine Learning - Normalize X_train, X_test")
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

logger.info("Machine Learning - Bulding Model")
model = build_model()

tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
logger.info("Machine Learning - Summary Model: {}".format(summary))

EPOCHS = 500
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=1)

[loss, mae] = model.evaluate(X_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae))

test_predictions = model.predict(X_test).flatten()

print(test_predictions)