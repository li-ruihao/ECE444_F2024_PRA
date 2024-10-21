import time

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path

import csv
import pickle
import json
import matplotlib.pyplot as plt

app = Flask(__name__)


def load_model():
    ###### model loading #####
    loaded_model = None
    with open(app.config["BASE_DIR"].joinpath('basic_classifier.pkl'), 'rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open(app.config["BASE_DIR"].joinpath('count_vectorizer.pkl'), 'rb') as vd:
        vectorizer = pickle.load(vd)
    ###################

    # how to use model to predict
    # prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]

    # output will be 'FAKE' if fake, 'REAL' if real

    return loaded_model, vectorizer


@app.route('/')
def index():
    return "Your Flask App Works! PRA5!"


@app.route('/predict', methods=['POST'])
def perdict_news():
    news_content = request.form['content']

    loaded_model, vectorizer = load_model()

    prediction = loaded_model.predict(vectorizer.transform([news_content]))[0]

    return jsonify({
        'prediction_result': prediction,
        'status': 'success'
    }), 200


@app.route('/test_latency_performance', methods=['GET'])
def test_latency_performance():
    test_client = app.test_client()

    #column 1 is the test name and number
    #column 2 is the content to be predicted
    #column 3 is the prediction result
    #column 4 is the time elapsed
    latency_performance_data = [['Tests', 'Content', 'Result', 'Time']]
    boxplot_data = []

    for i in range(100):
        content = "A goat can fly to the moon"
        start_time = time.time()
        test_client.post(
            "/predict",
            data=dict(content=content)
        )
        elapsed_time = time.time() - start_time

        timed_data = ["Fake Test 1", f'content: {content}', "FAKE", f'REST call Time: {elapsed_time:.6f} seconds']
        latency_performance_data.append(timed_data)
        boxplot_data.append(elapsed_time)

    for i in range(100):
        content = "Earth's water ran out since 2023"
        start_time = time.time()
        test_client.post(
            "/predict",
            data=dict(content=content)
        )
        elapsed_time = time.time() - start_time

        timed_data = ["Fake Test 2", f'content: {content}', "FAKE", f'REST call Time: {elapsed_time:.6f} seconds']
        latency_performance_data.append(timed_data)
        boxplot_data.append(elapsed_time)

    for i in range(100):
        content = "Lin Dan won olympics in the badminton category in years 2008 and 2012"
        start_time = time.time()
        test_client.post(
            "/predict",
            data=dict(content=content)
        )
        elapsed_time = time.time() - start_time

        timed_data = ["Real Test 1", f'content: {content}', "REAL", f'REST call Time: {elapsed_time:.6f} seconds']
        latency_performance_data.append(timed_data)
        boxplot_data.append(elapsed_time)

    for i in range(100):
        content = "Michael Phelps became a multiple time olympic winner today!"
        start_time = time.time()
        test_client.post(
            "/predict",
            data=dict(content=content)
        )
        elapsed_time = time.time() - start_time

        timed_data = ["Real Test 2", f'content: {content}', "REAL", f'REST call Time: {elapsed_time:.6f} seconds']
        latency_performance_data.append(timed_data)
        boxplot_data.append(elapsed_time)

    try:
        with open('perf_latency_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(latency_performance_data)

        plt.title("Boxplot of time data for REST API calls")
        plt.ylabel("Time Values")
        plt.boxplot(boxplot_data)
        plt.savefig("time_boxplot.png")

        return (f'perf_latency_output.csv wrote successfully, there are {len(latency_performance_data)} rows '
                f'in total'), 200
    except:
        return 'Failed to write perf and latency data', 400


if __name__ == '__main__':
    app.config["BASE_DIR"] = Path(__file__).resolve().parent
    app.run(port=5000, debug=True)
