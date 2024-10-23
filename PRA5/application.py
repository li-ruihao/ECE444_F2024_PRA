import io
import time
import zipfile

from flask import Flask, request, jsonify, render_template, send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path

import csv
import pickle
import json
import matplotlib.pyplot as plt

application = Flask(__name__)


def load_model():
    ###### model loading #####
    loaded_model = None
    with open(application.config["BASE_DIR"].joinpath('basic_classifier.pkl'), 'rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open(application.config["BASE_DIR"].joinpath('count_vectorizer.pkl'), 'rb') as vd:
        vectorizer = pickle.load(vd)
    ###################

    # how to use model to predict
    # prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]

    # output will be 'FAKE' if fake, 'REAL' if real

    return loaded_model, vectorizer


@application.route('/')
def index():
    return render_template('index.html'), 200


@application.route('/predict', methods=['POST'])
def perdict_news():
    news_content = request.form['content']

    loaded_model, vectorizer = load_model()

    prediction = loaded_model.predict(vectorizer.transform([news_content]))[0]

    return jsonify({
        'prediction_result': prediction,
        'status': 'success'
    }), 200


@application.route('/test_latency_performance', methods=['GET'])
def test_latency_performance():
    test_client = application.test_client()

    #column 1 is the test name and number
    #column 2 is the content to be predicted
    #column 3 is the prediction result
    #column 4 is the time elapsed
    latency_performance_data = [['Tests', 'Content', 'Result', 'Time']]
    boxplot_data = []

    total_time = 0

    for i in range(100):
        content = "A goat can fly to the moon"
        start_time = time.time()
        test_client.post(
            "/predict",
            data=dict(content=content)
        )
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

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
        total_time += elapsed_time

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
        total_time += elapsed_time

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
        total_time += elapsed_time

        timed_data = ["Real Test 2", f'content: {content}', "REAL", f'REST call Time: {elapsed_time:.6f} seconds']
        latency_performance_data.append(timed_data)
        boxplot_data.append(elapsed_time)

    avg_time = total_time / 400
    latency_performance_data.append([f'Average performance time: {avg_time:.6f}'])
    boxplot_data.append(avg_time)

    try:
        with open('perf_latency_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(latency_performance_data)

        plt.title("Boxplot of time data for REST API calls")
        plt.ylabel("Time Values")
        plt.boxplot(boxplot_data)
        plt.savefig('time_boxplot.png')

        return (f'perf_latency_output.csv wrote successfully, there are {len(latency_performance_data)} rows '
                f'in total'), 200
    except:
        return 'Failed to write perf and latency data', 400


@application.route('/download_perf_files', methods=['GET'])
def download_perf_files():
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(application.config["BASE_DIR"].joinpath('perf_latency_output.csv'), 'perf_latency_output.csv')
        zip_file.write(application.config["BASE_DIR"].joinpath('time_boxplot.png'), 'time_boxplot.png')

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='data.zip'
    ), 200


if __name__ == '__main__':
    application.config["BASE_DIR"] = Path(__file__).resolve().parent
    application.run(port=5000, debug=True)
