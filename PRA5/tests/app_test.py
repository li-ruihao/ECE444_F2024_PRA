import csv
import os
import pytest
import json
import time

from pathlib import Path
from application import application


@pytest.fixture
def client():
    application.config["TESTING"] = True
    application.config["BASE_DIR"] = Path(__file__).resolve().parent.parent

    yield application.test_client()


def test_index(client):
    response = client.get("/", content_type="html/text")
    assert response.status_code == 200


def test_fake_news_one(client):
    rv = client.post(
        "/predict",
        data=dict(content="A goat can fly to the moon")
    )

    assert rv.status_code == 200
    assert b"FAKE" in rv.data


def test_fake_news_two(client):
    rv = client.post(
        "/predict",
        data=dict(content="Earth's water ran out since 2023")
    )

    assert rv.status_code == 200
    assert b"FAKE" in rv.data


def test_real_news_one(client):
    rv = client.post(
        "/predict",
        data=dict(content="Lin Dan won olympics in the badminton category in years 2008 and 2012")
    )

    assert rv.status_code == 200
    assert b"REAL" in rv.data


def test_real_news_two(client):
    rv = client.post(
        "/predict",
        data=dict(content="Michael Phelps became a multiple time olympic winner today!")
    )

    assert rv.status_code == 200
    assert b"REAL" in rv.data


def test_perf_latency(client):
    rv = client.get('/test_latency_performance', content_type="html/text")

    assert rv.status_code == 200
    assert b'perf_latency_output.csv wrote successfully, there are 402 rows in total' in rv.data


def test_download_perf_files(client):
    rv = client.get('/download_perf_files', content_type="html/text")

    assert rv.status_code == 200
