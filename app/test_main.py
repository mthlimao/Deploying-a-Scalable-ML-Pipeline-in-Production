import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test GET request to "/"
def test_get_greetings():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Greetings": "User"}

# Test POST request to "/inference/" - Case 1 (<=50K)
def test_model_inference_case_1():
    payload = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0.0,
        "capital-loss": 0.0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = client.post("/inference/", json=payload)
    assert response.status_code == 200
    assert response.json() == {'prediction': [' <=50K']}

# Test POST request to "/inference/" - Case 2 (>50K)
def test_model_inference_case_2():
    payload = {
        "age": 28,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 987654,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 50000.0,
        "capital-loss": 0.0,
        "hours-per-week": 50,
        "native-country": "India"
    }
    
    response = client.post("/inference/", json=payload)
    assert response.status_code == 200
    assert response.json() == {'prediction': [' >50K']}
