import requests

url = "http://localhost:9696/predict"

ride = {
    "PULocationID": 10, 
    "DOLocationID": 50,
    "trip_distance": 40
}

response = requests.post(url, json=ride)
print(response.json())