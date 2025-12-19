import requests
import time
import json

url = "http://127.0.0.1:8080/api/data"
headers = {"Content-Type": "application/json"}


N = 100          # number of times to send
interval_ms = 10  # interval in milliseconds

for i in range(N):
    speed = 0
    payload = {
        "vehicleData": {
            "position": [528.0,12.0, 688.0],
            "finished": False,
            "speed": float(speed),
            "time": 10 * i
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"Request {i+1}: status {response.status_code}, response: {response.text}")
    
    if i < N - 1:  # no need to sleep after the last request
        time.sleep(interval_ms / 1000)  # convert ms to seconds
