from flask import Flask, request, jsonify

import platform
if platform.system() == "Darwin":
    from keyboard.macos import PressKey, ReleaseKey
elif platform.system() == "Windows":
    from keyboard.windows import PressKey, ReleaseKey
from keyboard.generic import W, A, S, D

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Learning.agent import Agent
from Learning.utils import new_features

app = Flask(__name__)

driver = Agent()

new_press_keys = set()

@app.route("/api/data", methods=["POST"])
def handle_data():
    data = request.get_json()

    if not data :
        return jsonify({"error": "No JSON payload provided"}), 400
    
    # Build and add new features (make it a cool and packed function)
    position = data["vehicleData"]["position"]
    distance_next_turn = new_features.distance_to_next_turn(position)
    data["distance_next_turn"] = distance_next_turn

    # Make it a simple dict with one level of keys/values. At that point every values should be float. WIP
    data = new_features.keep_relevant_features(data)

    ## Clear all pressed keys
    old_pressed_keys = new_press_keys.copy()
    new_press_keys.clear()

    ## Feed agent with data
    driver.feed(data)

    # get action
    keys = driver.get_keys()

    # Eventually update weights
    driver.learn()

    # Press and Release keys accordingly
    for key in keys:
        new_press_keys.add(eval(key))

    for k in new_press_keys - old_pressed_keys:
        print(k)
        PressKey(k)
    for k in old_pressed_keys - new_press_keys:
        ReleaseKey(k)

    return jsonify({"status": "success", "received": data}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
