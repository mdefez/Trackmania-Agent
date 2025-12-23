## Simple server that listens on /command a json indicating what keys to press or release and acts accordingly

from flask import Flask, request, jsonify
from trackmania.keyboard.windows import PressKey, ReleaseKey
from trackmania.keyboard.generic import W, A, S, D, Backspace
import json

app = Flask(__name__)

# Track currently pressed keys
pressed_keys = set()

@app.route("/command", methods=["POST"])
def handle_command():
    """
    Expects JSON payload with structure:
    {
        "press": ["W", "A"],      # Keys to press
        "release": ["S", "D"]     # Keys to release
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        
        # Get keys to press and release
        keys_to_press = data.get("press", [])
        keys_to_release = data.get("release", [])
        
        # Convert string keys to actual key objects
        key_map = {"W": W, "A": A, "S": S, "D": D, "Backspace": Backspace}
        
        # Press keys
        for key_name in keys_to_press:
            key_obj = key_map[key_name]
            pressed_keys.add(key_obj)
            PressKey(key_obj)
            print(f"Pressed: {key_name}")
        
        # Release keys
        for key_name in keys_to_release:
            key_obj = key_map[key_name]
            ReleaseKey(key_obj)
            pressed_keys.remove(key_obj)
            print(f"Released: {key_name}")
        
        return jsonify({
            "status": "success",
            "pressed_keys": list(data.get("press", [])),
            "released_keys": list(data.get("release", []))
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    """Get currently pressed keys"""
    return jsonify({"currently_pressed": [k.name for k in pressed_keys]}), 200

@app.route("/reset", methods=["POST"])
def reset_all():
    """Release all currently pressed keys"""
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    pressed_keys.clear()
    PressKey(Backspace)
    return jsonify({"status": "all keys released"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
