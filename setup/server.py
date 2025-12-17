from flask import Flask, request, jsonify

from keyboard.emulate import PressKey, W, A, S, D, ReleaseKey

app = Flask(__name__)

@app.route("/api/data", methods=["POST"])
def handle_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    print("Received data:", data)
    
    PressKey(W)
    # ReleaseKey(W)

    return jsonify({"status": "success", "received": data}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
