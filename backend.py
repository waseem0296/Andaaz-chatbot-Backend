
from flask import Flask, request, jsonify
from flask_cors import CORS
from Ai_logic import answer_query
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)  

import traceback  

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        print("Question:", question)
        response = answer_query(question)
        print("Answer:", response)
        return jsonify({"answer": response})
    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()  
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
