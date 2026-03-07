from flask import Flask, request, jsonify, render_template, session
from flow.strand_flow import build_flow
import uuid

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())


@app.route("/")
def index():
    session["conversation"] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    conversation = session.get("conversation", [])
    conversation.append({"role": "user", "content": user_message})

    try:
        response = build_flow(conversation)
    except Exception as e:
        response = "Something went wrong. Please try again."

    conversation.append({"role": "assistant", "content": response})
    session["conversation"] = conversation

    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    session["conversation"] = []
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        threaded=True,
    )
