from flask import Flask, request, jsonify, render_template, session
from flow.strand_flow import build_flow
import uuid

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

try:
    from memory.db import init_db, save_message, load_conversation, delete_session
    init_db()
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))

    if MEMORY_ENABLED:
        conversation = load_conversation(session_id)
    else:
        conversation = session.get("conversation", [])

    conversation.append({"role": "user", "content": user_message})

    if MEMORY_ENABLED:
        save_message(session_id, "user", user_message)

    try:
        response = build_flow(conversation)
    except Exception:
        response = "Something went wrong. Please try again."

    if MEMORY_ENABLED:
        save_message(session_id, "assistant", response)
    else:
        conversation.append({"role": "assistant", "content": response})
        session["conversation"] = conversation

    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id")
    if MEMORY_ENABLED and session_id:
        delete_session(session_id)
    session["session_id"] = str(uuid.uuid4())
    session["conversation"] = []
    return jsonify({"status": "reset"})


@app.route("/history", methods=["GET"])
def history():
    session_id = session.get("session_id")
    if MEMORY_ENABLED and session_id:
        return jsonify(load_conversation(session_id))
    return jsonify([])


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        threaded=True,
    )
