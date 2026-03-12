from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from flow.strand_flow import build_flow, build_flow_stream
from flow.vision_flow import build_vision_flow
import uuid
import base64
import requests
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fusionai-secret-2024-stable")

try:
    from memory.db import init_db, save_message, load_conversation, delete_session
    init_db()
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False


def extract_pdf_text(file_bytes):
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text[:10000]
    except Exception:
        return None


def groq_ocr(image_data, image_mime):
    try:
        from config.settings import GROQ_API_KEY
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.2-11b-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime};base64,{image_data}"}
                            },
                            {
                                "type": "text",
                                "text": "Extract ALL text from this image exactly as it appears. Preserve all formatting, numbering, and structure. Return only the extracted text, nothing else."
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
            },
            timeout=15,
        )
        data = response.json()
        if "error" in data:
            return None
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def sambanova_ocr(image_data, image_mime):
    try:
        from config.settings import SAMBANOVA_API_KEY
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "Llama-3.2-11B-Vision-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime};base64,{image_data}"}
                            },
                            {
                                "type": "text",
                                "text": "Extract ALL text from this image exactly as it appears. Preserve all formatting, numbering, and structure. Return only the extracted text, nothing else."
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
            },
            timeout=15,
        )
        data = response.json()
        if "error" in data:
            return None
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def process_image(file_bytes, image_mime):
    image_data_b64 = base64.b64encode(file_bytes).decode("utf-8")
    ocr_text = groq_ocr(image_data_b64, image_mime)
    if ocr_text:
        return ocr_text, None, None
    ocr_text = sambanova_ocr(image_data_b64, image_mime)
    if ocr_text:
        return ocr_text, None, None
    return None, image_data_b64, image_mime


def prepare_request(req):
    """Parse request — returns message, image_data, image_mime, is_vision."""
    image_data = None
    image_mime = None
    is_vision = False

    if req.content_type and "multipart/form-data" in req.content_type:
        user_message = req.form.get("message", "").strip()

        if "image" in req.files:
            file = req.files["image"]
            if file.filename:
                file_bytes = file.read()

                if file.content_type == "application/pdf":
                    pdf_text = extract_pdf_text(file_bytes)
                    if pdf_text:
                        user_message = (
                            f"{user_message}\n\nDocument Content:\n{pdf_text}"
                            if user_message
                            else f"Document Content:\n{pdf_text}"
                        )
                else:
                    ocr_text, img_data, img_mime = process_image(
                        file_bytes, file.content_type)
                    if ocr_text:
                        user_message = (
                            f"{user_message}\n\nExtracted Content:\n{ocr_text}"
                            if user_message
                            else f"Extracted Content:\n{ocr_text}"
                        )
                    else:
                        image_data = img_data
                        image_mime = img_mime
                        is_vision = True
    else:
        data = req.get_json()
        user_message = data.get("message", "").strip()
        image_data = data.get("image_data")
        image_mime = data.get("image_mime")

        if image_data and image_mime:
            if image_mime == "application/pdf":
                file_bytes = base64.b64decode(image_data)
                pdf_text = extract_pdf_text(file_bytes)
                if pdf_text:
                    user_message = (
                        f"{user_message}\n\nDocument Content:\n{pdf_text}"
                        if user_message
                        else f"Document Content:\n{pdf_text}"
                    )
                image_data = None
                image_mime = None

            else:
                file_bytes = base64.b64decode(image_data)
                ocr_text, img_data, img_mime = process_image(
                    file_bytes, image_mime)
                if ocr_text:
                    user_message = (
                        f"{user_message}\n\nExtracted Content:\n{ocr_text}"
                        if user_message
                        else f"Extracted Content:\n{ocr_text}"
                    )
                    image_data = None
                    image_mime = None

                else:
                    image_data = img_data
                    image_mime = img_mime
                    is_vision = True

    return user_message, image_data, image_mime, is_vision


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message, image_data, image_mime, is_vision = prepare_request(request)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    conversation = load_conversation(session_id) if MEMORY_ENABLED else []
    conversation.append({"role": "user", "content": user_message})

    if MEMORY_ENABLED:
        save_message(session_id, "user", user_message)

    try:
        if is_vision:
            response = build_vision_flow(
                conversation,
                image_data=image_data,
                image_mime=image_mime
            )
        else:
            response = build_flow(conversation)
    except Exception:
        response = "Something went wrong. Please try again."

    if MEMORY_ENABLED:
        save_message(session_id, "assistant", response)

    return jsonify({"response": response})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_message, image_data, image_mime, is_vision = prepare_request(request)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    conversation = load_conversation(session_id) if MEMORY_ENABLED else []
    conversation.append({"role": "user", "content": user_message})

    if MEMORY_ENABLED:
        save_message(session_id, "user", user_message)

    full_response = []

    def generate():
        try:
            if is_vision:
                result = build_vision_flow(
                    conversation,
                    image_data=image_data,
                    image_mime=image_mime
                )
                if not result:
                    result = "Something went wrong. Please try again."
                full_response.append(result)
                yield f"data: {result}\n\n"
            else:
                for chunk in build_flow_stream(conversation):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            print(f"Stream error: {e}")
            yield "data: Something went wrong. Please try again.\n\n"
        finally:
            yield "data: [DONE]\n\n"
            complete = "".join(full_response)
            if MEMORY_ENABLED and complete:
                save_message(session_id, "assistant", complete)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id")
    if MEMORY_ENABLED and session_id:
        delete_session(session_id)
    session["session_id"] = str(uuid.uuid4())
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
