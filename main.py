import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*"
        }
    }
)

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

if not API_KEY:
    print("WARNING: GEMINI_API_KEY is not configured.")

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Gemini Chat API",
        "model": MODEL
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "gemini_configured": bool(API_KEY)
    })


@app.post("/api/chat")
def chat():
    if not client:
        return jsonify({
            "error": "Gemini API key is not configured on the server."
        }), 500

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    messages = data.get("messages")

    if not isinstance(messages, list):
        return jsonify({
            "error": "messages must be an array."
        }), 400

    if not messages:
        return jsonify({
            "error": "Message history is empty."
        }), 400

    # Keep requests reasonably sized.
    messages = messages[-30:]

    contents = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        api_role = "model" if role == "assistant" else "user"

        contents.append(
            types.Content(
                role=api_role,
                parts=[
                    types.Part(
                        text=content
                    )
                ]
            )
        )

    if not contents:
        return jsonify({
            "error": "No valid messages found."
        }), 400

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful AI assistant. "
                    "Give clear, accurate and friendly answers. "
                    "When explaining technical topics, use simple "
                    "step-by-step explanations when useful."
                ),
                temperature=0.7,
                max_output_tokens=2048
            )
        )

        reply = response.text

        if not reply:
            return jsonify({
                "error": "Gemini returned an empty response."
            }), 502

        return jsonify({
            "reply": reply,
            "model": MODEL
        })

    except Exception as e:
        error_text = str(e)

        print("Gemini API error:", error_text)

        return jsonify({
            "error": "Gemini request failed. Please try again.",
            "details": error_text[:500]
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port
    )
