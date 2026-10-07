import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="ap-southeast-5")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

SYSTEM_PROMPT = (
    "You are an empathetic, energetic, and practical personal motivation coach. "
    "Your role is to listen actively, offer genuine encouragement, help users break "
    "through mental blocks, and give concrete, actionable advice. Keep your tone warm, "
    "direct, and uplifting. Avoid generic platitudes — be specific and personalized. "
    "When a user shares a goal, celebrate it and help them take the next real step. "
    "When they share a setback, acknowledge their feelings first, then reframe and inspire. "
    "Keep responses focused and conversational — typically 2-4 short paragraphs unless "
    "the user clearly wants more depth."
)

OPENING_MESSAGE = (
    "Hey there! I'm your personal motivation coach, ready to chat about whatever's on your "
    "mind — whether it's a goal you're chasing, a setback you're facing, or just needing a "
    "pep talk. What's going on with you today?"
)


def sanitize_history(history: list) -> list:
    """Ensure history alternates user/assistant and only contains valid roles."""
    sanitized = []
    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            sanitized.append({"role": role, "content": content.strip()})
    return sanitized


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}

    history = sanitize_history(data.get("history", []))
    message = (data.get("message") or "").strip()

    if not message:
        return Response(
            "Please send a message.",
            status=400,
            content_type="text/plain; charset=utf-8",
            headers=CORS_HEADERS,
        )

    # Build full messages list: prior history + new user message
    messages = history + [{"role": "user", "content": message}]

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }

    def generate():
        response = BEDROCK_CLIENT.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
        )
        for event in response["body"]:
            chunk = event.get("chunk")
            if chunk:
                payload = json.loads(chunk["bytes"])
                if payload.get("type") == "content_block_delta":
                    delta = payload.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield text

    return Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
