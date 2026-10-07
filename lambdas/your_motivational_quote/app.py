import json
import boto3
from flask import Flask, request, Response

app = Flask(__name__)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="ap-southeast-5")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

SYSTEM_PROMPT = (
    "You are a wise and creative author of motivational quotes, able to channel the voice "
    "of a legendary speaker, sage mentor, or inspiring persona."
)


def build_prompt(mood: str, tone: str, theme: str) -> str:
    theme_line = (
        f"Optional Personal Theme or Keyword: {theme}"
        if theme
        else "Optional Personal Theme or Keyword: (none provided)"
    )
    return f"""Craft one completely original, never-before-seen motivational quote tailored to the following specifications:

Mood or Topic: {mood}
Desired Tone: {tone}
{theme_line}

If a theme or keyword is provided, weave it naturally into the quote. If none is provided, focus purely on the mood and tone. The quote should feel punchy, memorable, and emotionally resonant, matching the requested tone exactly (for example, Tough Love should feel firm and challenging, Calm and Zen should feel peaceful and grounded, Humorous should bring a clever smile, Uplifting should feel bright and energizing).

Format your response exactly like this:

Line 1: the quote text in quotation marks
Line 2: a fictional attribution styled like a persona or pen name (not a real person), such as - The Mountain Sage, - Coach Ironwill, - A Wandering Zen Monk, or similar invented signature that fits the tone and mood
Line 3: blank line
Line 4: one single sentence of practical, actionable advice explaining how someone could apply the wisdom of this quote to daily life today, prefixed with Apply it:

Keep the advice concise, specific, and genuinely useful, avoiding vague platitudes, and do not repeat the quote itself. Do not include any extra commentary, explanation, or preamble beyond these four lines."""


@app.route("/", methods=["OPTIONS"])
@app.route("/quote", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
@app.route("/quote", methods=["POST"])
def generate_quote():
    data = request.get_json(force=True, silent=True) or {}
    mood  = data.get("mood",  "Success")
    tone  = data.get("tone",  "Uplifting")
    theme = data.get("theme", "").strip()

    prompt = build_prompt(mood, tone, theme)

    messages = [{"role": "user", "content": prompt}]

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }

    # Collect full response (API Gateway does not support streaming)
    full_text = ""
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
                    full_text += text

    return Response(
        full_text,
        status=200,
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
