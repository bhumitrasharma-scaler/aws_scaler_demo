import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI, OpenAIError

app = Flask(__name__)
load_dotenv(Path(__file__).resolve().with_name(".env"))


@app.route("/", methods=["GET", "POST"])
def index():
    title = ""
    blurb = None
    error = None

    if request.method == "POST":
        title = request.form.get("title", "").strip()

        if not title:
            error = "Please enter a book title."
        elif not os.environ.get("GROQ_API_KEY"):
            error = "GROQ_API_KEY is not configured in .env or the Scaler environment."
        else:
            try:
                client = OpenAI(
                    api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1"
                )
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Write one punchy, one-line bookshop "
                                "shelf-talker blurb for the given title. "
                                "Under 20 words. No surrounding quotes."
                            )
                        },
                        {"role": "user", "content": title}
                    ],
                    max_completion_tokens=300,
                    reasoning_effort="low"
                )

                if not response.choices:
                    app.logger.error("Groq returned no choices: %s", response.model_dump())
                    error = "The AI returned no blurb. Please try again."
                else:
                    choice = response.choices[0]
                    blurb = (choice.message.content or "").strip()

                    if not blurb:
                        app.logger.error(
                            "Empty Groq response: finish_reason=%s, response=%s",
                            choice.finish_reason,
                            response.model_dump()
                        )
                        error = "The AI returned no visible text. Please try again."
            except OpenAIError:
                app.logger.exception("Groq API request failed")
                error = "Blurb generation failed. Check the server logs and try again."

    return render_template(
        "index.html",
        title=title,
        blurb=blurb,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
