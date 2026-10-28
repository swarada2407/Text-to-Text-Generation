from functools import lru_cache

from flask import Flask, render_template, request
from requests.exceptions import RequestException

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-base"
MAX_QUESTION_LENGTH = 2000


@lru_cache(maxsize=1)
def load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def generate_answer(question):
    import torch

    tokenizer, model = load_model()
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = None
    error = None
    status_code = 200

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Enter a question to get started."
            status_code = 400
        elif len(question) > MAX_QUESTION_LENGTH:
            error = f"Keep your question under {MAX_QUESTION_LENGTH} characters."
            status_code = 400
        else:
            try:
                answer = generate_answer(question)
            except (ImportError, OSError, RequestException, RuntimeError, ValueError):
                app.logger.exception("Unable to load the model or generate an answer")
                error = (
                    "The model could not generate an answer. Check your connection "
                    "and try again."
                )
                status_code = 503

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        error=error,
        max_question_length=MAX_QUESTION_LENGTH,
    ), status_code


if __name__ == "__main__":
    app.run(debug=True)
