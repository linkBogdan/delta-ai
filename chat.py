import sys
import os
from llama_cpp import Llama


# -------------------------------
# Silent stdout/stderr context
# -------------------------------
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


# -------------------------------
# Language detection
# -------------------------------
def detect_language(text: str) -> str:
    t = text.lower().strip()

    # Strong Romanian signals
    romanian_chars = "ăâîșț"
    romanian_words = {
        "ba", "da", "nu", "hai", "vorbeste", "romana",
        "ce", "cum", "unde", "de", "pot", "poti",
        "sunt", "este", "am", "ai", "face", "faci"
    }

    if any(c in t for c in romanian_chars):
        return "ro"

    words = set(t.split())
    if words & romanian_words:
        return "ro"

    # Short utterances default to Romanian if ambiguous
    if len(words) <= 3:
        return "ro"

    return "en"



def is_short_romanian(text: str) -> bool:
    return len(text.strip().split()) <= 3 and detect_language(text) == "ro"


# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "/home/bogdan/delta/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()


# -------------------------------
# Model initialization (silent)
# -------------------------------
with SuppressOutput():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        verbose=False
    )


# -------------------------------
# Chat loop
# -------------------------------
conversation = SYSTEM_PROMPT + "\n"

print("AI: Delta here!")

while True:
    user = input("You: ").strip()
    if user.lower() in ("exit", "quit"):
        print("AI: Talk later.")
        break

    lang = detect_language(user)

    # Guard against guessing short Romanian slang
    if is_short_romanian(user):
        conversation += (
            "\n[Note: Short Romanian phrase. "
            "If meaning is unclear or slang, ask for clarification instead of guessing.]\n"
        )

    conversation += (
        f"\n[User language: {lang}]\n"
        f"User: {user}\n"
        f"AI:"
    )

    output = llm(
        prompt=conversation,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nUser:"]
    )

    reply = output["choices"][0]["text"].strip()
    reply = reply.split("\nUser:")[0].strip()

    print(f"AI: {reply}")
    conversation += reply

    # Prevent unbounded context growth
    if len(conversation) > 8000:
        conversation = SYSTEM_PROMPT + conversation[-4000:]
