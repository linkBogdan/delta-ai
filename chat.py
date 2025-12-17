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
    romanian_chars = "ăâîșț"
    romanian_words = {
        "ba", "da", "nu", "hai", "vorbeste", "romana",
        "ce", "cum", "unde", "de", "vreau", "pot", "poti",
        "sunt", "este", "am", "ai", "face", "faci", "tare", "misto"
    }

    # Diacritics
    if any(c in t for c in romanian_chars):
        return "ro"

    words = set(t.split())
    if words & romanian_words:
        return "ro"

    # Short messages default to Romanian
    if len(words) <= 3:
        return "ro"

    return "en"


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
# Conversation memory
# -------------------------------
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("AI: Delta here!")

# -------------------------------
# Chat loop
# -------------------------------
while True:
    user_text = input("You: ").strip()
    if user_text.lower() in ("exit", "quit"):
        print("AI: Talk later.")
        break

    lang = detect_language(user_text)
    conversation.append({"role": "user", "content": user_text})

    # Lower temperature for Romanian to reduce creative errors
    temp = 0.5 if lang == "ro" else 0.7

    # Build prompt per turn
    prompt_text = ""
    for msg in conversation:
        if msg["role"] == "system":
            prompt_text += msg["content"] + "\n"
        elif msg["role"] == "user":
            prompt_text += f"User: {msg['content']}\n"
        else:  # assistant
            prompt_text += f"AI: {msg['content']}\n"

    # Per-turn reminder for friendly casual style
    prompt_text += (
    "[Reminder: Respond casually and friendly, in the same language as user. "
    "Keep responses short and concise (1–2 sentences max). "
    "Do not invent translations or facts. If unsure, ask or admit uncertainty.]\nAI:"
)


    output = llm(
        prompt=prompt_text,
        max_tokens=200,
        temperature=temp,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nUser:", "\n[", "\nNote:"]
    )

    reply = output["choices"][0]["text"].strip()
    reply = reply.split("\nUser:")[0].strip()
    print(f"AI: {reply}")

    conversation.append({"role": "assistant", "content": reply})

    # Trim conversation to last 10 turns plus system prompt to avoid context overflow
    if len(prompt_text) > 8000:
        conversation = [conversation[0]] + conversation[-10:]
