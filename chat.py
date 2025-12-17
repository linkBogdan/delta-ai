import os
import sys
from llama_cpp import Llama

# ---------------------------------------------
# Suppress output context manager
# ---------------------------------------------
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# ---------------------------------------------
# Configuration
# ---------------------------------------------
MODEL_PATH = "/home/bogdan/delta/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# ---------------------------------------------
# Initialize LLM
# ---------------------------------------------
with SuppressOutput():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        verbose=False
    )

# ---------------------------------------------
# Chat Loop 
# ---------------------------------------------
conversation = f"[INST] {SYSTEM_PROMPT} [/INST]\n"

print("AI: Delta aici.")

while True:
    user = input("You: ").strip()
    if user.lower() in ("exit", "quit"):
        print("AI: Vorbim mai târziu.")
        break

    conversation += f"[INST] {user} [/INST]"

    output = llm(
        prompt=conversation,
        max_tokens=200,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=[
            "[/INST]",
            "User:",
            "USER:",
            "Human:",
            "###"
              ]
    )

    reply = output["choices"][0]["text"].strip()
    print(f"AI: {reply}")

    conversation += reply
