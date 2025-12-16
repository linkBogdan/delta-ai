import os
import sys
from llama_cpp import Llama

#---------------------------------------------
# Suppress output context manager
#---------------------------------------------
class SupressOutput:
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

#---------------------------------------------
# Configuration
#---------------------------------------------
MODEL_PATH = "/home/bogdan/delta/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()

#---------------------------------------------
# Initialize LLM
#---------------------------------------------
with SupressOutput():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        verbose=False
    )
    
#---------------------------------------------
# Chat Loop
#---------------------------------------------

conversation = SYSTEM_PROMPT + "\n"

print("AI: Delta here!")

while True:
    user = input("You: ").strip()
    if user.lower() in ("exit", "quit"):
        print("AI: Talk later.")
        break

    conversation += f"\nUser: {user}\nAI:"

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

    if len(conversation) > 8000:
        conversation = SYSTEM_PROMPT + conversation[-4000:]
