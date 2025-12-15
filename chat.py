import subprocess

MODEL_PATH = "/home/bogdan/delta/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()

conversation = SYSTEM_PROMPT + "\n"

print("AI: Hey, I'm here!")

while True:
    user = input("You: ").strip()
    if user.lower() in ["exit", "quit"]:
        print("AI: Talk later.")
        break

    conversation += f"\nUser: {user}\nAI:"
    cmd = [
    "/home/bogdan/llama.cpp/build/bin/llama-cli",
    "-m", MODEL_PATH,
    "-p", conversation,
    "-n", "200",
    "--temp", "0.7",
    "--top-p", "0.9",
    "--repeat-penalty", "1.1",
    "--ctx-size", "4096",
    "--no-display-prompt"
]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    reply = result.stdout.strip()
    print(f"AI: {reply}")
    conversation += reply

    # prevent context from growing forever
    if len(conversation) > 8000:
        conversation = SYSTEM_PROMPT + conversation[-4000:]
