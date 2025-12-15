from llama_cpp import Llama

MODEL_PATH = "/home/bogdan/delta/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    verbose=False
)

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

    # prevent unbounded context growth
    if len(conversation) > 8000:
        conversation = SYSTEM_PROMPT + conversation[-4000:]
