from flow.strand_flow import build_flow


def main():
    conversation = []
    print("FusionAI | Type 'exit' to quit\n")

    while True:
        prompt = input("You: ").strip()

        if not prompt:
            continue

        if prompt.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": prompt})

        print("\n⏳ Thinking...\n")
        result = build_flow(conversation)

        print(f"FusionAI: {result}\n")

        conversation.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
