from flow.strand_flow import build_flow


def main():
    prompt = input("Ask: ")

    result = build_flow(prompt)

    print("\nFINAL ANSWER:\n")
    print(result)


if __name__ == "__main__":
    main()
