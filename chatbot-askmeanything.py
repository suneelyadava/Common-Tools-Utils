import openai
# Set the API key for OpenAI
openai.api_key = "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"


def chatbot():
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "bye":
            print("Chatbot: Goodbye!")
            break

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"{user_input}",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
        )

        print("Chatbot: ", response["choices"][0]["text"])


if __name__ == "__main__":
    chatbot()
