from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_5BWbPv_EqBeDwGw8cuSFHZeMDaPMb9hhG5BNzKKmRyWu9PnbqvvC3otUEsbiUGnmWBxe8E")

assistant = pc.assistant.create_assistant(
    assistant_name="PitchBot", 
    instructions="Answer directly and succinctly. Do not provide any additional information.", # Description or directive for the assistant to apply to all responses.
    timeout=30 # Maximum seconds to wait for assistant status to become "Ready" before timing out.
)
