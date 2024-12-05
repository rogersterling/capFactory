from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_5BWbPv_EqBeDwGw8cuSFHZeMDaPMb9hhG5BNzKKmRyWu9PnbqvvC3otUEsbiUGnmWBxe8E")

# Get the assistant.
assistant = pc.assistant.Assistant(
    assistant_name="PitchBot", 
)

# Upload a file.
response = assistant.upload_file(
    file_path="/Users/samgaddis/dev/capFactory/companies.json",
    timeout=None
)
