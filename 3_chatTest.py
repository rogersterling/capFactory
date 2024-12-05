from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

pc = Pinecone(api_key='pcsk_5BWbPv_EqBeDwGw8cuSFHZeMDaPMb9hhG5BNzKKmRyWu9PnbqvvC3otUEsbiUGnmWBxe8E')
assistant = pc.assistant.Assistant(assistant_name="PitchBot")

msg = Message(role="user", content="Which companies deal with bikes?")
resp = assistant.chat(messages=[msg])

print(resp)
