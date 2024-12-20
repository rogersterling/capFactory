import streamlit as st
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

# Initialize Pinecone
pc = Pinecone(api_key='pcsk_5BWbPv_EqBeDwGw8cuSFHZeMDaPMb9hhG5BNzKKmRyWu9PnbqvvC3otUEsbiUGnmWBxe8E')
assistant = pc.assistant.Assistant(assistant_name="PitchBot")

# Set page title
st.title("PitchBot Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the companies?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show loading spinner while processing
    with st.spinner("Thinking..."):
        # Get assistant response
        msg = Message(role="user", content=prompt)
        response = assistant.chat(messages=st.session_state.messages)
        response_text = response.get('message', {}).get('content', '')

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
