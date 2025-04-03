import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Streamlit UI
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot using LangChain")

# Ensure memory persists across reruns
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Define prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="This is the conversation history:\n{chat_history}\nUser: {question}\nAI:"
)

# Create chat chain
chat_chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

# Display chat history stored in memory
chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])

for message in chat_history:
    if isinstance(message, dict):  # Ensure correct format
        with st.chat_message("user"):
            st.markdown(message.get("question", ""))
        with st.chat_message("assistant"):
            st.markdown(message.get("response", ""))

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Get response from Gemini
    response = chat_chain.run({"chat_history": chat_history, "question": user_input})

    # Store conversation in memory
    st.session_state.memory.save_context({"question": user_input}, {"response": response})

    # Display the conversation
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
