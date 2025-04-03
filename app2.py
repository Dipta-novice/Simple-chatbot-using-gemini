import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Set up Streamlit UI
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot using LangChain")

# Initialize Memory inside session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define Prompt Template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous Conversation:\n{chat_history}\nUser: {question}\nAI:"
)

def chat_function(input_text):
    # Retrieve past chat history
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

    # Run the LLM chain
    response = (prompt | llm).invoke({"chat_history": chat_history, "question": input_text})

    # Extract the actual text response, ignoring metadata
    if isinstance(response, dict):
        response_text = response.get("content", "")  # Get only the content if present
    elif hasattr(response, "content"):  # Handle cases where response is an object
        response_text = response.content
    elif isinstance(response, str):
        response_text = response  # If it's already a string, use it
    else:
        response_text = str(response)  # Convert to string as a fallback

    # Store user input and AI response in memory
    st.session_state.memory.save_context({"question": input_text}, {"response": response_text})

    return response_text


# Display chat history
for chat in st.session_state.memory.chat_memory.messages:
    role = "user" if "User:" in chat.content else "assistant"
    with st.chat_message(role):
        st.markdown(chat.content)

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Get response from Gemini
    response = chat_function(user_input)

    # Display the conversation
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
