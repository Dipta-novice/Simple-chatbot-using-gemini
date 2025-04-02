import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Define prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="This is the conversation history:\n{chat_history}\nUser: {question}\nAI:"
)

# Create chat chain
chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot using LangChain")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Get response from Gemini
    response = chat_chain.run(user_input)

    # Store in session state
    st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display the conversation
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)

