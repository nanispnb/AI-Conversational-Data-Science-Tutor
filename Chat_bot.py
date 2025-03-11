import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chat_model = GoogleGenerativeAI(model="models/gemini-2.0-pro-exp", api_key="Your_API_Key_Here")

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
        You are an AI chatbot who is an expert in Data Science, capable of solving any issue related to data science 
        and providing an exact code example for the user's input. You are only a data science tutor chatbot. 
        If someone asks non-data science related queries, politely tell them to ask relevant questions.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

output_parser = StrOutputParser()

memory_buffer = {"history": []}

def get_history_from_buffer(human_input):
    return memory_buffer["history"]

runnable_get_history_from_buffer = RunnableLambda(get_history_from_buffer)
chat_history = runnable_get_history_from_buffer

chain = RunnablePassthrough.assign(
        chat_history=runnable_get_history_from_buffer
    ) | chat_template | chat_model | output_parser

st.set_page_config(page_title="AI Data Science Tutor", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .chat-container {
            max-width: 700px;
            margin: auto;
        }
        .user-message {
            background-color: #1E88E5;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: left;
            color: white;
        }
        .ai-message {
            background-color: #424242;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: left;
            color: white;
        }
        input[type="text"] {
            background-color: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 20px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Conversational Data Science Tutor")
st.write("Ask me anything related to Data Science!")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.chat_input("Type your message:")

if user_input:
    query = {"human_input": user_input}
    response = chain.invoke(query)
    
    memory_buffer["history"].append(HumanMessage(content=query["human_input"]))
    memory_buffer["history"].append(AIMessage(content=response))
    
    st.session_state["chat_history"].append((user_input, response))

st.write("### Chat History")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for human, ai in st.session_state["chat_history"]:
    st.markdown(f'<div class="user-message"><b>User:</b> {human}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-message"><b>AI:</b> {ai}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
