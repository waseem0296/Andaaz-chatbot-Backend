
import streamlit as st
import requests
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("Chatbot")

FLASK_BACKEND_URL = "http://localhost:5000/ask"


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask a question based on the website...")

if user_input:
   
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
       
        response = requests.post(FLASK_BACKEND_URL, json={"question": user_input})
        response.raise_for_status()
        result = response.json()

       
        answer = result.get("answer", "Sorry, no answer received.")

    except Exception as e:
        answer = f" Error: {str(e)}"


    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
