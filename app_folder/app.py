from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.llms import OpenAI
import os
import openai
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(temperature = .9)
st.title('Welcome to the Computer Science Learning tool.')



question = st.text_input('Please enter your question here: ')

if st.button('Submit'):
    if question == "":
        output = 'Please ask a valid question'
    else:
        answer = llm(question)

    st.write(answer)