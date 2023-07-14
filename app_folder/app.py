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
answer = llm(question)

if st.button('Submit'):
    if question == "":
        output = 'Please ask a valid question'
    else:
        output = f"The answer to {question} is blah blah blah"

    st.write(answer)