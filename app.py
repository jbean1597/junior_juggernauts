import streamlit as st




st.title('Welcome to the Computer Science Learning tool.')


question = st.text_input('Please enter your question here: ')


if st.button('Submit'):
    if question == "":
        output = 'Please ask a valid question'
    else:
        output = f"The answer to {question} is blah blah blah"

    st.write(output)