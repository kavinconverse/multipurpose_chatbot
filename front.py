import streamlit as st
import os
from back import *
from streamlit_extras.add_vertical_space import add_vertical_space
import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


st.set_page_config(page_title="GooglePalm two diff Chatbots 🚀")
load_dotenv()
st.title('Multi Purpose Chatbot')
st.sidebar.title('Google PALM Chatbot💬')    
st.sidebar.subheader('Enter your keys')
GOOGLE_API_KEY = st.sidebar.text_input("Google Palm API Key", key="gp_chatbot_api_key", type="password")
if not GOOGLE_API_KEY:
    st.warning("Please add your GPalm API key in the sidebar to continue . You can get a key at "" https://developers.generativeai.google/products/makersuite")
    st.stop()
selected_model = st.sidebar.selectbox('Choose Chatbot model', ['None','PDF_model', 'Agent_model'], key='selected_model',index=None,placeholder='Select Chat method')
if selected_model is None:
    st.sidebar.error('!oops Select any one type None selected ⚠️🚨')

if selected_model == 'PDF_model':    
    st.title('Custom Chatbot for Files') 
    st.markdown('<style>h1{color: Green; text-align: center;}</style>', unsafe_allow_html=True)
    st.header('Upload your Files')
    pdf_file = st.file_uploader('Upload your files📝')
    if pdf_file is not None:
        text_chunks = load_vs(pdf_file) 
        vectorstore = emb_vs(pdf_file,text_chunks)
        
        store_display_clear()
        
        # User-provided prompt
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)  

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_ans(vectorstore,prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
    else:
        st.write('Pdf file not found')                                               
elif selected_model == 'Agent_model':
    st.title('Chatbot with Agent')
    st.markdown('<style>h1{color: purple; text-align: center;}</style>', unsafe_allow_html=True)
    agent = up_to_date()
    
    #for store,display,clearchat
    store_display_clear() 
    
        # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)  
        
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
    




