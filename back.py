from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain.agents import load_tools,initialize_agent,AgentType,Tool
from langchain.chains import llm_math,LLMMathChain
from langchain.embeddings import GooglePalmEmbeddings
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from langchain.chains.question_answering import load_qa_chain
import pickle
import os
import streamlit as st

def load_vs(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200,length_function = len)
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def emb_vs(pdf_file,text_chunks):
    store_name = pdf_file.name[:-4]#name for file
    if os.path.exists(f'{store_name}.pkl'):#check below created file exists else do emb and store
        with open(f'{store_name}.pkl','rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
        vectorstore = FAISS.from_texts(text_chunks,embeddings)  #each time run emb is costly so save and load
        with open(f'{store_name}.pkl','wb') as f:
            pickle.dump(vectorstore,f)        
    return vectorstore

def store_display_clear(): 
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    #clear chat 
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)  
    
#pdf model
def query_ans(vectorstore,query):
    ss_docs = vectorstore.similarity_search(query,k=2)
    llm = GooglePalm(temperature = 0.1)
    chain = load_qa_chain(llm=llm,chain_type='stuff')
    response = chain.run(input_documents = ss_docs,question = query)
    return response  

#agent model
def up_to_date():
    llm= GooglePalm(temperature=0.3)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    )]
    #tools=load_tools(['wikipedia','llm_math_chain'],llm=llm)
    agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True)
    return agent

