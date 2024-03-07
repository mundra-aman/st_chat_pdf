from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import ConversationalRetrievalChain

#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback

from streamlit.components.v1 import html
import streamlit.components.v1 as components
from streamlit_chat import message

import os
import sys

#from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
#from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"



if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.set_page_config(layout="wide")
col1, col2 = st.columns([1,2])

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        if prompt:
          docs = knowledge_base.similarity_search(prompt)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=prompt)
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)


load_dotenv()
# Left column: Upload PDF text
col1.header("Upload PDF Text")
col1.header("Ask your PDF 💬")

# upload file
pdf = col1.file_uploader("Upload your PDF", type="pdf")

# extract the text
if pdf is not None:
  pdf_reader = PdfReader(pdf)

  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()

  t1=f"""<font color='white'>{text}</fon>"""
  with col2:
      html(t1, height=400, scrolling=True)
  

  # split into chunks
  #text_splitter = CharacterTextSplitter(
  #  separator="\n",
  #  chunk_size=1000,
  #  chunk_overlap=200,
  #  length_function=len
  #)
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
  chunks = text_splitter.split_text(text)

  # create embeddings
  #embeddings = OpenAIEmbeddings()
  knowledge_base = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())

  # show user input
  st.text_input("Ask a question about your PDF:", key="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")
  st.button("Send", on_click=send_click)

   # col1.write(response)
  if st.session_state.prompts:
    for i in range(len(st.session_state.responses)-1, -1, -1):
        message(st.session_state.responses[i], key=str(i), seed='Milo')
        message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)
