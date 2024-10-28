import streamlit as st
import openai
import os
import streamlit as st
import os
from openai import OpenAI
import base64
import json
from urllib.parse import urlparse
import fitz
import openai
import shutil
from PIL import Image
from io import BytesIO
from llama_index.core import VectorStoreIndex,Document,SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.tools.function_tool import FunctionTool
st.title("pOLICY BOT")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.title("Upload pdf files")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
openai.api_key = st.secrets["api_key"]
if uploaded_files:
   
      
    for filename in os.listdir("uploaded_pdf"):
            file_path = os.path.join("uploaded_pdf", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    for uploaded_file in uploaded_files:
        save_path = os.path.join("uploaded_pdf",uploaded_file.name)
        with open(save_path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File '{uploaded_file.name}' saved sucessfully")
    file_saved = True

documents=SimpleDirectoryReader("uploaded_pdf").load_data()
pc = Pinecone(api_key="264040b3-b298-4918-9d56-b31134d5ba48")
pinecone_index = pc.Index("policy")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)   
Settings.chunk_size = 512

settings = Settings.embed_model = OpenAIEmbedding()
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store,settings=settings)

query_engine = RetrieverQueryEngine.from_args(
retriever=index.as_retriever(similarity_top_k=10),
node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    # text_qa_template=custom_query_prompt,
    # refine_template=custom_refine_prompt,
llm=OpenAI(model="gpt-4", max_tokens=600, temperature=0.0),
   )
query_engine_tool = QueryEngineTool(
query_engine=query_engine,
metadata=ToolMetadata(
        name="Policy_insurance",  
        description="Helpful for resolving client query"
    ),

)   
llm = OpenAI(model="gpt-4o-2024-08-06", temperature=0)
agent = ReActAgent.from_tools([query_engine_tool], llm=llm, verbose=True)
 


if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = agent.chat(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
