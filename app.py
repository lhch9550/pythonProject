import streamlit as st
import pandas as pd
import pickle

import lancedb

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import LanceDB

# 환경 변수에서 API 키를 불러옵니다.
openai_api_key = st.secrets["sk-proj-ltNc1yMvbvwwrxG30aH9T3BlbkFJSNc6aG8VTjdvgmVU4uxa"]

client = OpenAI(api_key=openai_api_key)

with open('coursera.pkl', 'rb') as file:
    course = pickle.load(file)

uri = "sample-course-lancedb"

# Connect to the LanceDB database
db = lancedb.connect(uri)

# Assuming 'course' is already defined
#table = db.create_table("vectorstore", course)

embeddings = OpenAIEmbeddings(
    deployment="SL-document_embedder",
    model="text-embedding-ada-002",
    show_progress_bar=True,
    openai_api_key=openai_api_key)

docsearch = LanceDB(connection=db, embedding=embeddings)


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0,
    api_key=openai_api_key
)

# Define the custom prompt
template = """You are a course recommender system that help users to find course that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, suggest five courses, with a short description of the course and the reason why the user might like it.
Additionally, share the URL of the courses.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

# Streamlit app
st.title("Course Recommender System")

query = st.text_input("Enter your course preference query:")

if st.button("Get Recommendations"):
    if query:
        with st.spinner("Fetching recommendations..."):
            result = qa_chain({"query": query})
            st.success("Recommendations fetched successfully!")

            if 'result' in result:
                st.write(result['result'])
            else:
                st.write("Sorry, I couldn't find any recommendations for your query.")
    else:
        st.write("Please enter a query to get recommendations.")