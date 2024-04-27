from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers  import  StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from IPython.display import Markdown as md

#Setup API Key 
f = open(r"C:\Users\HP\Downloads\RAG System-on 'Leave No Context Behind' Paper\geminiai_key.txt")
Google_API_Key =  f.read()

# Create the chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=Google_API_Key, model="gemini-1.5-pro-latest")

#create the Title and Header.
st.title("Retrieval-Augmented Generation")
st.header("Ask some questions from 'Leave No Context Behind' Paper")

#create the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key = Google_API_Key, model = "models/embedding-001")

#set up a connection with chorma for retrival
connection = Chroma(persist_directory="./Chroma_db_rag", embedding_function = embedding_model)

#Converting CHROMA db_connection to Retriever Object 
retriever = connection.as_retriever(search_kwargs ={"k": 5})

#User query 
user_query = st.text_input("Enter your query")

#Chatbot prompt templates 
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext: {Context}\nQuestion: {question}\nAnswer:")
])

#Output parser for chatbot response 
output_parser = StrOutputParser()

#Function to format retrieved documents 
def format_docus(docs): 
    formatted_content = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
    return formatted_content if formatted_content else "No relevant context found"

#RAG chain for chatbot interaction 
rag_chain = (
    {"Context": retriever | format_docus, "question": RunnablePassthrough()}
    |chat_template
    | chat_model
    | output_parser
)

if st.button("Result"):
    if user_query:
        response = rag_chain.invoke(user_query)
        st.write("Answer")
        st.write(response)