import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma

# Correct API key retrieval for Cohere
os.environ["COHERE_API_KEY"] ="u4amCdXLmpaG8GEGJcjHAILcFb5YneUdMQWpe6sf"

def load_pdfs_from_directory(directory):
    files = os.listdir(directory)
    print("Here are files", files)
   
    new_page_data = []
   
    for file in files:
        print("all files", file)
        if file.endswith('.pdf'):
            pdf_path = os.path.join(directory, file)
            file_path = pdf_path
            modified_path = file_path.replace("./", "").replace("\\", "/")
            print("........", modified_path)
            loader = PyPDFLoader(modified_path)
            pages = loader.load_and_split()
            new_page_data.extend([page.page_content for page in pages])
    print(new_page_data)
    return new_page_data

# Directory where resumes are located
 
resume_directory = "/Users/vanishaagarwal/Downloads/archive.zip.download 2/archive/data/data/abcd"
 
# Load resumes from the directory
resume_contents = load_pdfs_from_directory(resume_directory)
 
print(resume_contents)
 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
 
texts = text_splitter.create_documents(resume_contents)
# print("this is text",texts)
emb = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
db = Chroma.from_documents(texts, emb)
 
model = "llama2"
llm = ChatOllama(model=model,format="json")
 
# Setup the retriever
retriever = MultiQueryRetriever.from_llm(db.as_retriever(), llm, prompt=PromptTemplate(
    input_variables=["question"],
    template="you are an AI language model tool. Answer the question. Question: {question}"
    "we have given you multiple pdfs so please give multiple answers"
    "answer should be in json format"
))
 
template = """Answer the question based on the following context: {context}
Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
 
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
question = input("Enter your question: ")
result = chain.invoke(question)
print(result)