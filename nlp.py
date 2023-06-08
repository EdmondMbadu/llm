import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import panel as pn
import tempfile
os.environ["OPENAI_API_KEY"] = "Enter your api Key here"

def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result


print("Questions & Answers with LangChain (PDFs)")
print("============================================")
print(
    "You will be prompted the filename you want to inspect (make sure to have a pdf file on the same directory as the code")
print(
    "You will then be prompted to enter a question at each iteration from the pdf files inserted. Enter 'done' when done. ")
filename = input("What is the filename you need to inspect (e.g state_of_the_union.pdf)?: ")
question = input("Q. What is your question: ")
while not question.startswith("done"):
    result = qa("./" + filename, question, "map_reduce", 2)
    question = input("Q. Another question: ")

print("Finished answering questions for the file" + filename + "...")
