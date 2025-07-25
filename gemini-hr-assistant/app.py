from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import RetrievalQA
import os

# Step 1: Load documents
docs = []
for file in os.listdir("data"):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join("data", file))
        docs.extend(loader.load())
print("A")
print(docs)
print("B")
# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print("C")
print(chunks)
print("D")
print(len(chunks))
print("E")
# Step 3: Create embeddings using Gemini/VertexAI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Users\swapn\PycharmProjects\AIProject\gemini-hr-assistant\genai-hr-bot-dd17eebdd3fc.json"
print("F")
from vertexai import init
print("G")
init(project="genai-hr-bot", location="us-central1")
print("H")
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")
print("I")
# Step 4: Save embeddings to Chroma
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("D")
# Step 5: Set up Q&A chain using Gemini Chat Model
retriever = vectorstore.as_retriever()
llm = ChatVertexAI(model_name="gemini-pro", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Step 6: Ask a question
query = "How many days of paid leave are employees entitled to?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("Source Document Snippet(s):")
for doc in result["source_documents"]:
    print(doc.page_content)
