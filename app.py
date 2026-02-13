import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pinecone import Pinecone

app = Flask(__name__)

# --- CONFIGURATION (We will set these in the cloud later) ---
# For local testing, replace these with your actual keys if you want to run it locally.
# On the live server, we will use Environment Variables.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-demo"

# --- SETUP AI & DATABASE ---
# 1. Embeddings: Converts text to numbers (using free HuggingFace model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Pinecone: The Cloud Database
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# 3. LLM: Groq (Llama 3)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY, 
    model_name="llama3-8b-8192", 
    temperature=0.0
)

# Folder to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process PDF
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # Save to Pinecone (Cloud)
            vectorstore.add_documents(texts)
            
            return jsonify({"message": "File processed and memory updated successfully!"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Search Pinecone + Send to Groq
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    response = qa_chain.invoke(user_query)
    
    return jsonify({"answer": response['result']})

if __name__ == '__main__':
    app.run(debug=True)