from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

faiss_index_path= "faiss_index"

def build_vectorstore(pdf_path: str):
    # Load the documents
    loader= PyPDFLoader(pdf_path)
    pages= loader.load()

    splitter= RecursiveCharacterTextSplitter(
        # break document into smaller overlaping chunks for better retrieval    
        chunk_size= 500,
        chunk_overlap=100,
        )
    chunks= splitter.split_documents(pages)

    # Turn chunks into vector
    embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store vector into vector database
    vectorstore= FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_index_path)
    print(f"Indexed {len(chunks)} chunks from {pdf_path} -> saved to {faiss_index_path}")
    return vectorstore

# reload an already saved index instead of rebuilding it from scratch
def load_vectorstore():
     embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
     return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
     
     
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")

        sys.exit(1)
    build_vectorstore(sys.argv[1])



 