from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer 
from langchain_chroma import Chroma
"""
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
"""

def load_docs(dir= 'docs'):
    print("Loading documents from the Docs folder...")
    loader = DirectoryLoader(dir, glob= '**/*.txt', loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from the Docs folder.")
    return documents

def split_documents(documents, chunk_size=800, overlap=0 ):

    text_splitter = CharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= overlap,
    )
    chunks = text_splitter.split_documents(documents)

    # Displaying the chunks and their content
    print(f"Total number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):  # Displaying only the first 5 chunks for brevity
        print('~'*80)
        print(f'--- Chunk {i+1}: ---')
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")
    
    print(f'And other {len(chunks) - 5} chunks remaining...')
    return chunks

def encode_chunks(chunks):
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = model.encode(chunks)
    print('~'*80)
    for chunk, embedding in zip(chunks[:5], embeddings[:5]):
        print('~'*80)
        print('--- Chunk {i+1}: ---')
        print(f"chunk: {chunk}")
        print(f"embedding: {embedding}")
        print(f"dimensions: {embedding.shape}")
    print(f'And other {len(chunks) - 5} chunk/embedding pairs remaining...')
    return embeddings

def create_vector_store(chunks, persist_directory= "db/chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create ChromaDB vectore Store
    print("--- creating vectorestore ---")
    vectorstore = Chroma.from_documents(
        documents= chunks,
        embedding= embedding_model,
        persist_directory= persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print('--- Finished Creating vectorstore ---')
    print(f'--- Vectorstore created and saved at {persist_directory} ---')
    return vectorstore

def main():
    print("Starting the ingestion pipeline...")
    # Loading files from the Docs folder
    documents = load_docs('docs')
    # Chunking documetnts content
    chunks = split_documents(documents)
    # Embedding the chunks and storing in a vector database
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
