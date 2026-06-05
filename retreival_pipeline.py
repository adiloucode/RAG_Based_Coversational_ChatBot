import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()
persist_directory= "db/chroma_db"

def retrieve_relevant_docs(query):
    embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    db = Chroma(
        persist_directory= persist_directory,
        embedding_function = embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    retriever= db.as_retriever(
        #search_type= "similarity_score_threshold",
        search_kwargs={
            "k": 10,
            #"score_threshold": 0.3 # Only retrieve chunks with a similarity score above 0.3
            },
        )

    relevant_docs = retriever.invoke(query)
    return relevant_docs

def formulate_answer(query, relevant_docs, conversation_context = None):
    # This function will use the retrieved relevant documents to formulate an answer to the query.
    # You can use a language model like GPT-3.5 or GPT-4 to generate the answer based on the content of the relevant documents.

    llm = ChatOllama(
        model="mistral:7b",
        temperature=0.7
    )
    system_prompt = "You are a helpful assistant, the brain in a RAG system, your role is to provide information to answer a query based on the provided relevant documents. You should use the content of the relevant documents to formulate a comprehensive and accurate answer to the query. If the relevant documents do not contain enough information to answer the query, you should not answer from any prior knowledge, and just indicate that that no releveant answer could be formulated based on the provided documents. "
    user_prompt = f"Based on the following relevant documents, answer the query: '{query}'\n\n"
    if conversation_context:
        conversation_context = 'Context history of the conversation:\n' + conversation_context + '\n\n'
    else:
        conversation_context = ''
    
    for i, doc in enumerate(relevant_docs):
        user_prompt += f"Document {i+1}:\n{doc.page_content}\n\n"
    prompt = system_prompt + conversation_context + "\n\n" + user_prompt
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    # Example query
    query = "Where did SPaeceX launch its first rocket?"

    #Retrieving relevant docs
    relevant_docs = retrieve_relevant_docs(query)

    # Fetching human readable content from the retrieved docs
    answer = formulate_answer(query, relevant_docs)

    # Displaying the answer
    print("--------"+"Answer:"+ "-"*30)
    print(answer.content)
    print("-"*50)
