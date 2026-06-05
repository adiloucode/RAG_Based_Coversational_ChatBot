from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from retreival_pipeline import retrieve_relevant_docs

load_dotenv()
chat_history = []

def reformulate_query_with_context(chat_history, query):
    if chat_history:
        # Retrieve the context to add it to the query's context for the LLM.
        system_prompt = [SystemMessage(content="Based on the following context history of the converstaion, rewrite the new question to be standalone and searchable. Just return the rewritten question.\n\n")]
        messages = system_prompt + chat_history + [HumanMessage(content=query)]
        llm = ChatOllama(
            model="mistral:7b",
            temperature= 0.7
        )
        question = llm.invoke(messages)
        return question.content
    else:
        return query
    
def formulate_answer(query, relevant_docs, conversation_history = None):
    # This function will use the retrieved relevant documents to formulate an answer to the query.
    # You can use a language model like GPT-3.5 or GPT-4 to generate the answer based on the content of the relevant documents.

    llm = ChatOllama(
        model="mistral:7b",
        temperature=1.0
    )
    system_prompt = [SystemMessage(content="You are a helpful assistant, the brain in a RAG system, your role is to provide information to answer a query based on the provided relevant documents. You should use the content of the relevant documents to formulate a comprehensive and accurate answer to the query. If the relevant documents do not contain enough information to answer the query, you should not answer from any prior knowledge, and just indicate that that no releveant answer could be formulated based on the provided documents. Take into consideration the context history of the conversation if provided, to better understand the user's intent and provide a more accurate answer. Always use the relevant documents to answer, do not use any prior knowledge you have, again if the answer is not found in the relevant documents, do not do any inference, just indicate that no relevant answer could be formulated based on the provided documents. \n\n")]

    # Adding conversation History
    if conversation_history:
        conversation_context = [SystemMessage(content='Context history of the conversation:\n')] + conversation_history
    else:
        conversation_context = [SystemMessage(content='this discussion history is empty.\n')]
    
    # Constructing User Prompt
    # Starting with Query
    user_prompt_as_str = f"Based on the following relevant documents, answer the query: '{query}'\n\n"
    # Injecting Relevant Chunks into the prompt
    for i, doc in enumerate(relevant_docs):
        user_prompt_as_str += f"Document {i+1}:\n{doc.page_content}\n\n"
    user_prompt = [HumanMessage(content=user_prompt_as_str)]

    # Constructing the final messages to send to the LLM
    messages = system_prompt + conversation_context + user_prompt
    response = llm.invoke(messages)
    return response

def start_chat_session():
    print("Welcome to the History-Aware Chat Session! Press'Q' to exit.")
    while True:
        # User asks first
        user_input = input("User> ")

        # Check if the user wants to exit the chat session
        if user_input.lower() == 'q':
            print("Exiting chat session. Goodbye!")
            break

        # Get assistant's answer, he knows the context if any available, and perfomrs the retrieval internally .
        question = reformulate_query_with_context(chat_history, user_input)
        print(f"Reformulated Question for Retrieval: {question}")

        relevant_docs = retrieve_relevant_docs(question)
        answer = formulate_answer(question, relevant_docs, conversation_history=chat_history)

        print("Assistant> " + answer.content)

        # Updating chat history with the latest user input and assistant's answer
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer.content))

if __name__ == "__main__":
    start_chat_session()