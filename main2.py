import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")

    vector_store = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings
    )

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, please output "Sorry i don't know", don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks, Terima Kasih, Kamsahamnida!" at the end of your answer.

    Context:
    ```
    {context}
    ```

    Question: `{question}`

    Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke("Pekerjaan cybersecurity yang paling menjanjikan?")
    print(result)
    