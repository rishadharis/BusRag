import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

if __name__ == "__main__":
    print('Ingesting...')
    loader = TextLoader('./articles/medium2.txt', encoding='utf-8')
    document = loader.load()
    # print(document)

    print('Splitting...')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-ada-002')

    print('Storing...')
    PineconeVectorStore.from_documents(texts, embeddings, index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key)
    print('Done!')
