from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from vectorstores import LAB_LANGCHAIN_FAISS

import os
import math
import argparse

def read_and_process_file_to_langchain_docs(file_path, separator = "\n\n",chunk_size=10, chunk_overlap=0):
    documents = TextLoader(file_path).load()
    text_splitter = CharacterTextSplitter(separator = separator,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

def initial_langchain_embeddings(embeddings_model_name, model_kwargs, public):
    if public:
        os.environ["OPENAI_API_KEY"] = input("Input OPENAI_API_KEY Here:")
        embedding_function = OpenAIEmbeddings()
    else:
        embedding_function = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs)

    return embedding_function 

def initial_or_read_langchain_database(documents, embedding_function, db_persist_directory):
    if not os.path.exists(db_persist_directory):
        vectordb = LAB_LANGCHAIN_FAISS.from_documents(documents=documents, embedding=embedding_function)
        vectordb.save_local(db_persist_directory)
        print("Successfully create and save database")

    else:
        vectordb = LAB_LANGCHAIN_FAISS.load_local(db_persist_directory, embedding_function)

    return vectordb

def normalize_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1 - score / 768

def main():
    ## Model multi-qa-mpnet-base-cos-v1/multi-qa-MiniLM-L6-cos-v1/msmarco-distilbert-cos-v5
    ##       hkunlp/instructor-xl
    ##       shibing624/text2vec-base-chinese, hfl/chinese-roberta-wwm-ext
    embeddings_model_name = "shibing624/text2vec-base-chinese" 
    public = False
    model_kwargs = {'device': 'cpu'}
    answer_file_path = 'answer.txt'
    db_persist_directory = 'vectordbPublic' if public else "vectordbPrivate"
    

    docs = read_and_process_file_to_langchain_docs(answer_file_path, separator = "\n\n", chunk_size=10)
    embedding_function = initial_langchain_embeddings(embeddings_model_name, model_kwargs, public)
    vectordb = initial_or_read_langchain_database(docs, embedding_function, db_persist_directory)
    
    query = input("Q:")
    docs_and_scores = vectordb.similarity_search_with_score(query)
    docs_and_norm_scores = [(doc, normalize_score_fn(score)) for doc, score in docs_and_scores]
    print(docs_and_norm_scores)

    return 0

    
if __name__ == "__main__":
    main()