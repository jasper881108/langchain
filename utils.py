import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from vectorstores import LAB_FAISS
from embeddings import LabOpenAIEmbeddings, LabHuggingFaceEmbeddings

def read_and_process_knowledge_to_langchain_docs(knowledge_file_path, separator = "\n",chunk_size=64, chunk_overlap=0):
    documents = TextLoader(knowledge_file_path).load()
    text_splitter = CharacterTextSplitter(separator = separator,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

def read_and_process_question_to_list(question_file_path, separator = "\n"):
    documents = TextLoader(question_file_path).load()
    question_list = []
    for document in documents:
        questions = document.page_content.split(separator)
        if questions != "":
            question_list.extend(questions)

    return question_list

def initial_langchain_embeddings(embeddings_model_name, model_kwargs, public):
    if public:
        if not os.environ["OPENAI_API_KEY"]:
            os.environ["OPENAI_API_KEY"] = input("Input OPENAI_API_KEY Here:")
        embedding_function = LabOpenAIEmbeddings()
    else:
        embedding_function = LabHuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs)

    return embedding_function 

def initial_or_read_langchain_database_faiss(documents, embedding_function, db_persist_directory, renew_vectordb=True):
    if not os.path.exists(db_persist_directory) or renew_vectordb:
        vectordb = LAB_FAISS.from_documents(documents=documents, embedding=embedding_function, consine_sim=True)
        vectordb.save_local(db_persist_directory)
        print("Successfully create and save database")
    else:
        vectordb = LAB_FAISS.load_local(db_persist_directory, embedding_function)

    return vectordb

