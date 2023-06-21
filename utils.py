from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from vectorstores import LAB_FAISS
from embeddings import LabOpenAIEmbeddings, LabHuggingFaceEmbeddings
from tabulate import tabulate

import os
import math
import openai
import itertools

def read_and_process_answer_to_langchain_docs(answer_file_path, separator = "\n\n",chunk_size=10, chunk_overlap=0):
    documents = TextLoader(answer_file_path).load()
    text_splitter = CharacterTextSplitter(separator = separator,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

def read_and_process_question_to_list(question_file_path, separator = "\n\n"):
    documents = TextLoader(question_file_path).load()
    question_list = []
    for document in documents:
        questions = document.page_content.split(separator)
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

def initial_or_read_langchain_database(documents, embedding_function, db_persist_directory, renew_vectordb=True):
    if not os.path.exists(db_persist_directory) or renew_vectordb:
        vectordb = LAB_FAISS.from_documents(documents=documents, embedding=embedding_function, consine_sim=True)
        vectordb.save_local(db_persist_directory)
        print("Successfully create and save database")
    else:
        vectordb = LAB_FAISS.load_local(db_persist_directory, embedding_function)

    return vectordb

def messeage_prepare(system_info, prompt_info):
        message = [
            {"role": "system", "content": system_info},
            {"role": "user", "content": prompt_info}
            ]
        return message

def print_tabulate_question_and_answer(question_list, docs_and_scores_list, n=3, k=4):
    system_info = "你是國泰世華銀行的助手, 參考[公開資料]依照信用卡別回覆顧客的信用卡優惠[問題], 答案越精準越好"
    
    tabulate_format = []
    n_question = min(len(question_list), n)
    for idx in range(n_question):
        ## Print table
        question = question_list[idx]
        question_and_answer = [question]
        scores = ["Cosin Similarity"]
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            question_and_answer.append(docs.page_content)
            scores.append(str(round(score,4)))
        

        ## Print chatbot answer
        prompt_info = "\n".join(["[公開資料]"] + question_and_answer[1:] + ["[問題]", question])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messeage_prepare(system_info, prompt_info),
            temperature=0.1,
        )
        completions = response["choices"][0]["message"]["content"]
        
        tabulate_format.append(question_and_answer+[completions])
        tabulate_format.append(scores+[])

    print(tabulate(tabulate_format, ["Question"] + ["Truth"+str(i+1) for i in range(k)] + ["Chatbot"], tablefmt='fancy_grid', maxcolwidths=15))
