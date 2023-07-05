from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from vectorstores import LAB_FAISS
from embeddings import LabOpenAIEmbeddings, LabHuggingFaceEmbeddings
from tabulate import tabulate

import os
import math
import itertools
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import opencc
import openai

def read_and_process_knowledge_to_langchain_docs(knowledge_file_path, separator = "\n\n",chunk_size=10, chunk_overlap=0):
    documents = TextLoader(knowledge_file_path).load()
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

def print_and_save_qka_chatgpt(question_list, docs_and_scores_list, n=3, k=4, csv_saved_path='data/langchain_chatgpt.csv'):
    qka_dataframe = {
        "Question":[],
        "Knowledge":[],
        "Answer":[],
    }

    if os.path.exists(csv_saved_path):
        df = pd.read_csv(csv_saved_path)
        question_list = question_list[len(df):]
        docs_and_scores_list = docs_and_scores_list[len(df):]
        qka_dataframe = df.to_dict('list')

    system_info = "你是國泰世華銀行的助手-阿發, 參考[公開資料]依照信用卡別簡潔和專業的回覆顧客的信用卡優惠[問題], 如果無法獲取答案, 請說 “根據已知訊息無法回復該問題” 或 “沒有足夠的相關訊息”，不允許在答案中加入編造的內容，答案請使用繁體中文。"
    
    tabulate_format = []
    for idx in range(len(question_list)):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            knowledge.append(docs.page_content)
        
        knowledge = "\n".join(knowledge)
        prompt_info = "[公開資料]" + "\n" + knowledge + "\n\n" + "[問題]" + "\n" + question

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messeage_prepare(system_info, prompt_info),
            temperature=0.1,
        )
        answer = response["choices"][0]["message"]["content"]
        
        ## Save values
        tabulate_format.append([question, knowledge , answer])
        qka_dataframe['Question'].append(question)
        qka_dataframe['Knowledge'].append(knowledge)
        qka_dataframe['Answer'].append(answer)

        if (idx+1) % n == 0:
                print(tabulate(tabulate_format, ["Question"] + ["Knowledge"] + ["Answer"], tablefmt='fancy_grid', maxcolwidths=40))
                tabulate_format = []
                pd.DataFrame(qka_dataframe).to_csv(csv_saved_path, header=True, index=False, encoding="utf_8_sig")


def print_and_save_qka_chatglm(question_list, docs_and_scores_list, n=3, k=4, csv_saved_path='data/langchain_chatglm.csv'):
    qka_dataframe = {
        "Question":[],
        "Knowledge":[],
        "Answer":[],
    }
    if os.path.exists(csv_saved_path):
        df = pd.read_csv(csv_saved_path)
        question_list = question_list[len(df):]
        docs_and_scores_list = docs_and_scores_list[len(df):]
        qka_dataframe = df.to_dict('list')

    s2t = opencc.OpenCC('s2t.json')
    tokenizer = AutoTokenizer.from_pretrained("chatglm-6b/v2", trust_remote_code=True)
    model = AutoModel.from_pretrained("chatglm-6b/v2", trust_remote_code=True).quantize(4).cuda()
    model = model.eval()

    prompt_info = """你是國泰世華銀行的助手-阿發, 根據上述已知訊息, 簡潔和專業的回答顧客的問題。如果無法獲取答案, 請說 “根據已知訊息無法回復該問題” 或 “沒有足夠的相關訊息”，不允許在答案中加入編造的內容，答案請使用中文。問題是:"""
    
    tabulate_format = []
    for idx in range(len(question_list)):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            knowledge.append(docs.page_content)
        
        knowledge = "\n".join(knowledge)
        prompt = "已知信息:" + "\n" + knowledge + "\n\n" + prompt_info + question 
        
        completions, history = model.chat(tokenizer, prompt, history=[], eos_token_id=2, pad_token_id=2)
        answer = s2t.convert(completions)

        ## Save values
        tabulate_format.append([question, knowledge , answer])
        qka_dataframe['Question'].append(question)
        qka_dataframe['Knowledge'].append(knowledge)
        qka_dataframe['Answer'].append(answer)

        if (idx+1) % n == 0:
            print(tabulate(tabulate_format, ["Question"] + ["Knowledge"] + ["Answer"], tablefmt='fancy_grid', maxcolwidths=40))
            tabulate_format = []
            pd.DataFrame(qka_dataframe).to_csv(csv_saved_path, header=True, index=False, encoding="utf_8_sig")

