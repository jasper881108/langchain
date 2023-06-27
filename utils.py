from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from vectorstores import LAB_FAISS
from embeddings import LabOpenAIEmbeddings, LabHuggingFaceEmbeddings
from tabulate import tabulate

import os
import math
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

def print_tabulate_question_and_answer_openai(question_list, docs_and_scores_list, n=3, k=4):
    import openai
    system_info = "你是國泰世華銀行的阿發, 參考[公開資料]依照信用卡別簡潔和專業的回覆顧客的信用卡優惠[問題], 如果無法獲取答案, 請說 “根據已知訊息無法回復該問題” 或 “沒有足夠的相關訊息”，不允許在答案中加入編造的內容，答案請使用繁體中文。"
    
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

def print_tabulate_question_and_answer_chatglm(question_list, docs_and_scores_list, n=3, k=4):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("chatglm-6b/v2-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("chatglm-6b/v2-int4", trust_remote_code=True, device="cuda:0")
    model = model.eval()

    prompt_info = """你是國泰世華銀行的阿發, 根據上述已知訊息, 簡潔和專業的回答顧客的問題。如果無法獲取答案, 請說 “根據已知訊息無法回復該問題” 或 “沒有足夠的相關訊息”，不允許在答案中加入編造的內容，答案請使用繁體中文。問題是:"""
    
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

        system_info = "\n".join(["""已知信息:"""] + question_and_answer[1:])
        prompt = system_info + "\n\n" + prompt_info + question 
        print(prompt)
        completions, history = model.chat(tokenizer, prompt, history=[], eos_token_id=2, pad_token_id=2)

        ## Print chatbot answer
        tabulate_format.append(question_and_answer+[completions])
        tabulate_format.append(scores+[])

    print(tabulate(tabulate_format, ["Question"] + ["Truth"+str(i+1) for i in range(k)] + ["Chatbot"], tablefmt='fancy_grid', maxcolwidths=15))


def print_tabulate_question_and_answer_vicuna(question_list, docs_and_scores_list, n=3, k=4):
    from transformers import LlamaTokenizer, LlamaForCausalLM
    import torch
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

        prompt_info = "\n".join(["[公開資料]"] + question_and_answer[1:] + ["[問題]", question])
        
        
        tokenizer = AutoTokenizer.from_pretrained("learnanything/llama-7b-huggingface")
        model = AutoModel.from_pretrained("lmsys/vicuna-7b-v1.3").float()
        for step in range(100):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(system_info + prompt_info + tokenizer.eos_token, return_tensors='pt')
            # print(new_user_input_ids)

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(
                bot_input_ids, max_length=2048,
                pad_token_id=tokenizer.eos_token_id,  
                no_repeat_ngram_size=3,       
                do_sample=True, 
                top_k=50, 
                temperature = 0.1
            )
            
            # pretty print last ouput tokens from bot
            completions = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print("AI: {completions}".format())
            
        ## Print chatbot answer
        tabulate_format.append(question_and_answer+[completions])
        tabulate_format.append(scores+[])

    print(tabulate(tabulate_format, ["Question"] + ["Truth"+str(i+1) for i in range(k)] + ["Chatbot"], tablefmt='fancy_grid', maxcolwidths=15))

