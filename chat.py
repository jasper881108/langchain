import os
import torch
import opencc
import openai
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

def messeage_prepare(system_info, prompt_info):
        message = [
            {"role": "system", "content": system_info},
            {"role": "user", "content": prompt_info}
            ]
        return message

def print_and_save_qka_chatgpt(question_list, docs_and_scores_list, n=3, k=4, threshold=0.8, model="gpt-3.5-turbo",csv_saved_path='data/langchain_chatgpt.csv'):
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

    system_info = "你是國泰世華的聊天機器人-阿發, [公開資料]是由國泰世華提供的。 參考[公開資料]使用中文簡潔和專業的回覆顧客的[問題], 如果答案不在公開資料中, 請說 “對不起, 我所擁有的公開資料中沒有相關資訊, 請您換個問題或將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容。"
    s2t = opencc.OpenCC('s2t.json')

    tabulate_format = []
    for idx in tqdm(range(len(question_list))):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            if score > threshold:
                knowledge.append(docs.page_content)
            else:
                break
        if knowledge == []:
            knowledge.append("無法獲取答案")

        knowledge = "\n".join(knowledge)
        prompt_info = "[公開資料]" + "\n" + knowledge + "\n\n" + "[問題]" + "\n" + question

        response = openai.ChatCompletion.create(
            model=model,
            messages=messeage_prepare(system_info, prompt_info),
            temperature=0.1,
        )
        answer = s2t.convert(response["choices"][0]["message"]["content"])

        ## Save values
        tabulate_format.append([question, knowledge , answer])
        qka_dataframe['Question'].append(question)
        qka_dataframe['Knowledge'].append(knowledge)
        qka_dataframe['Answer'].append(answer)

        if (idx+1) % n == 0:
                print(tabulate(tabulate_format, ["Question"] + ["Knowledge"] + ["Answer"], tablefmt='fancy_grid', maxcolwidths=40))
                tabulate_format = []
                filter_config = {
                                    "Answer":{
                                        r'根據.*?，':"",
                                        r'我所擁有的公開資料':"我本次檢索的資料",
                                        r'\[公開資料\]':"公開資料",
                                        r'公開資料':"網路上的公開資料",
                                    }
                                }
                pd.DataFrame(qka_dataframe).replace(filter_config, regex=True).to_csv(csv_saved_path, header=True, index=False, encoding="utf_8_sig")


def print_and_save_qka_chatglm(question_list, docs_and_scores_list, n=3, k=4, threshold=0.8, csv_saved_path='data/langchain_chatglm.csv'):
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

    prompt_info = "你是國泰世華的聊天機器人-阿發, 根據上述已知訊息使用中文簡潔和專業的回覆顧客問題, 如果無法獲取答案, 請說 “請您將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容。\n問:"
    
    tabulate_format = []
    for idx in tqdm(range(len(question_list))):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            if score > threshold:
                knowledge.append(docs.page_content)
            else:
                break
        if knowledge == []:
            knowledge.append("無法獲取答案")

        knowledge = "\n".join(knowledge)
        prompt = "已知訊息:\n" + knowledge + "\n\n" + prompt_info + question  + "\n答:"
        
        completions, history = model.chat(tokenizer,
                                          prompt,
                                          history=[],
                                          eos_token_id=2,
                                          pad_token_id=2,
                                          temperature=0.1)
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

def print_and_save_qka_vicuna(question_list, docs_and_scores_list, n=3, k=4, threshold=0.8, csv_saved_path='data/langchain_vicuna.csv'):
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

    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
    model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", load_in_4bit=True)
    model = model.eval()
    system_info = "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions."
    prompt_info = "你是國泰世華的聊天機器人-阿發, 參考[公開資料]依照信用卡別簡潔和專業的回覆顧客的問題, 如果無法獲取答案, 請說 “請您將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容，答案請使用繁體中文。"
    
    tabulate_format = []
    for idx in tqdm(range(len(question_list))):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            if score > threshold:
                knowledge.append(docs.page_content)
            else:
                break
        if knowledge == []:
            knowledge.append("無法獲取答案")

        knowledge = "\n".join(knowledge)
        prompt = system_info + prompt_info + "\n\n" + "[公開資料]" + "\n" + knowledge + "\n\n" + "USER:" + question + "\n" + "ASSISTANT:"
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, temperature=0.1, max_length=2048)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        ## Save values
        tabulate_format.append([question, knowledge , answer])
        qka_dataframe['Question'].append(question)
        qka_dataframe['Knowledge'].append(knowledge)
        qka_dataframe['Answer'].append(answer)

        if (idx+1) % n == 0:
            print(tabulate(tabulate_format, ["Question"] + ["Knowledge"] + ["Answer"], tablefmt='fancy_grid', maxcolwidths=40))
            tabulate_format = []
            pd.DataFrame(qka_dataframe).to_csv(csv_saved_path, header=True, index=False, encoding="utf_8_sig")

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

def print_and_save_qka_chatglm_finetune(question_list, docs_and_scores_list, n=3, k=4, threshold=0.8, peft_model="chatglm-sft-lora", csv_saved_path='data/langchain_chatglm.csv'):
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

    
    peft_model_list = [peft_model] if peft_model=="chatglm-sft-lora" else ["chatglm-sft-lora", peft_model]

    s2t = opencc.OpenCC('s2t.json')
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

    for peft_adapter in peft_model_list:
        model = PeftModel.from_pretrained(model, os.path.join("Jasper881108", peft_adapter)).merge_and_unload()

    print("Merged {} peft model".format(len(peft_model_list)))
    model = model.quantize(4).cuda()
    model = model.eval()

    prompt_info = "你是國泰世華的聊天機器人-阿發, 參考[檢索資料]使用中文簡潔和專業的回覆顧客的問題"
    
    tabulate_format = []
    for idx in tqdm(range(len(question_list))):
        ## Print table
        question = question_list[idx]
        knowledge = []
        for i in range(k):
            docs, score = docs_and_scores_list[idx][i]
            if score > threshold:
                knowledge.append(docs.page_content)
            else:
                break
        if knowledge == []:
            knowledge.append("無法獲取答案")

        knowledge = "\n".join(knowledge)
        prompt =  "{}\n\n[檢索資料]\n{}[Round 1]\n\n問：{}\n\n答：".format(prompt_info, knowledge, question)
        
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": 8192, "num_beams": 1, "do_sample": True, "top_p": 0.8,
                      "temperature": 0.1, "logits_processor": logits_processor}
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        answer = s2t.convert(response)

        ## Save values
        tabulate_format.append([question, knowledge , answer])
        qka_dataframe['Question'].append(question)
        qka_dataframe['Knowledge'].append(knowledge)
        qka_dataframe['Answer'].append(answer)

        if (idx+1) % n == 0:
            print(tabulate(tabulate_format, ["Question"] + ["Knowledge"] + ["Answer"], tablefmt='fancy_grid', maxcolwidths=40))
            tabulate_format = []
            pd.DataFrame(qka_dataframe).to_csv(csv_saved_path, header=True, index=False, encoding="utf_8_sig")
