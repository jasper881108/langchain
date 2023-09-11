import os
import torch
import opencc
import openai
import argparse
import mdtex2html
import gradio as gr
from copy import deepcopy
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from utils import (
    read_and_process_knowledge_to_langchain_docs,
    initial_langchain_embeddings,
    initial_or_read_langchain_database_faiss,
)
from typing import List, Tuple

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

"""Override Chatbot.postprocess"""
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
    
def build_inputs(model, tokenizer, query: str, history: List[Tuple[str, str]] = None):
    prompt = tokenizer.build_prompt(query, history=history)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    return inputs

def build_stream_inputs(model, tokenizer, query: str, history: List[Tuple[str, str]] = None):
    if history:
        prompt = "\n\n[Round {}]\n\n問：{}\n\n答：".format(len(history) + 1, query)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = input_ids[1:]
        inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
    else:
        inputs = tokenizer([query], return_tensors="pt")
    inputs = inputs.to(model.device)
    return inputs

@torch.inference_mode()
def stream_chat(model, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                max_length: int = 8192, do_sample=True, top_p=0.8, num_beams=1, temperature=0.1, logits_processor=None,
                return_past_key_values=False, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p, "num_beams":num_beams,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if past_key_values is None and not return_past_key_values:
        inputs = model.build_inputs(tokenizer, query, history=history)
    else:
        inputs = model.build_stream_inputs(tokenizer, query, history=history)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if model.transformer.pre_seq_len is not None:
            past_length -= model.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    for outputs in model.stream_generate(**inputs, past_key_values=past_key_values,
                                        return_past_key_values=return_past_key_values, **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            response = model.process_response(response)
            new_history = history + [(query, response)]
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        line = line.replace("$", "")
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def messeage_prepare(system_info, prompt_info):
        message = [
            {"role": "system", "content": system_info},
            {"role": "user", "content": prompt_info}
            ]
        return message

def predict(user_input, chatbot, modelDrop, temperature, top_k, history, past_key_values):
    if modelDrop=="gpt-4" or modelDrop=="gpt-3.5-turbo":
        system_info = "你是國泰世華的聊天機器人-阿發, [檢索資料]是由國泰世華提供的。 參考[檢索資料]使用中文簡潔和專業的回覆顧客的[問題], 如果答案不在公開資料中, 請說 “對不起, 我所擁有的公開資料中沒有相關資訊, 請您換個問題或將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容。\n\n"
        docs_and_scores_list = public_vectordb.similarity_search_with_score([user_input], k=top_k)[0]
        knowledge = "\n".join([docs_and_scores[0].page_content for docs_and_scores in docs_and_scores_list])
        prompt_info =  "[檢索資料]\n{}\n\n[問題]\n{}".format(knowledge, user_input)
        chatbot.append((parse_text(user_input), ""))
        response = openai.ChatCompletion.create(
            model=modelDrop,
            messages=messeage_prepare(system_info, prompt_info),
            temperature=temperature,
        )

        chatbot[-1] = (parse_text(user_input), parse_text(response["choices"][0]["message"]["content"]))
                   
        yield chatbot, [], None, parse_text(knowledge)

    else:
        prompt_info = "你是國泰世華的聊天機器人-阿發, 參考[檢索資料]使用中文簡潔和專業的回覆顧客的問題"
        docs_and_scores_list = private_vectordb.similarity_search_with_score([user_input], k=top_k)[0]
        knowledge = "\n".join([docs_and_scores[0].page_content for docs_and_scores in docs_and_scores_list])
        prompt =  "{}\n\n[檢索資料]\n{}[Round 1]\n\n問：{}\n\n答：".format(prompt_info, knowledge, user_input)
        chatbot.append((parse_text(user_input), ""))

        for response, history, past_key_values in stream_chat(model, tokenizer, prompt, history,
                                                            return_past_key_values=True,
                                                            past_key_values=past_key_values,
                                                            temperature=temperature):
            
            chatbot[-1] = (parse_text(user_input), parse_text(response))
            yield chatbot, history, past_key_values, parse_text(knowledge)

        yield chatbot, [], None, parse_text(knowledge)

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None, []

def reset_model(modelDrop):
    global model
    if modelDrop=="gpt-4" or modelDrop=="gpt-3.5-turbo":
        pass
    else:
        peft_model_list = lora_checkpoint_config[modelDrop]
        model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, os.path.join("Jasper881108", peft_model_list[0])).merge_and_unload()
        if peft_model_list == 2:
            model = PeftModel.from_pretrained(model, os.path.join("Jasper881108", peft_model_list[1])).merge_and_unload()
        print("Merged {} peft model".format(len(peft_model_list)))
        model = model.quantize(4).cuda()
    return [], [], None, []

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
openai.api_key_path = "openai_api.txt"
model_kwargs = {'device': 'cuda'}
docs = read_and_process_knowledge_to_langchain_docs("data/knowledge.txt", separator = '\n', chunk_size=128)
private_embedding_function = initial_langchain_embeddings("moka-ai/m3e-base", model_kwargs, False)
public_embedding_function = initial_langchain_embeddings("gpt-3.5-turbo", model_kwargs, True)
private_vectordb = initial_or_read_langchain_database_faiss(docs, private_embedding_function, "vectordb/vectordbPrivate", False)
public_vectordb = initial_or_read_langchain_database_faiss(docs, public_embedding_function, "vectordb/vectordbPublic", False)
s2t = opencc.OpenCC('s2t.json')
lora_checkpoint_config={
    "sft": ["chatglm-sft-lora"],
    "rlhf": ["chatglm-sft-lora", "chatglm-ppo-lora-delta"],
    "dpo": ["chatglm-sft-lora", "chatglm-dpo-lora-delta"]
}

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">LLM X Chatbot 信用卡優惠</h1>""")
    modelDrop = gr.Dropdown(["gpt-4", "gpt-3.5-turbo","sft", "rlhf", "dpo"], label="Model")

    with gr.Row(scale=6):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
        with gr.Column(scale=1):
            showHtml = gr.HTML()

    with gr.Row(scale=2):
        with gr.Column(scale=4):
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            changeBtn = gr.Button("Change Model")
            emptyBtn = gr.Button("Clear History")
            temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature", interactive=True)
            top_k = gr.Slider(0, 10, value=5, step=1, label="top_k", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, modelDrop, temperature, top_k, history, past_key_values],
                    [chatbot, history, past_key_values, showHtml], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values, showHtml], show_progress=True)
    changeBtn.click(reset_model, [modelDrop], [chatbot, history, past_key_values, showHtml], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
