import os
import torch
import opencc
import openai
import mdtex2html
import chatglm_cpp
import gradio as gr
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from utils import (
    read_and_process_knowledge_to_langchain_docs,
    read_and_make_knowlegde_to_url,
    initial_langchain_embeddings,
    initial_or_read_langchain_database_faiss,
)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

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
    text = text.replace("$", "")
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
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
        knowledge_list = [docs_and_scores[0].page_content for docs_and_scores in docs_and_scores_list]
        knowledge = "\n".join(knowledge_list)
        prompt_info =  "[檢索資料]\n{}\n\n[問題]\n{}".format(knowledge, user_input)
        chatbot.append((parse_text(user_input), ""))
        response = openai.ChatCompletion.create(
            model=modelDrop,
            messages=messeage_prepare(system_info, prompt_info),
            temperature=temperature,
            stream=True,
        )
        partial_message = ""
        for chunk in response:
            if len(chunk['choices'][0]['delta']) != 0:
                partial_message = partial_message + chunk['choices'][0]['delta']['content']
                chatbot[-1] = (parse_text(user_input), s2t.convert(parse_text(partial_message)))
                yield chatbot, [], None, parse_text(knowledge)
        
        appendix=["更多資訊請參照以下網址"]
        original_len=len(appendix)
        for key in knowledge_list:
            try:
                url = knowledge_to_url_dict[key]
                if  url != "None" and url not in appendix:
                    appendix.append(url)
            except:
                pass

        if len(appendix) > original_len:
            partial_message = partial_message+"<br><br>"+ "\n".join(appendix[:(original_len+2)])

        chatbot[-1] = (parse_text(user_input), s2t.convert(parse_text(partial_message)))

        yield chatbot, [], None, parse_text(knowledge)

    else:
        prompt_info = "你是國泰世華的聊天機器人-阿發, 參考[檢索資料]使用中文簡潔和專業的回覆顧客的問題"
        docs_and_scores_list = private_vectordb.similarity_search_with_score([user_input], k=top_k)[0]
        knowledge_list = [docs_and_scores[0].page_content for docs_and_scores in docs_and_scores_list]
        knowledge = "\n".join(knowledge_list)
        prompt =  "{}\n\n[檢索資料]\n{}[Round 1]\n\n問：{}\n\n答：".format(prompt_info, knowledge, user_input)
        chatbot.append((parse_text(user_input), ""))
        response = ""

        for response_piece in model_dict[modelDrop].generate(prompt, temperature=temperature, stream=True):
            response += response_piece
            chatbot[-1] = (parse_text(user_input), s2t.convert(parse_text(response)))
            yield chatbot, [], None, parse_text(knowledge)

        appendix=["更多資訊請參照以下網址"]
        original_len=len(appendix)
        for key in knowledge_list:
            try:
                url = knowledge_to_url_dict[key]
                if  url != "None" and url not in appendix:
                    appendix.append(url)
            except:
                pass

        if len(appendix) > original_len:
            response = response+"<br><br>"+ "\n".join(appendix[:(original_len+2)])
        
        chatbot[-1] = (parse_text(user_input), s2t.convert(parse_text(response)))

        yield chatbot, [], None, parse_text(knowledge)

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None, []

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
openai.api_key_path = "openai_api.txt"
model_kwargs = {'device': 'cuda'}
knowledge_to_url_dict = read_and_make_knowlegde_to_url("data/knowledge.txt", "data/url.txt", separator = '\n')
docs = read_and_process_knowledge_to_langchain_docs("data/knowledge.txt", separator = '\n', chunk_size=1)
private_embedding_function = initial_langchain_embeddings("moka-ai/m3e-base", model_kwargs, False)
public_embedding_function = initial_langchain_embeddings("text-ada-embedding", model_kwargs, True)
private_vectordb = initial_or_read_langchain_database_faiss(docs, private_embedding_function, "vectordb/vectordbPrivate", True)
public_vectordb = initial_or_read_langchain_database_faiss(docs, public_embedding_function, "vectordb/vectordbPublic", True)
s2t = opencc.OpenCC('s2t.json')

sft_model = chatglm_cpp.Pipeline("model/chatglm-sft.bin", dtype="q4_0")
rlhf_model = chatglm_cpp.Pipeline("model/chatglm-rlhf.bin", dtype="q4_0")
dpo_model = chatglm_cpp.Pipeline("model/chatglm-dpo.bin", dtype="q4_0")
model_dict={
    "sft": sft_model,
    "rlhf": rlhf_model,
    "dpo": dpo_model
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
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature", interactive=True)
            top_k = gr.Slider(0, 10, value=5, step=1, label="top_k", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, modelDrop, temperature, top_k, history, past_key_values],
                    [chatbot, history, past_key_values, showHtml], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    modelDrop.change(reset_state, outputs=[chatbot, history, past_key_values, showHtml], show_progress=True)
    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values, showHtml], show_progress=True)

demo.queue(concurrency_count=10).launch(share=True, inbrowser=True)
