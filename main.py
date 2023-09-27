import os
import openai
import argparse
from utils import (
    read_and_process_knowledge_to_langchain_docs,
    read_and_process_question_to_list,
    initial_langchain_embeddings,
    initial_or_read_langchain_database_faiss,
)
from chat import (
    print_and_save_qka_chatgpt,
    print_and_save_qka_chatglm,
    print_and_save_qka_vicuna,
    print_and_save_qka_chatglm_finetune,
)
from functools import partial

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
    openai.api_key_path = "openai_api.txt"
    model_kwargs = {'device': 'cuda' if args.cuda else 'cpu'}
    db_persist_name = 'vectordbPublic' if args.public_embedding else 'vectordbPrivate'
    db_persist_directory = os.path.join(args.db_persist_directory, db_persist_name)

    docs = read_and_process_knowledge_to_langchain_docs(args.knowledge_file_path, separator = '\n', chunk_size=args.knowledge_len)
    embedding_function = initial_langchain_embeddings(args.embeddings_model_name, model_kwargs, args.public_embedding)
    vectordb = initial_or_read_langchain_database_faiss(docs, embedding_function, db_persist_directory, args.renew_vectordb)
    question_list = read_and_process_question_to_list(args.question_file_path, separator = '<sep>\n')
    docs_and_scores_list = vectordb.similarity_search_with_score(question_list, k=args.top_k)

    function_dict = {
        'gpt-3.5-turbo':partial(print_and_save_qka_chatgpt,model=args.model),
        'gpt-4':partial(print_and_save_qka_chatgpt,model=args.model),
        'chatglm':print_and_save_qka_chatglm,
        'vicuna':print_and_save_qka_vicuna,
        'chatglm-sft-lora':partial(print_and_save_qka_chatglm_finetune,peft_model=args.model),
        'chatglm-ppo-lora-delta':partial(print_and_save_qka_chatglm_finetune,peft_model=args.model),
        'chatglm-dpo-lora-delta':partial(print_and_save_qka_chatglm_finetune,peft_model=args.model),
    }

    function_dict[args.model](question_list, docs_and_scores_list, n=args.question_n, k=args.top_k, threshold=args.threshold, csv_saved_path=f'metadata/eval_data/langchain_{args.model}_k{args.top_k}.csv')
    
    return 0

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--knowledge_len', type=int, default=128)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model', type=str, default='gpt-4', choices=['gpt-3.5-turbo', 'gpt-4', 'chatglm', 'vicuna', 'chatglm-sft-lora', 'chatglm-ppo-lora-delta', 'chatglm-dpo-lora-delta' ])
    parser.add_argument('--public_embedding', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--question_n', type=int, default=1)
    parser.add_argument('--renew_vectordb', type=bool, default=False)
    parser.add_argument('--knowledge_file_path', type=str, default='data/knowledge.txt')
    parser.add_argument('--question_file_path', type=str, default='data/eval_question.txt')
    parser.add_argument('--db_persist_directory', type=str, default='vectordb')
    parser.add_argument('--embeddings_model_name', type=str, default='moka-ai/m3e-base') # moka-ai/m3e-base, sentence-transformers/use-cmlm-multilingual
    args = parser.parse_args()
    main(args)
