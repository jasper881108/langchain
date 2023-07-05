import os
import openai
import argparse
from utils import (
    read_and_process_knowledge_to_langchain_docs,
    read_and_process_question_to_list,
    initial_langchain_embeddings,
    initial_or_read_langchain_database,
    print_and_save_qka_chatgpt,
    print_and_save_qka_chatglm,
)

def main(args):
    ## nghuyong/ernie-3.0-base-zh
    ## Chinese Model: shibing624/text2vec-base-chinese, GanymedeNil/text2vec-large-chinese, hfl/chinese-macbert-base
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
    openai.api_key_path = "openai_api.txt"
    public = args.public
    top_k = args.top_k
    question_n = args.question_n
    renew_vectordb = args.renew_vectordb
    knowledge_file_path = args.knowledge_file_path
    question_file_path = args.question_file_path
    embeddings_model_name = args.embeddings_model_name
    model_kwargs = {'device': 'cuda' if args.cuda else 'cpu'}
    db_persist_name = 'vectordbPublic' if public else 'vectordbPrivate'
    db_persist_directory = os.path.join(args.db_persist_directory, db_persist_name)

    docs = read_and_process_knowledge_to_langchain_docs(knowledge_file_path, separator = '\n', chunk_size=128)
    embedding_function = initial_langchain_embeddings(embeddings_model_name, model_kwargs, public)
    vectordb = initial_or_read_langchain_database(docs, embedding_function, db_persist_directory, renew_vectordb)
    
    question_list = read_and_process_question_to_list(question_file_path, separator = '\n')[:10]
    
    docs_and_scores_list = vectordb.similarity_search_with_score(question_list, k=top_k)
    print_and_save_qka_chatglm(question_list, docs_and_scores_list, n=question_n, k=top_k)
    print_and_save_qka_chatgpt(question_list, docs_and_scores_list, n=question_n, k=top_k)
    
    
    return 0

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--public', type=bool, default=False)
    parser.add_argument('--question_n', type=int, default=5)
    parser.add_argument('--renew_vectordb', type=bool, default=False)
    parser.add_argument('--knowledge_file_path', type=str, default='data/knowledge.txt')
    parser.add_argument('--question_file_path', type=str, default='data/question.txt')
    parser.add_argument('--db_persist_directory', type=str, default='vectordb')
    parser.add_argument('--embeddings_model_name', type=str, default='moka-ai/m3e-base')
    args = parser.parse_args()
    main(args)
