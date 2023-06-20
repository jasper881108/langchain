import os
import openai
import argparse
from utils import (
    read_and_process_answer_to_langchain_docs,
    read_and_process_question_to_list,
    initial_langchain_embeddings,
    initial_or_read_langchain_database,
    print_tabulate_question_and_answer
)

def main(args):
    ## nghuyong/ernie-3.0-base-zh
    ## Chinese Model: shibing624/text2vec-base-chinese, GanymedeNil/text2vec-large-chinese, hfl/chinese-macbert-base
    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key_path = "openai_api.txt"
    public = args.public
    top_k = args.top_k
    question_n = args.question_n
    renew_vectordb = args.renew_vectordb
    answer_file_path = args.answer_file_path
    question_file_path = args.question_file_path
    embeddings_model_name = args.embeddings_model_name
    model_kwargs = {'device': 'cuda' if args.cuda else 'cpu'}
    db_persist_name = 'vectordbPublic' if public else 'vectordbPrivate'
    db_persist_directory = os.path.join(args.db_persist_directory, db_persist_name)

    docs = read_and_process_answer_to_langchain_docs(answer_file_path, separator = '\n', chunk_size=1)
    embedding_function = initial_langchain_embeddings(embeddings_model_name, model_kwargs, public)
    vectordb = initial_or_read_langchain_database(docs, embedding_function, db_persist_directory, renew_vectordb)
    
    question_list = read_and_process_question_to_list(question_file_path, separator = '\n')[:question_n]
   
    docs_and_scores_list = vectordb.similarity_search_with_score(question_list, k=top_k)
    print_tabulate_question_and_answer(question_list, docs_and_scores_list, n=question_n, k=top_k)
    
    return 0

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--public', type=bool, default=False)
    parser.add_argument('--question_n', type=int, default=3)
    parser.add_argument('--renew_vectordb', type=bool, default=False)
    parser.add_argument('--answer_file_path', type=str, default='data/answer.txt')
    parser.add_argument('--question_file_path', type=str, default='data/question.txt')
    parser.add_argument('--db_persist_directory', type=str, default='vectordb')
    parser.add_argument('--embeddings_model_name', type=str, default='nghuyong/ernie-3.0-base-zh')
    args = parser.parse_args()
    main(args)
