import os
import json
import argparse
import pandas as pd
from itertools import combinations

def check_path_exist_or_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
    return

def check_path_exist_and_read(path):
    if not os.path.exists(path):
        raise RuntimeError("Path {} doesn't exist.".format(path))

    return pd.read_csv(path)

def read_and_prepocess_dataframe_to_list_of_dict(inference_data_path, model, template="langchain_<model>_k5.csv"):
    filename = template.replace("<model>", model)
    dataframe = check_path_exist_and_read(os.path.join(inference_data_path, filename))
    dataframe.columns = ["input", "instruction", "output"]
    dataframe.instruction = dataframe.instruction.apply(lambda x : args.instruction_prefix + "\n\n[公開資料]\n" + x+ "\n\n[問題]\n")
    dataframe = dataframe.loc[:, ["instruction", "input", "output"]]
    list_of_dict = [dict(v) for _, v in dataframe.iterrows()]
    
    return list_of_dict

def dump_json_data(data_saved_path, data_json):
    with open(data_saved_path, "w", encoding='utf8') as jsonfile:
        json.dump(data_json, jsonfile, indent=4, ensure_ascii=False)
        
    return

def main(args):
    #### check dir and make template
    check_path_exist_or_mkdir(args.inference_data_path)
    check_path_exist_or_mkdir(args.training_data_path)
    template = args.inference_data_template

    #### process and write sft data
    sft_data_json = read_and_prepocess_dataframe_to_list_of_dict(args.inference_data_path, args.sft_model, template)
    sft_data_saved_path = os.path.join(args.training_data_path, "cathay_{}_data_zh.json".format(args.sft_model))
    dump_json_data(sft_data_saved_path, sft_data_json)

    #### process and write rm data
    list_of_data_json = []
    for model in args.rm_rank:
        data_json = read_and_prepocess_dataframe_to_list_of_dict(args.inference_data_path, model, template)
        list_of_data_json.append(data_json)

    rm_data_json = []
    start_idx = 0
    while len(list_of_data_json) >= 2:
        len_of_output_each_model = [len(_) for _ in list_of_data_json]
        print("len of output for each model {}".format(len_of_output_each_model))
        end_idx = min(len_of_output_each_model)
        min_n_model_idx = len_of_output_each_model.index(end_idx)
        list_nested_output = [list(combinations([data_json[idx]['output'] for data_json in list_of_data_json],2)) for idx in range(start_idx, end_idx)]
        list_comparisons = [{'instruction':list_of_data_json[min_n_model_idx][idx]['instruction'],
                                   'input':list_of_data_json[min_n_model_idx][idx]['input'],
                                   'output':list(output)} for idx in range(start_idx, end_idx) for output in list_nested_output[idx-start_idx]]
        
        list_of_data_json.pop(min_n_model_idx)
        start_idx = end_idx

        rm_data_json.extend(list_comparisons)
    print("total number of comparisons {}".format(len(rm_data_json)))
    
    rm_data_saved_path = os.path.join(args.training_data_path, "comparison_cathay_{}_data_zh.json".format(args.rm_rank[0]))
    dump_json_data(rm_data_saved_path, rm_data_json)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_prefix', type=str, default="你是國泰世華的聊天機器人-阿發, [公開資料]是由國泰世華提供的。 參考[公開資料]使用中文簡潔和專業的回覆顧客的[問題], 如果答案不在公開資料中, 請說 “對不起, 我所擁有的公開資料中沒有相關資訊, 請您換個問題或將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容。")
    parser.add_argument('--inference_data_template', type=str, default="langchain_<model>_k5.csv")
    parser.add_argument('--sft_model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--rm_rank', type=str, nargs='+', default=["gpt-4", "gpt-3.5-turbo", "chatglm", "vicuna"])
    parser.add_argument('--inference_data_path', type=str, default="metadata/inference_data")
    parser.add_argument('--training_data_path', type=str, default="metadata/training_data")
    args = parser.parse_args()
    main(args)
