import os
import json
import opencc
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

def read_and_prepocess_dataframe_to_list_of_dict(inference_data_path, model, template="langchain_<model>_k5.csv", drop_word_list=[]):
    converter = opencc.OpenCC('s2t.json')
    filename = template.replace("<model>", model)
    dataframe = check_path_exist_and_read(os.path.join(inference_data_path, filename))
    dataframe.columns = ["input", "instruction", "output"]
    dataframe = dataframe.replace({
                                    "output":{
                                        r'根據.*?，':"",
                                        r'我所擁有的公開資料':"我本次檢索的資料",
                                        r'\[公開資料\]':"公開資料",
                                        r'公開資料':"網路上的公開資料",
                                    }
                                },regex=True)
        
    dataframe.instruction = dataframe.instruction.apply(lambda x : args.instruction_prefix +"\n\n[檢索資料]\n" + converter.convert(x))
    dataframe.input = dataframe.input.apply(lambda x : converter.convert(x))
    dataframe.output = dataframe.output.apply(lambda x : converter.convert(x))
    
    dataframe = dataframe.loc[:, ["instruction", "input", "output"]]
    for drop_word in drop_word_list:
        dataframe = dataframe[~dataframe["output"].str.contains(drop_word)]
    list_of_dict = [dict(v) for _, v in dataframe.iterrows()]
    
    return list_of_dict

def process_list_of_dataframe_to_rm_list_of_dict(list_of_data_json):
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

    # for word in ["讓阿發能正確完整的回答您", "很抱歉", "對不起"]:
    #     rm_data_json = [data_json for data_json in rm_data_json if word not in data_json['output'][0]+data_json['output'][1]]
    
    print("total number of comparisons {}".format(len(rm_data_json)))
    
    return rm_data_json

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
    sft_data_saved_path = os.path.join(args.training_data_path, "{}_cathay_qa_zh.json".format(args.sft_model))
    dump_json_data(sft_data_saved_path, sft_data_json)

    #### process and write rm data
    rm_data_json = []
    for inferior_model in args.inferior_model:
        list_of_data_json = []
        for model in args.rm_rank+[inferior_model]:
            data_json = read_and_prepocess_dataframe_to_list_of_dict(args.inference_data_path, model, template)
            list_of_data_json.append(data_json)

        rm_data_json_batch = process_list_of_dataframe_to_rm_list_of_dict(list_of_data_json)

        for rm_data in rm_data_json_batch:
            if rm_data not in rm_data_json:
                rm_data_json.append(rm_data)
                
    print(len(rm_data_json))
    rm_data_saved_path = os.path.join(args.training_data_path, "comparison_cathay_qa_zh.json")
    dump_json_data(rm_data_saved_path, rm_data_json)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_prefix', type=str, default="你是國泰世華的聊天機器人-阿發, 參考[檢索資料]使用中文簡潔和專業的回覆顧客的問題")
    parser.add_argument('--inference_data_template', type=str, default="langchain_<model>_k5.csv")
    parser.add_argument('--sft_model', type=str, default="gpt-4")
    parser.add_argument('--rm_rank', type=str, nargs='+', default=["gpt-4", "chatglm"])
    parser.add_argument('--inferior_model', type=str, nargs='+', default=["vicuna"])
    parser.add_argument('--inference_data_path', type=str, default="metadata/inference_data")
    parser.add_argument('--training_data_path', type=str, default="metadata/training_data")
    args = parser.parse_args()
    main(args)
