import re
import json
import argparse
import warnings

class JsonParser:
    def __init__(self, overlap_len=5, min_content_len=20, knowledge_len=128, cleaner=re.compile('<.*?>|&nbsp|\n|;|\s\s')):
        self.overlap_len = overlap_len
        self.min_content_len = min_content_len
        self.knowledge_len = knowledge_len
        self.cleaner = cleaner

    def reset_memory(self):
        self.parse_data = []
        self.list_item = []
        
    def serialize_from_json_and_output_data(self, obj, schema, header):
        self.reset_memory()
        self.serialize_from_json(obj, schema, header=header)

        return self.parse_data
    
    def serialize_from_json(self, obj, schema, header=[]):
        if isinstance(obj, str) and obj != "":
            current_header = " ".join(header) + " "
            clean_obj = self.cleaner.sub("", obj) if self.list_item == [] else ",".join(self.list_item)
            require_obj_len = self.knowledge_len-len(current_header)
            
            if require_obj_len <= self.min_content_len :
                warnings.warn(f"有文本前綴詞大於文本最大長度, 將會分開處理後續內容")
                parse_data = current_header[:self.knowledge_len]
                if parse_data not in self.parse_data:
                    self.parse_data.append(parse_data)
                self.serialize_from_json(obj, schema=schema, header=header[:-1])

            elif len(clean_obj) <= require_obj_len:
                parse_data = current_header + clean_obj
                if parse_data not in self.parse_data:
                    self.parse_data.append(parse_data)

            else:
                for idx in range(0, len(clean_obj), require_obj_len-self.overlap_len):
                    parse_data = current_header + clean_obj[idx:idx+require_obj_len]
                    if parse_data not in self.parse_data:
                        self.parse_data.append(parse_data)

        if isinstance(obj, dict) and isinstance(schema, dict):
            for key, value in schema.items():  
                if key in obj.keys() and key != "":
                    if value == "前綴":
                        header = header+[self.cleaner.sub("", obj[key])]

                    elif value =="序列":
                        self.list_item.append(self.cleaner.sub("", obj[key]))

                    else:
                        self.serialize_from_json(obj[key], schema=schema[key], header=header)

        if isinstance(obj, list) and isinstance(schema, list):
            for idx in range(len(obj)):
                if len(schema) >= len(obj):
                    self.serialize_from_json(obj[idx], schema=schema[idx], header=header)
                else:
                    for schema_try in schema:
                        self.serialize_from_json(obj[idx], schema=schema_try, header=header)                    

            if len(self.list_item) > 0:
                self.serialize_from_json("。", "", header=header)
                self.list_item.clear()
                
def main(args):
    
                
    data_path = [["cubcardlist.txt", "cubcardlist.json", "cubcardlist.txt"],
                 ["cube卡.txt", "cube卡.json", "cube卡.txt"],
                 ["overview_creditcard.txt", "overview_creditcard.json", "overview_creditcard.txt"],
                 ["eva.txt", "eva.json", "eva.txt"],
                 ["shopee.txt", "shopee.json", "shopee.txt"],
                 ["world.txt", "world.json", "world.txt"]]
    
    meta_data = ["(CUBE卡)", "(CUBE卡)", "(信用卡優惠)", "(長榮航空聯名卡)", "(蝦皮購物聯名卡)","(世界卡)"]

    all_parse_data = []
    for idx in range(len(data_path)):

        config_path, read_path, write_path = data_path[idx]
        header = meta_data[idx]
        with open("data/clean_card_config/"+config_path, "r") as f:
            schema = f.readlines()
            schema = eval("".join(schema).replace("\n",""))
            
        with open("data/raw_card_data/"+read_path, "r") as f:
            obj = json.load(f)

        json_parser = JsonParser(overlap_len = args.overlap_len, min_content_len = args.min_content_len, knowledge_len = args.knowledge_len)
        parse_data = json_parser.serialize_from_json_and_output_data(obj, schema, header=[header])
        all_parse_data.extend(parse_data)

        with open("data/clean_card_data/"+write_path, "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(parse_data))

    with open("data/knowledge.txt", "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(all_parse_data))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlap_len', type=int, default=5)
    parser.add_argument('--min_content_len', type=int, default=20)
    parser.add_argument('--knowledge_len', type=int, default=64)
    args = parser.parse_args()
    main(args)
