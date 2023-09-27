import re
import json
import argparse
import warnings

class JsonParser:
    def __init__(self, overlap_len=5, min_content_len=20, knowledge_len=128, cleaner=re.compile('<.*?>|&nbsp|\n|;|\s\s|')):
        self.overlap_len = overlap_len
        self.min_content_len = min_content_len
        self.knowledge_len = knowledge_len
        self.cleaner = cleaner

    def reset_memory(self):
        self.parse_data = {"context":[], "url":[]}
        self.list_item = []
        
    def serialize_from_json_and_output_data(self, obj, schema, header, url):
        self.reset_memory()
        self.serialize_from_json(obj, schema, header, url)

        return self.parse_data
    
    def serialize_from_json(self, obj, schema, header=[], url=""):
        if isinstance(obj, str) and obj != "":
            url = "None" if url =="" else url
            current_header = " ".join(header) + " "
            clean_obj = self.cleaner.sub("", obj) if self.list_item == [] else ",".join(self.list_item)
            require_obj_len = self.knowledge_len-len(current_header)
            
            if require_obj_len <= self.min_content_len :
                warnings.warn(f"有文本前綴詞大於文本最大長度, 將會分開處理後續內容")
                context = current_header[:self.knowledge_len]
                if context not in self.parse_data["context"]:
                    self.parse_data["context"].append(context)
                    self.parse_data["url"].append(url)
                self.serialize_from_json(obj, schema=schema, header=header[:-1])

            elif len(clean_obj) <= require_obj_len:
                context = current_header + clean_obj
                if context not in self.parse_data["context"]:
                    self.parse_data["context"].append(context)
                    self.parse_data["url"].append(url)

            else:
                for idx in range(0, len(clean_obj), require_obj_len-self.overlap_len):
                    context = current_header + clean_obj[idx:idx+require_obj_len]
                    if context not in self.parse_data["context"]:
                        self.parse_data["context"].append(context)
                        self.parse_data["url"].append(url)

        if isinstance(obj, dict) and isinstance(schema, dict):
            for key, value in schema.items():  
                if key in obj.keys() and key != "":

                    obj[key] = self.cleaner.sub("", obj[key]) if isinstance(obj[key], str) else obj[key]
                    if value == "前綴":
                        header = header+[obj[key]]
                    elif (value == "開始日期" or value == "結束日期") and obj[key] != "":
                        match = re.match(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", obj[key])
                        if match:
                            year, month, day = match.groups()
                            formatted_date = [f"(活動日期: {year}/{month}/{day}~"] if value == "開始日期" else [f"{year}/{month}/{day})"]
                        header = header+formatted_date
                    elif value =="序列":
                        self.list_item.append(obj[key])
                    elif value =="網址前綴":
                        header = header+[obj[key]]
                        url += "[{}]".format(obj[key])
                    elif value =="網址":
                        url += "({})".format(obj[key])
                    else:
                        self.serialize_from_json(obj[key], schema=schema[key], header=header, url=url)

        if isinstance(obj, list) and isinstance(schema, list):
            for idx in range(len(obj)):
                if len(schema) >= len(obj):
                    self.serialize_from_json(obj[idx], schema=schema[idx], header=header, url=url)
                else:
                    for schema_try in schema:
                        self.serialize_from_json(obj[idx], schema=schema_try, header=header, url=url)                    

            if len(self.list_item) > 0:
                self.serialize_from_json("。", "", header=header, url=url)
                self.list_item.clear()
                
def main(args):                
    data_path = [["cubcardlist.txt", "cubcardlist.json", "cubcardlist.txt"],
                 ["cube卡.txt", "cube卡.json", "cube卡.txt"],
                 ["overview_creditcard.txt", "overview_creditcard.json", "overview_creditcard.txt"],
                 ["eva.txt", "eva.json", "eva.txt"],
                 ["shopee.txt", "shopee.json", "shopee.txt"],
                 ["world.txt", "world.json", "world.txt"]]
    
    default_headers = ["CUBE卡權益", "CUBE卡", "限時活動", "長榮航空聯名卡", "蝦皮購物聯名卡","世界卡"]
    default_urls = ["[CUBE卡](https://www.cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/cube/)", 
                    "[CUBE卡](https://www.cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/cube/)", 
                    "", 
                    "[長榮航空聯名卡](https://cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/eva/)", 
                    "[蝦皮購物聯名卡](https://cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/shopee/)",
                    "[世界卡](https://cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/world/)"]

    context = []
    url = []
    for idx in range(len(data_path)):

        config_path, read_path, write_path = data_path[idx]
        default_header = default_headers[idx]
        default_url = default_urls[idx]

        with open("data/clean_card_config/"+config_path, "r") as f:
            schema = f.readlines()
            schema = eval("".join(schema).replace("\n",""))
            
        with open("data/raw_card_data/"+read_path, "r") as f:
            obj = json.load(f)

        json_parser = JsonParser(overlap_len = args.overlap_len, min_content_len = args.min_content_len, knowledge_len = args.knowledge_len)
        parse_data = json_parser.serialize_from_json_and_output_data(obj, schema, header=[default_header], url=default_url)

        context.extend(parse_data["context"])
        url.extend(parse_data["url"])
    
        with open("data/clean_card_data/"+write_path, "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(parse_data["context"]))

    with open("data/knowledge.txt", "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(context))

    with open("data/url.txt", "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(url))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlap_len', type=int, default=15)
    parser.add_argument('--min_content_len', type=int, default=5)
    parser.add_argument('--knowledge_len', type=int, default=128)
    args = parser.parse_args()
    main(args)
