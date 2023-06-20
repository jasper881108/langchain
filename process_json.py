import re
import json
import argparse

def serialize_from_json(parse_data, obj, schema,overlap_len=20, header=[], cleaner=re.compile('<.*?>|&nbsp|\n|;|\s')):
    
    if isinstance(obj, str) and obj != "":
        current_header = "-".join(header) + "-"
        clean_obj = cleaner.sub("", obj)
        require_obj_len = 128-len(current_header)
        
        if len(clean_obj) <= require_obj_len:
          parse_data.append(current_header + clean_obj)
          
        else:
            for idx in range(0, len(clean_obj), require_obj_len-overlap_len):
                parse_data.append(current_header + clean_obj[idx:idx+require_obj_len])

    if isinstance(obj, dict) and isinstance(schema, dict):       
        for key, value in schema.items():  
            if key in obj.keys():
                if value == "前綴":
                    header = header+[cleaner.sub("", obj[key])]
                else:
                    parse_data = serialize_from_json(
                                        parse_data,
                                        obj[key], 
                                        schema=schema[key], 
                                        overlap_len=overlap_len, 
                                        header=header, 
                                        cleaner=cleaner
                                    )
                

    if isinstance(obj, list) and isinstance(schema, list):
        for idx in range(len(obj)):
            idx_for_schema = 0 if len(schema) < idx + 1 else idx
            parse_data = serialize_from_json(
                                    parse_data,
                                    obj[idx], 
                                    schema=schema[idx_for_schema], 
                                    overlap_len=overlap_len, 
                                    header=header, 
                                    cleaner=cleaner
                                )
        
    return parse_data

def main(args):
    data_path = [["cubcardlist.txt", "cubcardlist.json", "cubcardlist.txt"],
                 ["eva.txt", "eva.json", "eva.txt"],
                 ["shopee.txt", "shopee.json", "shopee.txt"],
                 ["world.txt", "world.json", "world.txt"],
                 ["cube卡.txt", "cube卡.json", "cube卡.txt"],]
    
    meta_data = ["CUBE卡", "長榮航空聯名卡", "蝦皮購物聯名卡","世界卡", "CUBE卡"]

    all_parse_data = []
    for idx in range(len(data_path)):
        config_path, read_path, write_path = data_path[idx]
        header = meta_data[idx]
        with open("data/clean_card_config/"+config_path, "r") as f:
            schema = f.readlines()
            schema = eval("".join(schema).replace("\n",""))
            
        with open("data/raw_card_data/"+read_path, "r") as f:
            obj = json.load(f)
            
        parse_data = serialize_from_json([], obj, schema, overlap_len = 20, cleaner = re.compile('<.*?>|&nbsp|\n|;|\s'), header=[header])
        all_parse_data.extend(parse_data)
        
        with open("data/clean_card_data/"+write_path, "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(parse_data))

    with open("data/answer.txt", "w", encoding='utf8') as txt_file:
            txt_file.write("\n".join(all_parse_data))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
