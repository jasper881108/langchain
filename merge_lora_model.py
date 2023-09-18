import os
import argparse
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

def main(args):
    lora_checkpoint_config={
        "sft": ["Jasper881108/chatglm-sft-lora"],
        "rlhf": ["Jasper881108/chatglm-sft-lora", "Jasper881108/chatglm-ppo-lora-delta"],
        "dpo": ["Jasper881108/chatglm-sft-lora", "Jasper881108/chatglm-dpo-lora-delta"]
    }
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

    sft_model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    for peft_model in lora_checkpoint_config["sft"]:
        sft_model = PeftModel.from_pretrained(sft_model, peft_model).merge_and_unload()
    sft_model.save_pretrained("model/chatglm-sft", max_shard_size="2GB")
    tokenizer.save_pretrained("model/chatglm-sft")
    
    rlhf_model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    for peft_model in lora_checkpoint_config["rlhf"]:
        rlhf_model = PeftModel.from_pretrained(rlhf_model, peft_model).merge_and_unload()
    rlhf_model.save_pretrained("model/chatglm-rlhf", max_shard_size="2GB")
    tokenizer.save_pretrained("model/chatglm-rlhf")

    dpo_model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    for peft_model in lora_checkpoint_config["dpo"]:
        dpo_model = PeftModel.from_pretrained(dpo_model, peft_model).merge_and_unload()
    dpo_model.save_pretrained("model/chatglm-dpo", max_shard_size="2GB")
    tokenizer.save_pretrained("model/chatglm-dpo")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
