# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# MODEL_PATH = "/home/anath/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# # Prefer bf16 if supported; else fp16
# dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# # Leave headroom so sharding avoids OOM
# ngpu = torch.cuda.device_count()
# max_memory = {i: "40GiB" for i in range(ngpu)}  # for 48 GB cards
# max_memory["cpu"] = "64GiB"                      # optional spillover

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=dtype,
#     low_cpu_mem_usage=True,
#     device_map="balanced_low_0",   # better placement than "auto" for big models
#     max_memory=max_memory,
#     # offload_folder="./offload",    # used only if needed
# )

# tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
# tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
# model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id

# gen = pipeline("text-generation", model=model, tokenizer=tok)

# # Keep KV-cache modest during testing
# out = gen("Hello,", max_new_tokens=128, do_sample=True)
# print(out[0]["generated_text"])



# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# path = "/home/anath/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained(path)
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# # inputs = tokenizer.apply_chat_template(
# # 	messages,
# # 	add_generation_prompt=True,
# # 	tokenize=True,
# # 	return_dict=True,
# # 	return_tensors="pt",
# # ).to(model.device)


# print("Generating...")
# print("Model device:", model.device)
# outputs = model.generate(**inputs, max_new_tokens=40)
# # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))



# import gc
# from argparse import ArgumentParser
# from threading import Thread
# from typing import Iterator, Optional
# import torch
# from transformers import (
#     AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,pipeline
# )

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import json
# import os


# default_model_path_path = '/home/anath/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b'



# model = AutoModelForCausalLM.from_pretrained(default_model_path_path)
# tokenizer = AutoTokenizer.from_pretrained(default_model_path_path)

# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
# if model.config.pad_token_id is None:
#     model.config.pad_token_id = model.config.eos_token_id





# # Flask API
# app = Flask(__name__)
# CORS(app)

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# print("Model loaded.")

# @app.route(f'/completions', methods=['POST'])
# def completions():
#     content = request.json
#     prompt = content['prompt']

#     max_repeat_prompt = 20

#     # print(f'========================================== Prompt ==========================================')
#     # print(f'{prompt}\n')
#     # print(f'============================================================================================')
#     # print(f'\n\n')

#     messages = [{"role": "user", "content": prompt}]

#     # prompt = tokenizer.apply_chat_template(
#     #     messages, 
#     #     tokenize=False, 
#     #     add_generation_prompt=True
#     # )

#     outputs = pipe(prompt, max_new_tokens=1024, do_sample=True)

#     content = outputs[0]["generated_text"].split(
#         "<|start_header_id|>assistant<|end_header_id|>"
#     )[1]



#     # Send back the response.
#     return jsonify(
#             {'content': content}
#         )
    

# # arguments
# parser = ArgumentParser()
# parser.add_argument('--d', nargs='+', default=['0'])
# parser.add_argument('--quantization', default=False, action='store_true')
# parser.add_argument('--path', type=str, default=default_model_path_path)
# parser.add_argument('--host', type=str, default='127.0.0.1')
# parser.add_argument('--port', type=int, default=11015)
# args = parser.parse_args()

# if __name__ == '__main__':
#     app.run(host=args.host, port=args.port, threaded=False,debug=False,use_reloader=False)






