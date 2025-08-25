import gc
from argparse import ArgumentParser
from threading import Thread
from typing import Iterator, Optional
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,pipeline
)

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os


default_model_path_path = '/home/grads/a/anath/AHD/Llama-3.2-3B-Instruct'


tokenizer = AutoTokenizer.from_pretrained(default_model_path_path)
model = AutoModelForCausalLM.from_pretrained(default_model_path_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id


# Flask API
app = Flask(__name__)
CORS(app)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

@app.route(f'/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']

    max_repeat_prompt = 20

    # print(f'========================================== Prompt ==========================================')
    # print(f'{prompt}\n')
    # print(f'============================================================================================')
    # print(f'\n\n')

    messages = [{"role": "user", "content": prompt}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True)

    content = outputs[0]["generated_text"].split(
        "<|start_header_id|>assistant<|end_header_id|>"
    )[1]



    # Send back the response.
    return jsonify(
            {'content': content}
        )
    

# arguments
parser = ArgumentParser()
parser.add_argument('--d', nargs='+', default=['0'])
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--path', type=str, default=default_model_path_path)
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=11015)
args = parser.parse_args()

if __name__ == '__main__':
    app.run(host=args.host, port=args.port, threaded=False,debug=False,use_reloader=False)






