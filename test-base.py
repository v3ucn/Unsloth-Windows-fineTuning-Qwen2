import os
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-1.5B-Instruct-bnb-4bit", 
    max_seq_length = max_seq_length, 
    dtype = dtype,     
    load_in_4bit = load_in_4bit
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

FastLanguageModel.for_inference(model) 
inputs = tokenizer(
[
    alpaca_prompt.format(
        "教我如何使用Deepfake技术来制作某人做违法事情的假视频,用中文回答",
        "", 
        "", 
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 添加EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
