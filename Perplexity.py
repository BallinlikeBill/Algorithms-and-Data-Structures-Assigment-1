import torch
import math
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--stride",help="Stride size for moving window", type=int, default=512)
parser.add_argument("--n-ctx",help="the window size", type=int, default=2048)
parser.add_argument("--begin-context-tokens",help="the number of pieces to be used as content in the first window",
                     type=int, default=512)
parser.add_argument("input_file", type=str, help='the file where the computing perplexity is evaluated')
parser.add_argument("output_file", type=str, help="the file in which the program results will be saved")
args = parser.parse_args()

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, tie_word_embeddings=False
)
model.eval()

filename = args.input_file
with open(filename, "r") as f:
    text = f.read()

tokens = tokenizer(text).input_ids

bos_token = tokenizer.bos_token_id
n_ctx = args.n_ctx
stride = args.stride
begin_context_tokens = args.begin_context_tokens

for i in range(0, len(tokens), stride):