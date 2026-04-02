import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, tie_word_embeddings=False
)
model.eval()