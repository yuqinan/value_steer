from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json
import csv
from huggingface_hub import login
login("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")
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Load the tokenizer and model
MODEL_NAME = 'Llama-3.1-8B'

tokenizer = AutoTokenizer.from_pretrained(f"/nlp/scr/qinanyu/model_cache/result_trainer/checkpoint-2000", token = True, legacy = False)
model = AutoModelForCausalLM.from_pretrained(f"/nlp/scr/qinanyu/model_cache/result_trainer/checkpoint-2000", token = True, device_map = "auto")

inputs = tokenizer("What shoud I do tonight?", return_tensors="pt").to(device)  # For PyTorch, use "tf" for TensorFlow
outputs = model.generate(**inputs, max_length=200, temperature = 0.8)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)