from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, shutil, pathlib

BASE_DIR   = "../models/Mistral-7B-v0.1"
LORA_DIR   = "../output/mistral-lora"          # the folder that now contains adapter_model.bin
OUT_DIR    = "../output/mistral-lora-merged"   # new folder to hold the merged model

# clean OUT_DIR if it already exists
if pathlib.Path(OUT_DIR).exists():
    shutil.rmtree(OUT_DIR)

base = AutoModelForCausalLM.from_pretrained(
    BASE_DIR,
    torch_dtype=torch.float16,      # HF-safe dtype for export
    trust_remote_code=True
)
lora = PeftModel.from_pretrained(base, LORA_DIR)

merged = lora.merge_and_unload()   # ← magic line
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)

merged.save_pretrained(OUT_DIR, safe_serialization=True)  # writes model-00001-of-nnnn.safetensors
tokenizer.save_pretrained(OUT_DIR)

print(f"✅  merged model saved to: {OUT_DIR}")