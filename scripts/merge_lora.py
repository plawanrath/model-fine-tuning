import sys
import types

# Robustly mock bitsandbytes entirely to avoid Apple Silicon compatibility issues
bnb_mock = types.ModuleType("bitsandbytes")
bnb_mock.nn = types.ModuleType("nn")
bnb_mock.optim = types.ModuleType("optim")
sys.modules["bitsandbytes"] = bnb_mock
sys.modules["bitsandbytes.nn"] = bnb_mock.nn
sys.modules["bitsandbytes.optim"] = bnb_mock.optim

# Explicitly mock necessary bitsandbytes attributes to satisfy PEFT's checks
setattr(bnb_mock.nn, "Linear4bit", None)
setattr(bnb_mock.nn, "Linear8bitLt", None)
setattr(bnb_mock.optim, "Adam8bit", None)
setattr(bnb_mock.optim, "AdamW8bit", None)

# Now also patch importlib's find_spec to satisfy all internal checks robustly
import importlib.util
original_find_spec = importlib.util.find_spec

def patched_find_spec(name, *args, **kwargs):
    if "bitsandbytes" in name:
        return importlib.machinery.ModuleSpec(name, None)
    return original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = patched_find_spec

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, shutil, pathlib

BASE_DIR   = "../models/yi-9b"
LORA_DIR   = "../output/yi-9b-rust-v1"          # the folder that now contains adapter_model.bin
OUT_DIR    = "../output/yi-9b-rust-v1-merged"   # new folder to hold the merged model

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