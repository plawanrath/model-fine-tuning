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

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

# Load base model and tokenizer
model_name = "../models/yi-9b"
output_dir = "../output/yi-9b-rust-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Explicitly load model in float16 for Apple Silicon MPS compatibility
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # Explicitly set float16
    device_map="cpu",            # Load onto CPU first
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Now explicitly move model onto MPS GPU
model = model.to('mps')

# Apply LoRA
# Simple LoRA Config compatible with PEFT 0.7.1 (no bitsandbytes triggered)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
# Do NOT convert back to float32 here!

# Load Alpaca-format dataset
dataset = load_dataset("json", data_files="../datasets/rust_alpaca.json")["train"]

# Format each sample
def format(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()

    prompt = f"{instruction}\n{input_text}".strip()
    if not prompt or not output_text:
        raise ValueError("Missing instruction or output")

    input_enc = tokenizer(prompt + tokenizer.eos_token, truncation=True, padding="max_length", max_length=512)
    output_enc = tokenizer(output_text + tokenizer.eos_token, truncation=True, padding="max_length", max_length=512)

    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": output_enc["input_ids"]
    }

dataset = dataset.map(format).remove_columns(dataset.column_names)

# Training setup
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=2,
    learning_rate=1.5e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Clean output dir first
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)

# Train
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
