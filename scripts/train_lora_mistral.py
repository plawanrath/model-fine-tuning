from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

# Load base model and tokenizer
# model_name = "../models/Mistral-7B-v0.1"
model_name = "../output/mistral-lora-merged"
output_dir = "../output/mistral-lora-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model = model.float()

# Load Alpaca-format dataset
dataset = load_dataset("json", data_files="../../../rust_alpaca.json")["train"]

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
    num_train_epochs=1,
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
