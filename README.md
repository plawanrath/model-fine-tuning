# model-fine-tuning

This repo contains scripts and fine-tuned models. 

For this I will be using LoRA to fine-tune models and I intend to use these locally using LMStudio


Rust Datasets

```
huggingface-cli download bigcode/the-stack \          
  --repo-type dataset \
  --include "data/rust/**" \
  --local-dir the_stack_rust \
  --local-dir-use-symlinks False
```

```
huggingface-cli download bigcode/the-stack-smol \
  --repo-type dataset \
  --include "data/rust/**" \
  --local-dir the_stack_rust_2 \
  --local-dir-use-symlinks False
```

# Converting to GGUF for LMStudio

### Install llama.cpp conversion tool (if needed)

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
python convert.py ./output/mistral-lora/merged --outfile mistral-lora.gguf
```