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
python ./convert-hf-to-gguf.py \
   ../output/mistral-lora-merged \
   --outfile ../output/mistral-lora-f16.gguf \
   --outtype f16
```

Optionally quantize

```
./quantize ../output/mistral-lora-f16.gguf \
           ../output/mistral-lora-q4_K_M.gguf \
           q4_K_M # or q5_1, q6_K, etc.
```


### Why LM Studio still can’t “see” the file

LM Studio doesn’t just scan every file directly inside ~/.lmstudio/models/.
For each model it expects two nested folders:

perl
Copy
Edit
~/.lmstudio/models/
└── <publisher-name>/
    └── <model-name>/
        └── model-file.gguf
(That layout matches what Hugging Face uses and is in the official docs.)