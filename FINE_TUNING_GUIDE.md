# Fine-Tuning Guide (OpenAI + Hugging Face)

This guide shows how to fine-tune an LLM on your Natural Language Geometry (Text-to-CAD) dataset.

Dataset files (already generated):
- fine_tuning_datasets/train_dataset.jsonl
- fine_tuning_datasets/validation_dataset.jsonl
- fine_tuning_datasets/train_dataset.json / .csv (for analysis)
- fine_tuning_datasets/validation_dataset.json / .csv (for analysis)
- fine_tuning_datasets/dataset_statistics.json

Important
- Never paste API keys directly in commands. Set them as environment variables.
- All examples use Windows PowerShell.

---

## 1) OpenAI Fine-Tuning

Requirements
- An OpenAI account with API access.
- openai Python package or OpenAI CLI. The CLI is included with the python package as of recent versions.

Install:
- python -m pip install --upgrade openai

Environment variables (PowerShell):
- $env:OPENAI_API_KEY = "{{OPENAI_API_KEY}}"
- $env:OPENAI_ORG = "{{OPENAI_ORG_ID}}"  (optional)

Recommended models
- gpt-4o-mini for cost/quality balance
- or a compatible instruct model that supports fine-tuning.

Using Python (scripted):
- Create a python file (openai_ft.py) and run a fine-tune job. Example (adjust model name as needed):

    ```python
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Upload files
    train_file = client.files.create(
        file=open("fine_tuning_datasets/train_dataset.jsonl", "rb"),
        purpose="fine-tune"
    )
    val_file = client.files.create(
        file=open("fine_tuning_datasets/validation_dataset.jsonl", "rb"),
        purpose="fine-tune"
    )

    # Create job
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model="gpt-4o-mini"  # choose your base model
    )

    print("Job id:", job.id)
    ```

- Monitor job:

    ```python
    from openai import OpenAI
    import os, time

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    job_id = "{{JOB_ID_FROM_CREATE}}"
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(job)
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(15)
    ```

- Inference with the fine-tuned model:

    ```python
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    model_id = "{{ft:gpt-4o-mini:your_ft_model_id}}"
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a CAD geometry compiler..."},
            {"role": "user", "content": "Create a 20mm cube"}
        ],
        temperature=0.2
    )
    print(resp.choices[0].message.content)
    ```

Tips
- Use temperature=0.1~0.3 for consistent JSON.
- Validate outputs against your JSON schema.

---

## 2) Hugging Face Fine-Tuning

Two common routes: AutoTrain (no-code web) or Transformers/TRL (code-based).

### Option A: AutoTrain (Web UI)

1. Login to https://huggingface.co/ and ensure you have a write token.
2. Create a dataset repo (web UI) or via CLI:
   - python -m pip install --upgrade huggingface_hub git-lfs
   - huggingface-cli login
   - huggingface-cli repo create your-geometry-dataset --type dataset
3. Upload your JSONL files (use Git LFS):
   - git lfs install
   - git clone https://huggingface.co/datasets/<username>/your-geometry-dataset
   - copy fine_tuning_datasets/*.jsonl into the cloned repo
   - git add .gitattributes *.jsonl
   - git commit -m "Add Text-to-CAD training data"
   - git push
4. Open https://autotrain.huggingface.co, create a new project, select your dataset, choose a base model (e.g., meta-llama/Meta-Llama-3-8B-Instruct or a smaller instruct model), and start fine-tuning.

### Option B: Transformers + TRL (code)

Install packages:
- python -m pip install --upgrade transformers datasets accelerate peft trl bitsandbytes

Prepare a small trainer script (sft_train.py) using your JSONL (OpenAI-format messages):

```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import os

# Base model (pick one with suitable license)
BASE_MODEL = os.environ.get("HF_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,  # optional: requires bitsandbytes
    device_map="auto"
)

# Load OpenAI JSONL messages as a dataset
train = load_dataset("json", data_files="fine_tuning_datasets/train_dataset.jsonl")["train"]
val = load_dataset("json", data_files="fine_tuning_datasets/validation_dataset.jsonl")["train"]

# Convert to prompt format
# Each example is {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

def format_example(example):
    msgs = example["messages"]
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    # Return full supervised pair
    return {"text": f"<s>[INST] {user} [/INST]\n{assistant}</s>"}

train = train.map(format_example)
val = val.map(format_example)

training_args = TrainingArguments(
    output_dir="./hf_ft_out",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    bf16=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=val,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()
trainer.save_model("./hf_ft_out/model")
```

Run:
- python sft_train.py

Inference (Transformers):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./hf_ft_out/model"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

prompt = "Create a 20mm cube"
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Tips
- Keep temperature low (0.1–0.3) for JSON consistency.
- Add JSON validation on outputs in your app pipeline.

---

## 3) Suggested Training Strategy

- Start with 1k examples to validate the pipeline.
- Move to 5k–10k examples for stronger performance.
- Monitor validation loss and JSON validity rate.
- Evaluate geometric accuracy by sampling outputs.

## 4) Using the Fine-Tuned Model in Your App

- OpenAI: set $env:OPENAI_API_KEY and your fine-tuned model id; your app’s optional LLM toggle can call it.
- Hugging Face: load your new model with Transformers and swap it into your inference stack.

## 5) Security and Secrets

- Always store API keys in environment variables.
- Do not print or commit secrets.
- Rotate keys periodically.

