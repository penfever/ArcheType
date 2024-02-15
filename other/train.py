from ast import literal_eval

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig,
    Trainer
)

from trl import SFTTrainer

LLAMA_PATH = "/home/bf996/text-generation-webui/models/llama-7b-cta-full"
#Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)
#Create a new token and add it to the tokenizer
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'

#dataset_train = load_dataset("timdettmers/openassistant-guanaco", split='train')
#dataset_test = load_dataset("timdettmers/openassistant-guanaco", split='test')

with open("/home/bf996/archetype/metadata/T2D/T2D_train_asst_instr.json", "rb") as f:
    l = literal_eval(f.read().decode("utf-8"))
    fr = [{"text" : v} for v in l]
    # dataset_1 = Dataset.from_list(fr)

with open("/home/bf996/archetype/metadata/T2D/T2D_test_asst_instr.json", "rb") as f:
    l = literal_eval(f.read().decode("utf-8"))
    fr = fr + [{"text" : v} for v in l]
    dataset_1f = Dataset.from_list(fr)

dataset = dataset_1f.train_test_split(test_size=0.3, shuffle=True)

pr_list = []

for i in range(len(dataset['test'])):
    p = dataset['test']['text'][i].split("### Assistant: ")[0].strip()
    r = dataset['test']['text'][i].split("### Assistant: ")[1].strip()
    d = {"prompt": p, "response": r}
    pr_list.append(d)

#dataset_full = concatenate_datasets([dataset_train, dataset_1f])
#print(dataset_full)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=compute_dtype,
)

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=True,
# )

#device_map={"": 0}

model = AutoModelForCausalLM.from_pretrained(
          LLAMA_PATH, quantization_config=bnb_config,
)

#Resize the embeddings
model.resize_token_embeddings(len(tokenizer))
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","v_proj"]
)

training_arguments = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        log_level="debug",
        optim="paged_adamw_32bit",
        save_steps=10, #change to 500
        logging_steps=1, #change to 100
        learning_rate=1e-4,
        eval_steps=200, #change to 200
        fp16=True,
        max_grad_norm=0.3,
        #num_train_epochs=3, # remove "#"
        max_steps=10, #remove this
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
)

# trainer = Trainer(
#         model=model,
#         args=training_arguments,
#         train_dataset=dataset['train'],
#         eval_dataset=dataset['test'],
#         tokenizer=tokenizer,
# )

trainer.train()

# model = AutoModelForCausalLM.from_pretrained(
#           "./results/checkpoint-10", # quantization_config=bnb_config,
# )

model = PeftModel.from_pretrained(model, "./results/checkpoint-10")

def generate(instruction):
    prompt = "### Human: "+instruction+"### Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=10
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        return output.split("### Assistant: ")[1].strip()

pr_list_model = []

from tqdm.auto import tqdm

for item in tqdm(pr_list):
    pr_list_model.append({"prompt": item["prompt"], "model response": generate(item["prompt"]), "true response": item["response"]} )

import json

with open("/home/bf996/archetype/results/peft_model_responses.json", "w") as f:
    f.write(json.dumps(pr_list_model, indent=4))