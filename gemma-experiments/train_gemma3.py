import os
import re
from unsloth import FastModel # type: ignore
from datasets import Dataset # type: ignore
from trl import SFTTrainer, SFTConfig # type: ignore
import pandas as pd # type: ignore
import numpy as np
from unsloth import FastLanguageModel # type: ignore
from trl import SFTTrainer # type: ignore
from unsloth.chat_templates import standardize_data_formats, get_chat_template, train_on_responses_only # type: ignore
from tqdm import tqdm # type: ignore

#reffered to
#https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb

max_seq_length = 2048

model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
load_in_4bit = True

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    load_in_8bit = False,
    full_finetuning = False
)

CHECKPOINT_BASE_NAME = f"/home/hice1/akhandelwal83/scratch/deep-learning/project/gemma3/checkpoints/gemma-3-4b-it-bnb-4bit-v1"
LOGS = f"{CHECKPOINT_BASE_NAME}/logs/"

if not os.path.exists(LOGS):
    os.makedirs(LOGS)

train = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_train.parquet')
val = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_valid.parquet')

model = FastModel.get_peft_model(
    model, 
    finetune_language_layers   = True,  
    finetune_attention_modules = True,  
    finetune_mlp_modules       = True,  

    r = 16,          
    lora_alpha = 32,  
    lora_dropout = 0.1,
    bias = "none",
)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

for param in model.model.vision_tower.vision_model.parameters():
    param.requires_grad = False

print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

def prepare_dataset(xdf):

  data_list = []

  for idx,row in xdf.iterrows():
    #print(row)
    part = {}
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']
    user_dict = {}
    content = f"""
    You will be given a prompt and responses from two different AI models. The prompt and responses may be in any language.
    Your task is to predict which response a human would prefer.

    Think carefully before answering, and respond with only one word: either **response_a** or **response_b**.

    Prompt:
    {prompt}

    Reponse A:
    {response_a}

    Response B:
    {response_b}

    Your prediction:
    """

    user_dict['content'] = content
    user_dict['role'] = 'user'

    model_dict = {}
    model_dict['content'] = "response_b" if row['winner'] == 1 else "response_a"
    model_dict['role'] = 'assistant'

    part['conversations'] = [user_dict,model_dict]
    #print(part)
    data_list.append(part) 

  return Dataset.from_list(data_list)

train_dataset = prepare_dataset(train)
eval_dataset = prepare_dataset(val)


train_dataset = standardize_data_formats(train_dataset)
eval_dataset = standardize_data_formats(eval_dataset)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma3" # omit model’s “Assistant:” prompt if you prefer
)

def apply_chat_template(examples):
    convos = examples["conversations"]          # a list of chats
    #print(convos)
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset_train = train_dataset.map(apply_chat_template, batched = True)
dataset_eval = eval_dataset.map(apply_chat_template, batched = True)

training_args = SFTConfig(
    output_dir=CHECKPOINT_BASE_NAME,
    overwrite_output_dir=True,
    dataset_text_field = "text",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, 
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    eval_steps = 200,
    save_strategy="steps",
    save_steps=50,
    logging_dir=LOGS,
    warmup_steps = 50,
    logging_strategy="steps",
    logging_steps=1,
    num_train_epochs = 1, 
    learning_rate = 5e-5,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",
    report_to = "tensorboard",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset = dataset_eval, # Can set up evaluation!
    args = training_args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


trainer_stats = trainer.train()

def extract_prediction(text):
    match = re.search(r"<start_of_turn>model\s*(.*?)<end_of_turn>", text, re.DOTALL)
    if match:
        prediction = match.group(1).strip().lower()
        if "response_a" in prediction:
            return "response_a"
        elif "response_b" in prediction:
            return "response_b"
    return "unknown"

total = 0
correct = 0
for example in tqdm(dataset_eval):
    total +=1
    conversation = example["conversations"]
    messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : conversation[0]['content'],
        }]
    }]
    true_label = conversation[1]['content']
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )
    outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 4
    )
    out = tokenizer.batch_decode(outputs)
    ff = extract_prediction(out[0])
    if ff == true_label:
        correct += 1

print("Eval accuracy: ",correct/total)