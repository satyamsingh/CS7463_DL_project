import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # type: ignore
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import log_loss, accuracy_score # type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from datasets import Dataset # type: ignore
import pandas as pd # type: ignore
import warnings
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore

warnings.filterwarnings("ignore")

checkpoint = "unsloth/gemma-2-2b-it-bnb-4bit"
max_length = 2048
optim_type = "adamw_8bit"
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
gradient_accumulation_steps = 4
freeze_layers = 12
n_epochs = 2
lr = 1e-4
warmup_steps = 50
lora_r = 16
lora_alpha = 32
lora_dropout = 0.15

CHECKPOINT_BASE_NAME = f"/home/hice1/akhandelwal83/scratch/deep-learning/project/checkpoints/gemma-2-2b-it-bnb-4bit-2048-16--1"
LOGS = f"{CHECKPOINT_BASE_NAME}/logs/"

if not os.path.exists(LOGS):
    os.makedirs(LOGS)

train = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_train.parquet')
val = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_valid.parquet')

train['winner'] = train['winner'].apply(lambda x: "response_b" if x == 1 else "response_a")
val['winner'] = val['winner'].apply(lambda x: "response_b" if x == 1 else "response_a")

print("Dataset Loaded")

train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

tokenizer = GemmaTokenizerFast.from_pretrained(checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

class Tokenizer:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        template = """  You will be given a prompt and responses from two different AI models. The prompt and responses may be in any language.
        Your task is to predict which response a human would prefer. 

        Think carefully before answering, and respond with only one word: either **response_a** or **response_b**."""
        prompt = ["<prompt>: " + t for t in batch["prompt"]]
        model_response_a = ["<response_a>: " + t for t in batch["response_a"]]
        model_response_b = ["<response_b>: " + t for t in batch["response_b"]]
        texts = [template + p + r_a + r_b for p, r_a, r_b in zip(prompt, model_response_a, model_response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        return {**tokenized, "labels": batch["winner"]}
    

encode = Tokenizer(tokenizer, max_length=max_length)

train = train.map(encode, batched=True)
val = val.map(encode, batched=True)

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=[i for i in range(42) if i >= freeze_layers],
    lora_dropout=lora_dropout,
    task_type=TaskType.SEQ_CLS,
)

model = Gemma2ForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def compute_metrics(eval_predictions):
    predictions = eval_predictions.predictions
    true_labels = eval_predictions.label_ids
    probs = torch.from_numpy(predictions).float().softmax(-1).numpy()
    loss = log_loss(y_true=true_labels, y_pred=probs)
    acc = accuracy_score(y_true=true_labels, y_pred=predictions.argmax(-1))
    return {"acc": acc, "log_loss": loss}

training_args = TrainingArguments(
    output_dir=CHECKPOINT_BASE_NAME,
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    logging_dir=LOGS,
    eval_strategy="steps",
    eval_steps = 250,
    save_strategy="steps",
    save_steps=50,
    logging_strategy="steps",
    logging_steps=1,
    save_total_limit=3,
    optim=optim_type,
    fp16=True,
    learning_rate=lr,
    warmup_steps=warmup_steps,
    ddp_find_unused_parameters=False,
    lr_scheduler_type = "cosine",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    torch_compile=True,
    weight_decay = 0.01
)

trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=val,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train(resume_from_checkpoint=True)

df = pd.DataFrame(trainer.state.log_history)

smooth = 0.85
smoothed_values = []
last = df['Value'].iloc[0] 

for val in df['Value']:
    smoothed_val = last * smooth + (1 - smooth) * val
    smoothed_values.append(smoothed_val)
    last = smoothed_val

df['Smoothed Value'] = smoothed_values
plt.figure(figsize=(10,6))


plt.plot(df['Step'], df['Value'], color='gray', alpha=0.3, label='Raw Data')
plt.plot(df['Step'], df['Smoothed Value'], color='#1f77b4', linewidth=2, label='Smoothed Curve')

plt.xlabel('Steps')
plt.ylabel('Value')
plt.title('Training Loss for Generative head')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"loss_t2.png")

y_true = val["winner"]
logits = trainer.predict(val).predictions
y_pred_probs = torch.from_numpy(logits).float().softmax(-1).numpy()

acc = accuracy_score(y_true=y_true, y_pred=y_pred_probs.argmax(-1))
print(f"Accuracy: {acc:.4f}")

test = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_test.parquet')
test = Dataset.from_pandas(test)

test = test.map(encode, batched=True)

y_true = test["winner"]
logits = trainer.predict(test).predictions
y_pred_probs = torch.from_numpy(logits).float().softmax(-1).numpy()

acc = accuracy_score(y_true=y_true, y_pred=y_pred_probs.argmax(-1))
print(f"Accuracy: {acc:.4f}")


