import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # type: ignore
from transformers import GemmaTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import log_loss, accuracy_score # type: ignore
from datasets import Dataset # type: ignore
import pandas as pd # type: ignore
import warnings
import torch # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from torch import nn
from transformers import Gemma2PreTrainedModel, Gemma2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils.model_parallel_utils import assert_device_map


## https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma2/modeling_gemma2.py#L943
class Gemma2ForSequenceClassificationMeanPool(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size//2),  
        #     nn.ReLU(),
        #     nn.LayerNorm(config.hidden_size//2),                 
        #     nn.Dropout(p=0.1),
        # )
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None, 
    ) -> tuple | SequenceClassifierOutputWithPast:  

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        #hidden_states = outputs.last_hidden_state
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state



        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = hidden_states.mean(dim=1)

        #logits = self.bottleneck(pooled)
        logits = self.score(pooled)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #loss = self.loss_function(logits=logits, labels=labels, pooled_logits=logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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

CHECKPOINT_BASE_NAME = f"/home/hice1/akhandelwal83/scratch/deep-learning/project/checkpoints/gemma-2-2b-it-bnb-4bit-2048-16--3"
LOGS = f"{CHECKPOINT_BASE_NAME}/logs/"

if not os.path.exists(LOGS):
    os.makedirs(LOGS)

train = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_train.parquet')
val = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_valid.parquet')

print("Dataset Loaded")

train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

tokenizer = GemmaTokenizerFast.from_pretrained(checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

print("Dataset Loaded")

class Tokenizer:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        template = """  You will be given a prompt and responses from two different AI models. The prompt and responses may be in any language.
        Your task is to predict which response a human would prefer. 

        Think carefully before answering, and respond with a score of 0 for **response_a** or 1 for **response_b**."""
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

model = Gemma2ForSequenceClassificationMeanPool.from_pretrained(
    checkpoint,
    num_labels=2,
    torch_dtype=torch.float16,
    device_map="auto",
)

# for param in model.bottleneck.parameters():
#     if param.dtype.is_floating_point: 
#         param.requires_grad = True

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
    report_to="tensorboard",
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

logs = pd.DataFrame(trainer.state.log_history)
logs.to_csv(f"{LOGS}/training_logs.csv", index=False)

plt.plot(logs["step"], logs["loss"], label="Training Loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(f"{LOGS}/loss_curve.png")

y_true = val["winner"]
logits = trainer.predict(val).predictions
y_pred_probs = torch.from_numpy(logits).float().softmax(-1).numpy()

acc = accuracy_score(y_true=y_true, y_pred=y_pred_probs.argmax(-1))
print(f"Fold - Accuracy: {acc:.4f}")

test = pd.read_parquet('/home/hice1/akhandelwal83/scratch/deep-learning/project/dataset/WSDM_test.parquet')
test = Dataset.from_pandas(test)

test = test.map(encode, batched=True)

y_true = test["winner"]
logits = trainer.predict(test).predictions
y_pred_probs = torch.from_numpy(logits).float().softmax(-1).numpy()

acc = accuracy_score(y_true=y_true, y_pred=y_pred_probs.argmax(-1))
print(f"Fold - Accuracy: {acc:.4f}")