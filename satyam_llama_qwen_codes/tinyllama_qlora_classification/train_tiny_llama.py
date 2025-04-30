from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

set_seed(42)

dataset = load_dataset(
    "parquet",
    data_files={
        "val": "WSDM_valid.parquet",
        "test": "WSDM_test.parquet",
        "train": "WSDM_train.parquet",
    }
)

dataset = {
    split: dataset[split].shuffle(seed=42).select(range(int(0.01 * len(dataset[split]))))
    for split in dataset
}

# English / Multilingual split
english_train = dataset["train"].filter(lambda x: x["language"] == "English")
english_val = dataset["val"].filter(lambda x: x["language"] == "English")
english_test = dataset["test"].filter(lambda x: x["language"] == "English")

multi_train = dataset["train"].filter(lambda x: x["language"] != "English")
multi_val = dataset["val"].filter(lambda x: x["language"] != "English")
multi_test = dataset["test"].filter(lambda x: x["language"] != "English")

# === Print Dataset Sizes ===
print(f"English Train Size: {len(english_train)}")
print(f"English Validation Size: {len(english_val)}")
print(f"English Test Size: {len(english_test)}")
print(f"Multilingual Train Size: {len(multi_train)}")
print(f"Multilingual Validation Size: {len(multi_val)}")
print(f"Multilingual Test Size: {len(multi_test)}")

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token  # Some models lack a default pad_token

#=== Preprocess Function ===
def preprocess_function(examples, tokenizer):
    prompts = [
        f"""You are evaluating two responses to a user prompt. Choose the better response.
        Prompt: {prompt}
        Response A: {resp_a}
        Response B: {resp_b}
        Which response is better? Answer with only a single letter, either "A" or "B"."""
        for prompt, resp_a, resp_b in zip(examples["prompt"], examples["response_a"], examples["response_b"])
    ]

    model_inputs = tokenizer(prompts, max_length=1024, truncation=True, padding="max_length")

    if "winner" in examples:
        model_inputs["labels"] = examples["winner"]

    return model_inputs

preprocess_function_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)

# Map datasets
def preprocess_dataset(dataset_split):
    return dataset_split.map(
        preprocess_function_with_tokenizer,
        batched=True,
        remove_columns=dataset_split.column_names
    )

encoded_english_train = preprocess_dataset(english_train)
encoded_english_val = preprocess_dataset(english_val)
encoded_english_test = preprocess_dataset(english_test)

encoded_multi_train = preprocess_dataset(multi_train)
encoded_multi_val = preprocess_dataset(multi_val)
encoded_multi_test = preprocess_dataset(multi_test)

# PyTorch format
for ds in [
    encoded_english_train, encoded_english_val, encoded_english_test,
    encoded_multi_train, encoded_multi_val, encoded_multi_test
]:
    ds.set_format(type="torch")

model = AutoModelForSequenceClassification.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=2,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article
training_args = TrainingArguments(
    output_dir="./tinyllama_qlora_classification",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # pretend 4 samples per step
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_dir="./logs",
    report_to="none",   # no wandb
    save_total_limit=2,
    logging_steps=10,
    do_train=True,
    do_eval=True,
    no_cuda=True,
)

# Metric Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    return {"accuracy": acc}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Train English Dataset
trainer_english = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_english_train,
    eval_dataset=encoded_english_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer_english.train()
trainer_english.save_model("./tinyllama_qlora_classification/english_final")

# === Train Multilingual Dataset ===
model_multi = AutoModelForSequenceClassification.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=2,
)
model_multi = prepare_model_for_kbit_training(model_multi)
model_multi = get_peft_model(model_multi, lora_config)

trainer_multi = Trainer(
    model=model_multi,
    args=training_args,
    train_dataset=encoded_multi_train,
    eval_dataset=encoded_multi_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer_multi.train()
trainer_multi.save_model("./tinyllama_qlora_classification/multilingual_final")

# === Final Test Evaluation ===
pred_english = trainer_english.predict(encoded_english_test)
acc_english = np.mean(np.argmax(pred_english.predictions, axis=-1) == pred_english.label_ids)

pred_multi = trainer_multi.predict(encoded_multi_test)
acc_multi = np.mean(np.argmax(pred_multi.predictions, axis=-1) == pred_multi.label_ids)

print(f"\n English Test Accuracy: {acc_english * 100:.2f}%")
print(f"\n Multilingual Test Accuracy: {acc_multi * 100:.2f}%")

def plot_and_save_metrics(trainer, split_name, test_dataset, test_results, output_dir="./tinyllama_qlora_classification"):
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    steps = [log["step"] for log in logs if "loss" in log]
    eval_steps = [log["step"] for log in logs if "eval_loss" in log]

    # === Test Loss and Accuracy ===
    test_loss = test_results["test_loss"]
    test_accuracy = test_results["test_accuracy"]
    test_step = trainer.state.global_step  # <-- much better!

    plt.figure(figsize=(14, 6))

    # === Loss Plot ===
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, label="Train Loss", marker='o')
    plt.plot(eval_steps, eval_loss, label="Validation Loss", marker='x')
    plt.scatter(test_step, test_loss, label="Test Loss", color='red', marker='*', s=150)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"{split_name} Loss Curve")
    plt.legend()

    # === Accuracy Plot ===
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_accuracy, label="Validation Accuracy", marker='x', color='green')
    plt.scatter(test_step, test_accuracy, label="Test Accuracy", color='purple', marker='*', s=150)
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.title(f"{split_name} Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{split_name.lower()}_full_training_plot.png")
    plt.close()


# Evaluate on test set
test_results_english = trainer_english.evaluate(encoded_english_test, metric_key_prefix="test")
test_results_multi = trainer_multi.evaluate(encoded_multi_test, metric_key_prefix="test")
print(f"\n English Test Accuracy: {test_results_english['test_accuracy'] * 100:.2f}%")
print(f" English Test Loss: {test_results_english['test_loss']:.4f}")

print(f"\n Multilingual Test Accuracy: {test_results_multi['test_accuracy'] * 100:.2f}%")
print(f" Multilingual Test Loss: {test_results_multi['test_loss']:.4f}")

# # Then plot
plot_and_save_metrics(trainer_english, "English", encoded_english_test, test_results_english)
plot_and_save_metrics(trainer_multi, "Multilingual", encoded_multi_test, test_results_multi)
