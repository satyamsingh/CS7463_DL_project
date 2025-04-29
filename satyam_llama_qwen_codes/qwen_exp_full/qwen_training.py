from datasets import load_dataset

# Load training and test data with explicit split mapping
dataset = load_dataset(
    "parquet",
    data_files={
        "val": f"/home/hice1/ssingh934/scratch/deep-learning/project/WSDM_valid.parquet",
        "test": f"/home/hice1/ssingh934/scratch/deep-learning/project/WSDM_test.parquet",
        "train": f"/home/hice1/ssingh934/scratch/deep-learning/project/WSDM_train.parquet",
    }
)

# dataset = {
#     split: dataset[split].shuffle(seed=42).select(range(int(0.01 * len(dataset[split]))))
#     for split in dataset
# }
# Create English datasets
english_train = dataset["train"].filter(lambda x: x["language"] == "English")
english_val = dataset["val"].filter(lambda x: x["language"] == "English")
english_test = dataset["test"].filter(lambda x: x["language"] == "English")

# Create multilingual datasets (non-English)
multilingual_train = dataset["train"].filter(lambda x: x["language"] != "English")
multilingual_test = dataset["test"].filter(lambda x: x["language"] != "English")
multilingual_val = dataset["val"].filter(lambda x: x["language"] != "English")


import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from unsloth import FastLanguageModel
from peft import LoraConfig
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
import csv
import os
from transformers import BitsAndBytesConfig  # Add this line


os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

# Preprocessing function with proper tensor handling
def preprocess_function(examples, tokenizer):
    prompts = [
        f"""You are evaluating two responses to a user prompt. Choose the better response.
        Prompt: {prompt}
        Response A: {resp_a}
        Response B: {resp_b}
        Which response is better? Answer with only a single number, either 'A' or 'B'."""
        for prompt, resp_a, resp_b in zip(examples["prompt"], examples["response_a"], examples["response_b"])
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    if "winner" in examples:
        labels = torch.tensor(examples["winner"], dtype=torch.long)
        seq_len = model_inputs["input_ids"].shape[1]
        model_inputs["labels"] = labels.unsqueeze(1).expand(-1, seq_len)

    return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in model_inputs.items()}

# Enhanced plotting function with test metrics
def plot_metrics(train_losses, val_losses, val_accuracies, test_losses=None, test_accuracies=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    test_x = [epochs[-1] - 0.2, epochs[-1] + 0.2]  # offset to separate dots visually
    plt.scatter(test_x, test_losses, c='g', label='Test Loss')
    # if test_losses:
    #     plt.scatter(epochs[-1], test_losses, c='g', label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'm-', label='Validation Accuracy')
    plt.scatter(test_x, test_accuracies, c='y', label='Test Accuracy')

    # if test_accuracies:
    #     plt.scatter(epochs[-1], test_accuracies, c='y', label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("full_training_progress.png")
    plt.show()

def setup_qwen_with_lora():
    # model_name = "unsloth/qwen-3.2-1B-bnb-4bit"
    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True
    )
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset,
                train_losses, val_losses, val_accuracies,
                epochs=3, batch_size=16, learning_rate=1e-5):

    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {len(train_losses)+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_acc = evaluate_model(model, eval_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {len(train_losses)}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}")

    return model

def evaluate_model(model, eval_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def main():
    # Initialize metric trackers
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    test_metrics = {}

    # Model setup
    model, tokenizer = setup_qwen_with_lora()
    preprocess = partial(preprocess_function, tokenizer=tokenizer)

    # English training phase
    print("Processing English datasets...")
    english_train_processed = english_train.map(preprocess, batched=True, remove_columns=english_train.column_names)
    english_eval_processed = english_val.map(preprocess, batched=True, remove_columns=english_train.column_names)
    english_test_processed = english_test.map(preprocess, batched=True, remove_columns=english_test.column_names)

    print("Training on English data...")
    model = train_model(model, tokenizer,
                       english_train_processed, english_eval_processed,
                       all_train_losses, all_val_losses, all_val_accuracies,
                       epochs=10, batch_size=16, learning_rate=2e-5)


    # english_train_processed = english_train.map(preprocess, batched=True,
    #                                            remove_columns=english_train.column_names,
    #                                            fn_kwargs={"tokenizer": tokenizer})
    # english_test_processed = english_test.map(preprocess, batched=True,
    #                                          remove_columns=english_test.column_names,
    #                                          fn_kwargs={"tokenizer": tokenizer})

    # Training (keep your existing training code)
    data_collator = DataCollatorWithPadding(tokenizer)
    english_test_loader = DataLoader(english_test_processed, batch_size=8, collate_fn=data_collator)
    en_loss, en_acc = evaluate_model(model, english_test_loader, torch.device("cuda"))

    print(f"\nFinal Metrics:\n"
          f"English Test Loss: {en_loss:.4f}, Accuracy: {en_acc:.2%}\n")
    # Save model and create submission
    model.save_pretrained("english_qwen3_wsdm_classifier")

    # Multilingual fine-tuning phase
    print("Processing multilingual datasets...")
    multi_train_processed = multilingual_train.map(preprocess, batched=True, remove_columns=multilingual_train.column_names)
    multi_eval_processed = multilingual_val.map(preprocess, batched=True, remove_columns=multilingual_train.column_names)
    multi_test_processed = multilingual_test.map(preprocess, batched=True, remove_columns=multilingual_test.column_names)

    print("Fine-tuning on multilingual data...")
    model = train_model(model, tokenizer,
                       multi_train_processed, multi_eval_processed,
                       all_train_losses, all_val_losses, all_val_accuracies,
                       epochs=10, batch_size=16, learning_rate=2e-5)

    # Final test evaluation
    print("Evaluating on test set...")
    print("\nEvaluating on English test set:")
    #preprocess = lambda examples: preprocess_function(examples, tokenizer)
    #preprocess = partial(preprocess_function, tokenizer=tokenizer)



    # multilingual_train_processed = multilingual_train.map(preprocess, batched=True,
    #                                                      remove_columns=multilingual_train.column_names,
    #                                                      fn_kwargs={"tokenizer": tokenizer})
    # multilingual_test_processed = multilingual_test.map(preprocess, batched=True,
    #                                                    remove_columns=multilingual_test.column_names,
    #                                                    fn_kwargs={"tokenizer": tokenizer})



    print("\nEvaluating on Multilingual test set:")
    multi_test_loader = DataLoader(multi_test_processed, batch_size=8, collate_fn=data_collator)
    ml_loss, ml_acc = evaluate_model(model, multi_test_loader, torch.device("cuda"))

    print(f"\nFinal Metrics:\n"
          f"English Test Loss: {en_loss:.4f}, Accuracy: {en_acc:.2%}\n"
          f"Multilingual Test Loss: {ml_loss:.4f}, Accuracy: {ml_acc:.2%}")

    test_metrics['loss'] = [en_loss, ml_loss]
    test_metrics['accuracy'] = [en_acc, ml_acc]

    # test_metrics['loss'] = [ml_loss]
    # test_metrics['accuracy'] = [ml_acc]

    # Generate final plots
    plot_metrics(all_train_losses, all_val_losses, all_val_accuracies,
                 test_losses=test_metrics['loss'],
                 test_accuracies=test_metrics['accuracy'])

    # Save model and create submission
    model.save_pretrained("multi_qwen3_wsdm_classifier")

    def create_submission(model, test_dataset, tokenizer, ids, output_file="submission.csv"):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.eval()
      model.to(device)

      predictions = []
      test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer))

      with torch.no_grad():
          for batch in tqdm(test_loader, desc="Generating submission"):
              batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}  # skip labels if present
              outputs = model(**batch)

              # Take model's predicted token at the position of the answer.
              preds = outputs.logits.argmax(dim=-1)  # shape: (batch_size, sequence_length)
              # Decode the full sequence, get last non-pad token, or custom logic depending on your dataset.
              decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)

              # Extract the final answer ("A" or "B") from the decoded string.
              for text in decoded:
                  if "A" in text and "B" in text:
                      # Pick last A/B, assuming model outputs entire prompt + answer.
                      answer = text.strip()[-1]
                      if answer not in ["A", "B"]:
                          answer = "A"  # fallback to A if the output is malformed.
                  else:
                      answer = "A"  # fallback if A/B not found.
                  predictions.append(answer)

      # Save predictions into a CSV.
      with open(output_file, mode="w", newline='') as f:
          writer = csv.writer(f)
          writer.writerow(["id", "prediction"])  # header

          for id_, pred in zip(ids, predictions):
              writer.writerow([id_, pred])

      print(f"Submission saved to {output_file}")

    create_submission(model, multi_test_processed, tokenizer, multi_test_processed["input_ids"])


if __name__ == "__main__":
    main()
