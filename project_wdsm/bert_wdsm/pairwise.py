import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_scheduler
from tqdm import tqdm
import pandas as pd

MODEL_NAME = "xlm-roberta-base"  # multilingual model
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dataset
class PairwiseDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        prompt = row['prompt']
        response_a = row['response_a']
        response_b = row['response_b']
        label = row['label']  # 1 if response_a is better, 0 if response_b is better

        encoded_a = self.tokenizer(prompt, response_a,
                                   padding='max_length', truncation=True,
                                   max_length=MAX_LEN, return_tensors='pt')
        encoded_b = self.tokenizer(prompt, response_b,
                                   padding='max_length', truncation=True,
                                   max_length=MAX_LEN, return_tensors='pt')

        return {
            'input_ids_a': encoded_a['input_ids'].squeeze(),
            'attention_mask_a': encoded_a['attention_mask'].squeeze(),
            'input_ids_b': encoded_b['input_ids'].squeeze(),
            'attention_mask_b': encoded_b['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# 2. Model
class PairwiseRankingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        output_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a).pooler_output
        output_b = self.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b).pooler_output
        combined = torch.cat([output_a, output_b], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze()

# 3. Training function
def train_fn(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids_a = batch['input_ids_a'].to(device)
        attention_mask_a = batch['attention_mask_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        attention_mask_b = batch['attention_mask_b'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 4. Main function
def main():
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')  # it should have: prompt, response_a, response_b, label
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = PairwiseDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PairwiseRankingModel(MODEL_NAME).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(dataloader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        avg_loss = train_fn(model, dataloader, optimizer, scheduler, criterion)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "pairwise_ranking_model.pt")

if __name__ == "__main__":
    main()
