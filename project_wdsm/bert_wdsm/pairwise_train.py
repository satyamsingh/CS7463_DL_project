import pandas as pd
import numpy as np
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set random seed for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(seed=2024)

# Configuration
class CFG:
    MODEL_NAME = "xlm-roberta-base"  # multilingual model
    TOKENIZER = None
    MAX_LEN = 512
    N_EPOCH = 10
    BS = 32
    N_WORKER = 4
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    N_WARMUP = 0
    N_CYCLES = 0.5
    GRAD_NORM = 0.1

# Load data
train = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_train.parquet')
val = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_valid.parquet')
test = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_test.parquet')

train = train.drop('fold', axis=1, errors='ignore')
val = val.drop('fold', axis=1, errors='ignore')
test = test.drop('fold', axis=1, errors='ignore')

# Create directories for saving
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("pairwise", exist_ok=True)

# Prepare datasets
CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class for pairwise ranking
class PairwiseDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        prompt = row['prompt'] if not pd.isna(row['prompt']) else ""
        response_a = row['response_a'] if not pd.isna(row['response_a']) else ""
        response_b = row['response_b'] if not pd.isna(row['response_b']) else ""
        
        # Use winner directly as it's already binary
        label = row['winner']
        
        # Encode both responses paired with the prompt
        encoded_a = CFG.TOKENIZER(prompt, response_a,
                              padding='max_length', truncation=True,
                              max_length=CFG.MAX_LEN, return_tensors='pt')
        
        encoded_b = CFG.TOKENIZER(prompt, response_b,
                              padding='max_length', truncation=True,
                              max_length=CFG.MAX_LEN, return_tensors='pt')
        
        inputs_a = {k: v.squeeze(0) for k, v in encoded_a.items()}
        inputs_b = {k: v.squeeze(0) for k, v in encoded_b.items()}
        
        return {
            'inputs_a': inputs_a,
            'inputs_b': inputs_b,
            'label': torch.tensor(label, dtype=torch.float)
        }

# Model architecture for pairwise ranking
class PairwiseRankingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 1)
    
    def encode(self, inputs):
        outputs = self.encoder(**inputs)
        return outputs.pooler_output
    
    def forward(self, inputs_a, inputs_b):
        output_a = self.encode(inputs_a)
        output_b = self.encode(inputs_b)
        combined = torch.cat([output_a, output_b], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)

# Prepare data loaders
train_dataset = PairwiseDataset(train)
valid_dataset = PairwiseDataset(val)
test_dataset = PairwiseDataset(test)

train_loader = DataLoader(train_dataset, batch_size=CFG.BS, shuffle=True, 
                         num_workers=CFG.N_WORKER, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.BS, shuffle=False, 
                         num_workers=CFG.N_WORKER, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=CFG.BS, shuffle=False, 
                        num_workers=CFG.N_WORKER, pin_memory=True, drop_last=False)

# Initialize model and optimizer
model = PairwiseRankingModel(CFG.MODEL_NAME)
model.to(device)

# Setup training components
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
num_training_steps = int(len(train_dataset) / CFG.BS * CFG.N_EPOCH)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG.N_WARMUP,
    num_training_steps=num_training_steps,
    num_cycles=CFG.N_CYCLES,
)

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("pairwise", exist_ok=True)

# Tracking metrics
history = {
    "train_loss": [],
    "valid_loss": [],
    "valid_acc": [],
}

start_epoch = 0
best_score = float('-inf')
checkpoint_path = f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_last.pth"

# Resume from checkpoint if exists
if os.path.exists(checkpoint_path):
    print("Resuming from last checkpoint...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_score = checkpoint['best_score']
    history = checkpoint['history']

# Training function
def train_step(model, train_loader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    loop = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in loop:
        optimizer.zero_grad()
        
        # Get inputs and move to device
        inputs_a = {k: v.to(device) for k, v in batch['inputs_a'].items()}
        inputs_b = {k: v.to(device) for k, v in batch['inputs_b'].items()}
        labels = batch['label'].to(device)
        
        # Forward pass
        logits = model(inputs_a, inputs_b)
        loss = criterion(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
    
    return losses

# Evaluation function
def eval_step(model, valid_loader, criterion):
    model.eval()
    losses = []
    predicts = []
    true_labels = []
    
    loop = tqdm(valid_loader, desc="Evaluating", leave=False)
    
    for batch in loop:
        # Get inputs and move to device
        inputs_a = {k: v.to(device) for k, v in batch['inputs_a'].items()}
        inputs_b = {k: v.to(device) for k, v in batch['inputs_b'].items()}
        labels = batch['label'].to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(inputs_a, inputs_b)
        
        loss = criterion(logits, labels)
        losses.append(loss.item())
        
        # Convert logits to predictions (0 or 1)
        preds = (torch.sigmoid(logits) > 0.5).float()
        predicts.append(preds.cpu().numpy())
        true_labels.append(labels.cpu().numpy())
        
        loop.set_postfix(loss=loss.item())
    
    predicts = np.concatenate(predicts)
    true_labels = np.concatenate(true_labels)
    
    return losses, predicts, true_labels

# Main training loop
for epoch in range(start_epoch, CFG.N_EPOCH):
    print(f"\nEpoch {epoch + 1}/{CFG.N_EPOCH}")
    
    # Training
    train_losses = train_step(model, train_loader, optimizer, scheduler, criterion)
    train_loss = np.mean(train_losses)
    print(f"Train Loss: {train_loss:.4f}")
    
    # Validation
    valid_losses, y_pred, y_true = eval_step(model, valid_loader, criterion)
    valid_loss = np.mean(valid_losses)
    valid_acc = np.mean(y_pred == y_true)
    
    # Track metrics
    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)
    
    print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.4f}")
    
    # Save best model
    if valid_acc > best_score:
        best_score = valid_acc
        print(f"New best model with accuracy: {valid_acc:.4f}")
        torch.save(model.state_dict(), f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_best.pth")
    
    # Save checkpoint for resuming
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'history': history
    }, checkpoint_path)
    
    # Save epoch-specific checkpoint
    epoch_path = f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'history': history
    }, epoch_path)

# Load best model for testing
print("\nEvaluating on test set...")
best_model_path = f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_best.pth"
if os.path.exists(best_model_path):
    state = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(state)
    
_, test_preds, test_true = eval_step(model, test_loader, criterion)
test_acc = np.mean(test_preds == test_true)
print(f"Test Accuracy: {test_acc:.4f}")

# Save training history plots
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["valid_loss"], label="Valid Loss") 
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history["valid_acc"], label="Validation Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig(f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_training_history.png")
plt.close()

# Function to flush GPU memory
def flush_gpu_memory():
    """This function flushes GPU memory."""
    torch.cuda.empty_cache()
    print("GPU memory flushed.")

# Flush GPU memory after training
# flush_gpu_memory() 