import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

# Config class to match training settings
class CFG:
    MODEL_NAME = "xlm-roberta-base"
    TOKENIZER = None
    MAX_LEN = 512
    BS = 32
    N_WORKER = 4

# Initialize tokenizer
CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

# Dataset class (same as training)
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

# Model architecture (same as training)
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

def eval_model(model, test_loader, device):
    """Evaluation function"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get inputs and move to device
            inputs_a = {k: v.to(device) for k, v in batch['inputs_a'].items()}
            inputs_b = {k: v.to(device) for k, v in batch['inputs_b'].items()}
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(inputs_a, inputs_b)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    test = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_test.parquet')
    test = test.drop('fold', axis=1, errors='ignore')

    # Create test dataset and loader
    test_dataset = PairwiseDataset(test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BS,
        shuffle=False,
        num_workers=CFG.N_WORKER,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model
    model = PairwiseRankingModel(CFG.MODEL_NAME)
    model.to(device)

    # Load best model weights
    model_path = f"pairwise/{CFG.MODEL_NAME.replace('/', '-')}_best.pth"
    print(f"Loading model from {model_path}")
    state = torch.load(model_path)
    model.load_state_dict(state)

    # Evaluate
    y_pred_test, y_true_test = eval_model(model, test_loader, device)
    test_acc = np.mean(y_pred_test == y_true_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main() 