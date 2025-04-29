import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

# Config class to match training settings
class CFG:
    MODEL_NAME = "FacebookAI/xlm-roberta-base"
    TOKENIZER = None
    MAX_LEN = 512
    BS = 64
    N_WORKER = 4

# Initialize tokenizer
CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

# Dataset class (same as training)
class WSDMDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values
        self.labels = df['winner'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = CFG.TOKENIZER.encode_plus(
            self.texts[item],
            return_tensors=None,
            add_special_tokens=True,
            max_length=CFG.MAX_LEN,
            padding='max_length',
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        label = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, label

# Model class (same as training)
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(CFG.MODEL_NAME, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(CFG.MODEL_NAME, config=self.config)
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        feature = outputs[0][:, 0, :]
        output = self.fc(feature)
        return output

def FE(df):
    """Feature engineering function"""
    drop_cols = ['id']
    df.drop(drop_cols, axis=1, inplace=True, errors='ignore')
    
    sep = '[SEP]'
    def text2sentence(text:str='hello world!'):
        return re.split(r'\.|\?|\!|\n', text)
    
    df[['prompt','response_a','response_b']] = df[['prompt','response_a','response_b']].fillna('nan')
    df['text'] = df['prompt'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                 df['prompt'].apply(lambda x:"".join(text2sentence(x)[-2:]))+'.'+sep+\
                 df['response_a'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                 df['response_a'].apply(lambda x:"".join(text2sentence(x)[-2:]))+'.'+sep+\
                 df['response_b'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                 df['response_b'].apply(lambda x:"".join(text2sentence(x)[-2:])+'.')
    
    df.drop(['prompt','response_a','response_b'], axis=1, inplace=True)
    return df

def eval_model(model, test_loader, device):
    """Evaluation function"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            predictions.extend(preds)
    return np.array(predictions)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    test = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_test.parquet')
    test = FE(test)
    y_true_test = test['winner'].values

    # Create test dataset and loader
    test_dataset = WSDMDataset(test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BS,
        shuffle=False,
        num_workers=CFG.N_WORKER,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model
    model = CustomModel()
    model.to(device)

    # Load best model weights
    model_path = f"{CFG.MODEL_NAME.replace('/', '-')}_best.pth"
    print(f"Loading model from {model_path}")
    state = torch.load(model_path)
    model.load_state_dict(state)

    # Evaluate
    y_pred_test = eval_model(model, test_loader, device)
    test_acc = np.mean(y_true_test == y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main() 