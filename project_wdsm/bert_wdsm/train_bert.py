import pandas as pd
import numpy as np
import os#interact with operation system
import re#python's built-in regular expressions.

import torch#pytorch一个开源的深度学习库
import torch.nn as nn#nn模块是神经网络的各种组件(全连接层、卷积层、激活函数等)
#DataSet是抽象类,定义数据集结构和如何访问单个样本. DataLoader是迭代器,批量加载,打乱数据
from torch.utils.data import DataLoader, Dataset

import transformers
from torch.optim import AdamW
#自动分词器,自动模型加载器,自动配置加载器,用于加载模型对应的分词器、模型和参数
from transformers import AutoTokenizer, AutoModel, AutoConfig
#余弦退火的学习率调度器
from transformers import get_cosine_schedule_with_warmup

import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

#分词器并行计算,这样的话处理大型文本数据集能够更快.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
    #operating system random seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    #set pytorch cpu's random seed
    torch.manual_seed(seed)
    #set pytorch gpu's random seed
    torch.cuda.manual_seed(seed)
    #使用固定的算法来执行深度学习操作,例如卷积
    torch.backends.cudnn.deterministic = True
    #在执行操作之前先进行基准测试选择最佳的算法,False就是关闭基准测试,每次使用同样的算法
    torch.backends.cudnn.benchmark = False
seed_everything(seed=2024)


test = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_test.parquet')
train = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_train.parquet')
val = pd.read_parquet('/home/hice1/ajairath8/DL/data/WSDM_valid.parquet')


train = train.drop('fold', axis=1)
test = test.drop('fold', axis=1)
val = val.drop('fold', axis=1)


#Feature Engineer
colnotintest=[col for col in train.columns if col not in test.columns and col!='winner']
colnotintrain=[col for col in test.columns if col not in train.columns]
def FE(df):
    drop_cols=colnotintest+colnotintrain+['id']#meaningless
    df.drop(drop_cols,axis=1,inplace=True,errors='ignore')

    sep='[SEP]'
    #prompt	response_a	response_b
    def text2sentence(text:str='hello world!'):
        return re.split(r'\.|\?|\!|\n',text)
    df[['prompt','response_a','response_b']]=df[['prompt','response_a','response_b']].fillna('nan')
    df['text']= df['prompt'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                df['prompt'].apply(lambda x:"".join(text2sentence(x)[-2:]))+'.'+sep+\
                df['response_a'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                df['response_a'].apply(lambda x:"".join(text2sentence(x)[-2:]))+'.'+sep+\
                df['response_b'].apply(lambda x:"".join(text2sentence(x)[:2]))+'.'+sep+\
                df['response_b'].apply(lambda x:"".join(text2sentence(x)[-2:])+'.')
    df.drop(['prompt','response_a','response_b'],axis=1,inplace=True)

    return df
train=FE(train)
# train['winner']=(train['winner']=='model_a').astype(np.int8)
test=FE(test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
train.head()

val = FE(val)
val.head()

test.head()

"""# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">3.Config</h1></span>"""

from transformers import AutoTokenizer

class CFG:
    MODEL_NAME = "FacebookAI/xlm-roberta-base"
    TOKENIZER = None
    MAX_LEN = 512
    N_EPOCH = 10
    TEST_SIZE = 5000
    BS = 64
    N_WORKER = 4
    LR = 5e-6
    WEIGHT_DECAY = 0.01
    N_WARMUP = 0
    N_CYCLES = 0.5
    GRAD_NORM = 0.1

# Print all the variables in the CFG class
for attr in dir(CFG):
    if not attr.startswith('_'):
        print(f"{attr}: {getattr(CFG, attr)}")

CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

"""# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">4.WSDMDataSet</h1></span>"""

class WSDMDataset(Dataset):
    #初始化X(texts)和y(labels)
    def __init__(self, df):
        self.texts = df['text'].values
        self.labels = df['winner'].values

    #知道有多少训练(验证、测试)数据
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):#item:index
        #encode_plus将文本转成模型能够理解的格式
        inputs = CFG.TOKENIZER.encode_plus(
            self.texts[item], #第item个文本数据.
            return_tensors=None, #返回一个字典
            add_special_tokens=True, #在开头和截尾加上[cls],[slp]
            max_length=CFG.MAX_LEN,#设置最大长度
            padding='max_length',#如果长度小于max_len就padding
            truncation=True#如果长度大于max_len,就进行截断
        )
        #将X和y都转成torch.Tensor
        #k:attention_mask,input_ids,token_type_ids
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        label = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, label


# tes_df = train.iloc[:CFG.TEST_SIZE]
# print(len(tes_df))
# val_df = train.iloc[CFG.TEST_SIZE:2*CFG.TEST_SIZE]
# print(len(val_df))
# print(len(train))
# trn_df = train.iloc[2*CFG.TEST_SIZE:]
# print(len(trn_df))

y_true_valid,y_true_test = val['winner'].values,test['winner'].values

train_dataset = WSDMDataset(train)
valid_dataset = WSDMDataset(val)
test_dataset = WSDMDataset(test)
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.BS,
                          shuffle=True,
                          num_workers=CFG.N_WORKER, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.BS,
                          shuffle=False,
                          num_workers=CFG.N_WORKER, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.BS,
                         shuffle=False,
                        num_workers=CFG.N_WORKER, pin_memory=True, drop_last=False)


#预训练模型
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        #加载预训练模型的参数,然后加载预训练模型,output_hidden_states=True,输出所有隐藏层的状态
        self.config = AutoConfig.from_pretrained(CFG.MODEL_NAME, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(CFG.MODEL_NAME, config=self.config)
        #输出层
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        #[0]是最后一层的隐藏状态,[:, 0, :] [CLS]代表的是整个序列的特征.
        feature = outputs[0][:, 0, :]
        output = self.fc(feature)
        return output

num_train_steps = int(len(train_dataset) / CFG.BS * CFG.N_EPOCH)
model = CustomModel()
model.to(device)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Setup
os.makedirs("checkpoints", exist_ok=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=CFG.LR)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG.N_WARMUP,
    num_training_steps=num_train_steps,
    num_cycles=CFG.N_CYCLES,
)

# Tracking
history = {
    "train_loss": [],
    "valid_loss": [],
    "valid_acc": [],
}

start_epoch = 0
best_score = float('-inf')
checkpoint_path = f"checkpoints/{CFG.MODEL_NAME.replace('/', '-')}_last.pth"

# Resume from checkpoint if exists
if os.path.exists(checkpoint_path):
    print("Resuming from last checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_score = checkpoint['best_score']
    history = checkpoint['history']

# Train step
def train_step(model, train_loader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    loop = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in loop:
        optimizer.zero_grad()
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        labels = labels.to(device)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
    return losses

# Eval step
def eval_step(model, valid_loader, criterion):
    model.eval()
    losses, predicts = [], []
    loop = tqdm(valid_loader, desc="Evaluating", leave=False)
    for inputs, labels in loop:
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        labels = labels.to(device)
        with torch.no_grad():
            y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        losses.append(loss.item())
        predicts.append(y_pred.argmax(1).int().cpu().numpy())
        loop.set_postfix(loss=loss.item())
    return losses, np.hstack(predicts)

# Main training loop
for epoch in range(start_epoch, CFG.N_EPOCH):
    print(f"\nEpoch {epoch + 1}/{CFG.N_EPOCH}")

    train_losses = train_step(model, train_loader, optimizer, scheduler, criterion)
    valid_losses, y_pred_valid = eval_step(model, valid_loader, criterion)
    acc = np.mean(y_true_valid == y_pred_valid)

    history["train_loss"].append(np.mean(train_losses))
    history["valid_loss"].append(np.mean(valid_losses))
    history["valid_acc"].append(acc)

    print(f"Validation Accuracy: {acc:.4f}")

    # Save best model
    if acc > best_score:
        best_score = acc
        torch.save(model.state_dict(), f"{CFG.MODEL_NAME.replace('/', '-')}_best.pth")

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
    epoch_path = f"checkpoints/{CFG.MODEL_NAME.replace('/', '-')}_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'history': history
    }, epoch_path)

# Load best model for testing
state = torch.load(f"{CFG.MODEL_NAME.replace('/', '-')}_best.pth")
model.load_state_dict(state)
_, y_pred_test = eval_step(model, test_loader, criterion)
test_acc = np.mean(y_true_test == y_pred_test)
print(f"Test Score: {test_acc:.4f}")

# Save training history plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["valid_loss"], label="Valid Loss") 
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["valid_acc"], label="Validation Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig(f"{CFG.MODEL_NAME.replace('/', '-')}_training_history.png")
plt.close()




# # prompt: write code to flush gpu memory

# import torch

# def flush_gpu_memory():
#   """
#   This function flushes GPU memory.
#   """
#   torch.cuda.empty_cache()
#   print("GPU memory flushed.")

# # Example usage after model training or inference:
# flush_gpu_memory()

