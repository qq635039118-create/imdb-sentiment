import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os
import re

# ========== 配置 ==========
GLOVE_PATH = "tiny_glove.json"
DATA_PATH = "imdb_top_500.csv"
MAX_LEN = 100
EMBED_DIM = 50
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# ========== 加载 GloVe ==========
print("Loading GloVe embeddings...")
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    glove = json.load(f)

word2idx = {"<PAD>": 0, "<UNK>": 1}
embedding_matrix = [np.zeros(EMBED_DIM), np.random.randn(EMBED_DIM) * 0.01]

for word, vec in glove.items():
    word2idx[word] = len(word2idx)
    embedding_matrix.append(np.array(vec, dtype=np.float32))

embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
VOCAB_SIZE = len(word2idx)
print(f"Vocab size: {VOCAB_SIZE}")

# ========== 文本预处理 ==========
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def text_to_indices(text, max_len=MAX_LEN):
    tokens = tokenize(text)
    indices = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens[:max_len]]
    if len(indices) < max_len:
        indices += [word2idx["<PAD>"]] * (max_len - len(indices))
    return indices

# ========== 加载数据 ==========
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

text_col = None
label_col = None
for col in df.columns:
    if col.lower() in ["review", "text", "sentence", "comment"]:
        text_col = col
    if col.lower() in ["sentiment", "label", "polarity", "score"]:
        label_col = col

if text_col is None:
    text_col = df.columns[0]
if label_col is None:
    label_col = df.columns[1]

print(f"Using text column: '{text_col}', label column: '{label_col}'")

labels = df[label_col].values
if labels.dtype == object:
    unique_labels = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    print(f"Label mapping: {label_map}")

X = np.array([text_to_indices(str(text)) for text in df[text_col].values])
y = labels.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Dataset & DataLoader ==========
class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(IMDBDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(IMDBDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ========== 模型定义 ==========
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=True,
            padding_idx=0
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        h = self.dropout(h)
        out = self.fc(h)
        return self.sigmoid(out).squeeze()

model = SentimentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, embedding_matrix)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# ========== 训练 ==========
print("\nStarting training...")
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)
    
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pt")
        print(f"  -> Saved best model (acc={acc:.4f})")

print(f"\nBest test accuracy: {best_acc:.4f}")

# ========== 保存配置 ==========
config = {
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM,
    "max_len": MAX_LEN,
    "word2idx": word2idx,
}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Saved config.json")

# ========== 上传到 Hugging Face ==========
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    repo_id = os.environ.get("HF_REPO", "qq635039118/imdb-sentiment")
    api = HfApi()
    
    print(f"\nUploading to Hugging Face: {repo_id}")
    api.create_repo(repo_id, token=HF_TOKEN, exist_ok=True)
    
    for filename in ["model.pt", "config.json"]:
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            token=HF_TOKEN,
        )
        print(f"  Uploaded {filename}")
    
    print(f"Done! https://huggingface.co/{repo_id}")
else:
    print("\nHF_TOKEN not set, skipping upload.")
