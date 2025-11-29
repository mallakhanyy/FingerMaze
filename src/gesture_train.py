# gesture_train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LandmarksDataset(Dataset):
    def __init__(self, csv_files):
        rows = []
        for f in csv_files:
            df = pd.read_csv(f)
            rows.append(df)
        df = pd.concat(rows, ignore_index=True)
        # assume columns: ts, idx_x, idx_y, lm0_x... lm20_x, lm0_y... lm20_y, label
        feature_cols = [c for c in df.columns if c not in ['ts','label']]
        self.X = df[feature_cols].fillna(0).values.astype(np.float32)
        labels = df['label'].fillna("none").values
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(labels)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, inp_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self,x): return self.net(x)

def train(csv_files, epochs=20):
    ds = LandmarksDataset(csv_files)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = SimpleMLP(ds.X.shape[1], n_classes=len(ds.le.classes_))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = 0
        loss_sum = 0
        for Xb, yb in loader:
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * Xb.size(0)
            total += Xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} Loss {loss_sum/total:.4f}")
    torch.save({"model":model.state_dict(), "classes": ds.le.classes_}, "models/gesture_mlp.pth")
    print("Saved models/gesture_mlp.pth")

if __name__ == "__main__":
    import glob, os
    os.makedirs("models", exist_ok=True)
    csvs = glob.glob("data/*.csv")
    if not csvs:
        print("No CSV files in data/")
    else:
        train(csvs)
