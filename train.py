"""
HATEGuard - BiLSTM Hate Speech Detection Model Training
Dataset: HateXplain (Twitter/Gab posts, 3-class labels)
Labels: 0 = Normal, 1 = Offensive, 2 = Hate Speech
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Config ───────────────────────────────────────────────────────────────────
EMBEDDING_DIM   = 128
HIDDEN_DIM      = 256
NUM_LAYERS      = 2
DROPOUT         = 0.4
NUM_CLASSES     = 3
BATCH_SIZE      = 64
EPOCHS          = 10
LEARNING_RATE   = 1e-3
MAX_SEQ_LEN     = 128
MIN_WORD_FREQ   = 2
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

# ─── Text Preprocessing ───────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "<user>", text)               # anonymise mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # strip hashtag symbol
    text = re.sub(r"[^a-z0-9\s<>]", " ", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq=MIN_WORD_FREQ):
        self.min_freq = min_freq
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        return [
            self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            for token in text.split()
        ]

    def __len__(self):
        return len(self.word2idx)

# ─── Dataset ──────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_SEQ_LEN):
        self.labels = labels
        self.max_len = max_len
        self.encoded = [vocab.encode(t)[:max_len] for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.encoded[idx], dtype=torch.long)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, lbl

def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_padded, torch.stack(labels)

# ─── Model ────────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with:
      - Embedding layer
      - Two BiLSTM layers with dropout
      - Attention pooling over time steps
      - Fully-connected classifier head
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_classes, dropout):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))          # (B, L, E)
        output, _ = self.lstm(embedded)                     # (B, L, 2H)

        # Attention
        attn_weights = torch.softmax(
            self.attention(output).squeeze(-1), dim=1
        )                                                   # (B, L)
        context = torch.bmm(
            attn_weights.unsqueeze(1), output
        ).squeeze(1)                                        # (B, 2H)

        out = self.dropout(context)
        logits = self.fc(out)                               # (B, C)
        return logits

# ─── Training Helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            logits = model(seqs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += seqs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

# ─── Plot Helpers ─────────────────────────────────────────────────────────────
def plot_history(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses,   label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs,   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    plt.show()


def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    names = list(LABEL_MAP.values())
    df_cm = pd.DataFrame(cm_norm, index=names, columns=names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix (normalised)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    # ── Load Data ──────────────────────────────────────────────────────────────
    # HateXplain dataset (https://huggingface.co/datasets/hatexplain)
    # Expected format: CSV with columns 'text' and 'label' (0/1/2)
    # You can also load directly from HuggingFace:
    #   from datasets import load_dataset
    #   ds = load_dataset("hatexplain")
    df = pd.read_csv("hatexplain.csv")          # <- swap path if needed
    df["text"]  = df["text"].apply(clean_text)
    df.dropna(subset=["text", "label"], inplace=True)
    df = df[df["text"].str.strip() != ""]

    texts  = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    print(f"Dataset size : {len(texts)}")
    for k, v in LABEL_MAP.items():
        print(f"  {v}: {labels.count(k)}")

    # ── Split ──────────────────────────────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── Vocabulary ─────────────────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=MIN_WORD_FREQ)
    vocab.build(X_train)
    print(f"Vocabulary size: {len(vocab)}")

    # ── Datasets & Loaders ─────────────────────────────────────────────────────
    train_ds = HateSpeechDataset(X_train, y_train, vocab)
    val_ds   = HateSpeechDataset(X_val,   y_val,   vocab)
    test_ds  = HateSpeechDataset(X_test,  y_test,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        vocab_size  = len(vocab),
        embed_dim   = EMBEDDING_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(DEVICE)

    # Class weights to handle imbalance
    class_counts = [labels.count(i) for i in range(NUM_CLASSES)]
    total        = sum(class_counts)
    weights      = torch.tensor(
        [total / (NUM_CLASSES * c) for c in class_counts],
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # ── Training Loop ──────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  |  "
            f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.4f}"
        )

        # Save best model
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "hate_model/model.pt")
            print("  ✔ Saved best model")

    # ── Final Evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load("hate_model/model.pt", map_location=DEVICE))
    _, test_acc, preds, true_labels = eval_epoch(model, test_loader, criterion)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        true_labels, preds,
        target_names=list(LABEL_MAP.values())
    ))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion(true_labels, preds)

    print("\nDone. Model saved to hate_model/model.pt")


if __name__ == "__main__":
    main()
