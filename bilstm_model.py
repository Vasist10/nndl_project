# bilstm_model.py
# BiLSTM with Attention — Hate Speech Classifier
# Trained on HateXplain dataset (Normal / Offensive / Hate Speech)

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ─── Config ───────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "hate_model"
MODEL_SAVE_PATH = "hate_model/model.pt"
MAX_LEN         = 128
BATCH_SIZE      = 32
EMBED_DIM       = 128
HIDDEN_DIM      = 256
NUM_LAYERS      = 2
NUM_CLASSES     = 3
DROPOUT         = 0.4
LEARNING_RATE   = 2e-4
EPOCHS          = 10

LABELS = ["normal", "offensive", "hatespeech"]

# ─── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# ─── Preprocessing ────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "<user>", text)            # anonymise mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # strip hashtag symbol
    text = re.sub(r"[^a-z0-9\s]", " ", text)         # remove special chars
    return re.sub(r"\s+", " ", text).strip()

# ─── Dataset ──────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts   = texts
        self.labels  = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = tokenizer.encode(
            clean_text(self.texts[idx]),
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
        )
        # Pad to max_len
        padding   = [0] * (self.max_len - len(token_ids))
        token_ids = token_ids + padding

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─── Model Architecture ───────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # Attention layer — learns which tokens to focus on
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))       # (batch, seq, embed)

        output, _ = self.lstm(embedded)                  # (batch, seq, hidden*2)

        # Attention weights
        attn_scores  = self.attention(output).squeeze(-1)   # (batch, seq)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (batch, seq)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1), output
        ).squeeze(1)                                        # (batch, hidden*2)

        out = self.dropout(context)
        return self.fc(out)                                 # (batch, num_classes)


# ─── Training Loop ────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / (len(loader) * loader.batch_size)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / (len(loader) * loader.batch_size)
    return avg_loss, accuracy


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("../archive/hatexplain.csv")
    df.dropna(inplace=True)

    # Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label_enc"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["label_enc"],
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Dataloaders
    train_loader = DataLoader(
        HateSpeechDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        HateSpeechDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Model, optimizer, loss
    model     = BiLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler — reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")

    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss:   {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✔ Best model saved (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")
