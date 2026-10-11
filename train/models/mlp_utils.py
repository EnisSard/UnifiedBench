import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train.evaluate import run_sklearn_cv
from train.config import NUM_LABELS, get_seed
from train.features.fused_features import apply_gaussian_augmentation

class TorchClassifierWrapper:
    def __init__(self, model_class, model_kwargs, epochs=100, batch_size=16, lr=5e-4, weight_decay=1e-3, patience=15, device='cpu'):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.model = None

    def fit(self, X_tr, y_tr, sample_weights=None):
        torch.manual_seed(get_seed())
        
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if sample_weights is not None:
            unique_classes = np.unique(y_tr)
            class_weights = np.ones(NUM_LABELS, dtype=np.float32)
            for c in unique_classes:
                idx = np.where(y_tr == c)[0][0]
                class_weights[c] = sample_weights[idx]
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        indices = np.random.default_rng(get_seed()).permutation(len(X_tr))  # seeded early-stopping split
        val_size = max(1, int(0.1 * len(X_tr)))
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        
        X_train, y_train = X_tr[train_idx], y_tr[train_idx]
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        # Use drop_last=True to prevent batch size of 1 crashing BatchNorm1d during training
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    logits = self.model(X_b)
                    val_loss += criterion(logits, y_b).item() * X_b.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_loss - 0.001:
                best_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break
        
        self.model.load_state_dict(self.best_state)
        return self

    def predict(self, X_te):
        self.model.eval()
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_te_t), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                logits = self.model(X_b)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds)


class DeepSolidHybrid(nn.Module):
    def __init__(self, input_dim=406, hidden_dim=128, num_classes=8, dropout=0.5):
        super().__init__()
        self.static_branch = nn.Sequential(
            nn.Linear(22, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.deep_branch = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, x):
        deep_x = x[:, :384]
        static_x = x[:, 384:]
        s_out = self.static_branch(static_x)
        d_out = self.deep_branch(deep_x)
        merged = torch.cat([s_out, d_out], dim=1)
        return self.classifier(merged)

class HybridVulDetect(nn.Module):
    def __init__(self, input_dim=406, hidden_dim=128, num_classes=8, dropout=0.5):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        merged = torch.cat([b1, b2], dim=1)
        return self.classifier(merged)

def run_fused_model_cv(model_name, model_class, texts, labels, source_datasets, splits):
    from train.features.fused_features import fused_features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_all = fused_features(texts, device=device)

    def train_fn(X_tr, y_tr, cw):
        # Gaussian-noise augmentation of the training split only; the class-weighted
        # loss (weights from the original labels) remains the primary imbalance mechanism.
        X_tr_aug, y_tr_aug = apply_gaussian_augmentation(X_tr, y_tr, NUM_LABELS)
        sample_weights = np.array([cw[y] for y in y_tr_aug])
        clf = TorchClassifierWrapper(
            model_class=model_class,
            model_kwargs={'input_dim': 406, 'hidden_dim': 128, 'num_classes': NUM_LABELS, 'dropout': 0.5},
            device=device
        )
        clf.fit(X_tr_aug, y_tr_aug, sample_weights)

        return clf

    def predict_fn(clf, X_te):
        return clf.predict(X_te).tolist()

    return run_sklearn_cv(
        model_name      = model_name,
        build_features_fn = lambda _: X_all,
        train_model_fn  = train_fn,
        predict_fn      = predict_fn,
        texts           = texts,
        labels          = labels,
        source_datasets = source_datasets,
        splits          = splits,
    )
