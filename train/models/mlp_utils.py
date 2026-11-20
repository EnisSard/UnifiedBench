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

    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val):
        """Train on (X_train, y_train) with per-sample weights; early-stop on the
        separately held-out, never-augmented validation set (X_val, y_val)."""
        torch.manual_seed(get_seed())
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(reduction='none')

        def t(a, dt=torch.float32):
            return torch.tensor(np.asarray(a), dtype=dt)

        train_ds = TensorDataset(t(X_train), t(y_train, torch.long), t(w_train))
        # drop_last=True prevents a batch of size 1 from crashing BatchNorm1d
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_ds = TensorDataset(t(X_val), t(y_val, torch.long), t(w_val))
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_loss = float('inf')
        patience_counter = 0
        self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        for epoch in range(self.epochs):
            self.model.train()
            for X_b, y_b, w_b in train_loader:
                X_b, y_b, w_b = X_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
                optimizer.zero_grad()
                loss = (ce(self.model(X_b), y_b) * w_b).sum() / w_b.sum()
                loss.backward()
                optimizer.step()

            self.model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for X_b, y_b, w_b in val_loader:
                    X_b, y_b, w_b = X_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
                    num += (ce(self.model(X_b), y_b) * w_b).sum().item()
                    den += w_b.sum().item()
            val_loss = num / max(den, 1e-12)

            if val_loss < best_loss - 0.001:
                best_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
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

def group_validation_split(tr_idx, frac=0.1, seed=None):
    """Positions (into tr_idx) of a ~10% validation slice made of whole clone
    families, and of the remaining training part. Falls back to a random slice
    when no clone-family ids are registered."""
    from train.config import get_groups
    rng = np.random.default_rng(get_seed() if seed is None else seed)
    groups = get_groups()
    n = len(tr_idx)
    if groups is None:
        perm = rng.permutation(n)
        k = max(1, int(frac * n))
        return perm[k:], perm[:k]
    g = np.asarray([groups[i] for i in tr_idx], dtype=object)
    uniq = rng.permutation(np.unique(g))
    val_groups, size = set(), 0
    for u in uniq:
        if size >= frac * n:
            break
        val_groups.add(u)
        size += int((g == u).sum())
    is_val = np.array([x in val_groups for x in g])
    return np.where(~is_val)[0], np.where(is_val)[0]


def run_fused_model_cv(model_name, model_class, texts, labels, source_datasets, splits, persist=True, hp=None):
    hp = {**dict(lr=5e-4, hidden=128, dropout=0.5), **(hp or {})}
    from train.features.fused_features import fused_features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_all = fused_features(texts, device=device)

    def train_fn(X_tr, y_tr, sw, tr_idx):
        # 1) hold out whole clone families for early stopping BEFORE augmentation,
        # 2) augment only the remaining training part (noisy copies keep their weight).
        fit_pos, val_pos = group_validation_split(tr_idx)
        X_fit, y_fit, w_fit = apply_gaussian_augmentation(
            X_tr[fit_pos], y_tr[fit_pos], NUM_LABELS, weights=sw[fit_pos])
        clf = TorchClassifierWrapper(
            model_class=model_class,
            model_kwargs={'input_dim': 406, 'hidden_dim': hp['hidden'], 'num_classes': NUM_LABELS,
                          'dropout': hp['dropout']},
            lr=hp['lr'], device=device
        )
        clf.fit(X_fit, y_fit, w_fit, X_tr[val_pos], y_tr[val_pos], sw[val_pos])
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
        persist         = persist,
    )
