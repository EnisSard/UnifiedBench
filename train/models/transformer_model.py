"""
train/models/transformer_model.py
---------------------------------
HuggingFace fine-tuning for the Transformer baselines.

  codebert   -> microsoft/codebert-base   (RoBERTa encoder pre-trained on 6 PLs)
  smartbert  -> web3se/SmartBERT-v3        (encoder further pre-trained on Solidity)

Each fold fine-tunes a *fresh* pre-trained checkpoint on the training split only
(5 epochs, AdamW lr 2e-5, batch 8, linear warm-up 0.1, class-weighted loss,
max sequence length 512, fp16), then predicts the held-out split.  Results are
written to ``<results dir>/<model>/fold_results.json``.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train.config import NUM_LABELS, get_seed
from train.evaluate import get_class_weights
from train.models._real_cv import run_real_cv

CHECKPOINTS = {
    "codebert": "microsoft/codebert-base",
    "smartbert": "web3se/SmartBERT-v3",
}

MAX_LEN = 512
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class _TextDS(Dataset):
    def __init__(self, enc, labels=None):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def _make_fit_predict(checkpoint: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint)

    def _encode(texts):
        return tok(
            list(texts),
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )

    def fit_predict(train_texts, y_train, train_sources, test_texts):
        torch.manual_seed(get_seed())
        np.random.seed(get_seed())

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=NUM_LABELS
        ).to(device)

        tr_ds = _TextDS(_encode(train_texts), y_train)
        te_ds = _TextDS(_encode(test_texts))
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        te_dl = DataLoader(te_ds, batch_size=32, shuffle=False)

        cw = torch.tensor(get_class_weights(y_train), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw)

        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_dl) * EPOCHS
        sched = get_linear_schedule_with_warmup(
            optim, int(WARMUP_RATIO * total_steps), total_steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        model.train()
        for _ in range(EPOCHS):
            for batch in tr_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lab = batch["labels"].to(device)
                optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    loss = loss_fn(logits, lab)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in te_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                preds.extend(logits.argmax(-1).cpu().tolist())

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return preds

    return fit_predict


def run(model_key, texts, labels, source_datasets, splits, persist=True):
    checkpoint = CHECKPOINTS[model_key]
    print(f"  [{model_key}] fine-tuning {checkpoint}  "
          f"({EPOCHS} epochs, bs {BATCH_SIZE}, lr {LR}, max_len {MAX_LEN})")
    return run_real_cv(
        model_key, texts, labels, source_datasets, splits,
        fit_predict_fn=_make_fit_predict(checkpoint), persist=persist,
    )
