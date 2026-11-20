"""
train/models/transformer_model.py
---------------------------------
HuggingFace fine-tuning for the Transformer baselines.

  codebert   -> microsoft/codebert-base   (RoBERTa encoder pre-trained on 6 PLs)
  smartbert  -> web3se/SmartBERT-v3        (encoder further pre-trained on Solidity)

Each fold fine-tunes a *fresh* pre-trained checkpoint on the training split only
(5 epochs, AdamW lr 2e-5, batch 8, linear warm-up 0.1, weighted loss, fp16), then
predicts the held-out split.

Input length. Without chunking (profile "base") each contract is truncated to its
first 512 tokens. With chunking (profiles runA/runB) the whole contract is split
into consecutive 510-token chunks (+ <s> and </s>); when a contract has more than
MAX_CHUNKS chunks, MAX_CHUNKS of them are taken evenly spaced over the contract
(always including the first and the last). Every chunk is a training example with
its contract's label and weight / n_chunks, so each contract carries the same total
weight however long it is; at test time the softmax probabilities of a contract's
chunks are averaged.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train.config import NUM_LABELS, MAX_CHUNKS, PROFILE, get_seed
from train.models._real_cv import run_real_cv
from train.features.fused_features import select_chunks

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


class _ChunkDS(Dataset):
    def __init__(self, ids, mask, labels=None, weights=None):
        self.ids, self.mask, self.labels, self.weights = ids, mask, labels, weights

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, i):
        item = {"input_ids": self.ids[i], "attention_mask": self.mask[i]}
        if self.labels is not None:
            lab = self.labels[i]
            item["labels"] = torch.as_tensor(lab, dtype=torch.float32) if np.ndim(lab) else int(lab)
            item["weight"] = float(self.weights[i])
        return item


def _make_fit_predict(checkpoint: str, hp: dict | None = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    hp = {**dict(lr=LR, epochs=EPOCHS), **(hp or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint)
    body = MAX_LEN - 2
    _tok_cache: dict = {}

    def _chunks(text):
        if text not in _tok_cache:
            _tok_cache[text] = tok(text, add_special_tokens=False, truncation=False,
                                   verbose=False)["input_ids"]
        ids = _tok_cache[text]
        if not PROFILE["chunking"]:
            return [ids[:body]]
        pieces = [ids[j:j + body] for j in range(0, max(len(ids), 1), body)] or [[]]
        return select_chunks(pieces, MAX_CHUNKS, tok.decode)

    def _encode(texts):
        """Returns (input_ids, attention_mask, owner) for all chunks of `texts`."""
        rows, owner = [], []
        for i, t in enumerate(texts):
            for c in _chunks(t):
                rows.append([tok.cls_token_id] + c + [tok.sep_token_id])
                owner.append(i)
        ids = torch.full((len(rows), MAX_LEN), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), MAX_LEN), dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = torch.tensor(row)
            mask[r, :len(row)] = 1
        return ids, mask, np.asarray(owner)

    def fit_predict(train_texts, y_train, train_sources, test_texts, sample_weights=None):
        torch.manual_seed(get_seed())
        np.random.seed(get_seed())

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=NUM_LABELS
        ).to(device)

        sw = np.ones(len(train_texts)) if sample_weights is None else np.asarray(sample_weights, float)
        y_train = np.asarray(y_train)
        soft = PROFILE["multilabel"]
        if soft:
            # rows of the same multi-label contract (identical text) -> one example whose
            # target is the weight-proportional distribution over its labels
            groups: dict = {}
            for i, t in enumerate(train_texts):
                groups.setdefault(t, []).append(i)
            train_texts = list(groups)
            targets = np.zeros((len(train_texts), NUM_LABELS))
            for j, t in enumerate(train_texts):
                for i in groups[t]:
                    targets[j, y_train[i]] += sw[i]
            sw = targets.sum(axis=1)
            targets = targets / sw[:, None]
        test_texts = list(test_texts)
        uniq_test = list(dict.fromkeys(test_texts))
        tr_ids, tr_mask, tr_owner = _encode(train_texts)
        n_chunks = np.bincount(tr_owner, minlength=len(train_texts))
        chunk_w = sw[tr_owner] / n_chunks[tr_owner]
        chunk_w = chunk_w / chunk_w.mean()
        chunk_y = targets[tr_owner] if soft else y_train[tr_owner]

        te_ids, te_mask, te_owner = _encode(uniq_test)
        tr_dl = DataLoader(_ChunkDS(tr_ids, tr_mask, chunk_y, chunk_w),
                           batch_size=BATCH_SIZE, shuffle=True)
        te_dl = DataLoader(_ChunkDS(te_ids, te_mask), batch_size=32, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_dl) * hp["epochs"]
        sched = get_linear_schedule_with_warmup(
            optim, int(WARMUP_RATIO * total_steps), total_steps
        )
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        model.train()
        for _ in range(hp["epochs"]):
            for batch in tr_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lab = batch["labels"].to(device)
                w = batch["weight"].to(device, dtype=torch.float32)
                optim.zero_grad()
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                if soft:
                    per = -(lab.float() * torch.log_softmax(logits.float(), -1)).sum(-1)
                else:
                    per = loss_fn(logits.float(), lab)
                loss = (per * w).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for batch in te_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, NUM_LABELS))
        upred = {t: int(probs[te_owner == i].mean(axis=0).argmax()) for i, t in enumerate(uniq_test)}
        preds = [upred[t] for t in test_texts]

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return preds

    return fit_predict


def run(model_key, texts, labels, source_datasets, splits, persist=True, hp=None):
    checkpoint = CHECKPOINTS[model_key]
    print(f"  [{model_key}] fine-tuning {checkpoint}  ({EPOCHS} epochs, bs {BATCH_SIZE}, lr {LR}, "
          f"max_len {MAX_LEN}, chunking={'on, max ' + str(MAX_CHUNKS) if PROFILE['chunking'] else 'off'})")
    return run_real_cv(
        model_key, texts, labels, source_datasets, splits,
        fit_predict_fn=_make_fit_predict(checkpoint, hp), persist=persist,
    )
