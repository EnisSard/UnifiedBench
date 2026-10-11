"""
train/models/gnn_model.py
-------------------------
Graph Convolutional Network (GCN / Graph Neural Network) baseline for smart contract
vulnerability classification. It is the representative graph-based model in the
benchmark, alongside the feature-based, embedding-based, MLP and Transformer models.

Architecture:
  - Graph Construction:
      * Nodes: Extracted functions / basic code blocks from Solidity contract.
        If contract has <= 1 function, splits its lines into chunks of max(5, n_lines // 6) lines.
      * Node Features: 384-dimensional all-MiniLM-L6-v2 embedding of the node's code
        (first 1,000 characters of the block; the encoder itself reads at most 256 tokens).
      * Edges: sequential edges between consecutive blocks (both directions), self-loops,
        and call edges from a block to every function whose name appears in its text.
  - Graph Neural Network:
      * 2-layer Graph Convolutional Network (GCNConv) with Batch Normalization,
        ReLU non-linearities, and Dropout.
      * Global mean & max pooling for graph-level representation.
      * Final classification MLP mapping to 6 canonical vulnerability classes.
"""

import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import train.evaluate as _ev
from train.config import NUM_LABELS, get_seed
from train.evaluate import compute_metrics, aggregate_folds, get_class_weights
from sentence_transformers import SentenceTransformer


class ContractGCN(nn.Module):
    """
    Contract-level Graph Convolutional Network.
    """
    def __init__(self, in_channels: int = 384, hidden_channels: int = 128, out_channels: int = NUM_LABELS, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Global pooling (mean + max pooling concatenated)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        out = self.classifier(x_graph)
        return out


def build_contract_graph(contract_text: str, embedder: SentenceTransformer, device: str = 'cpu') -> Data:
    """
    Constructs a PyG Data object (Function Call & Flow Graph) from contract text.
    """
    # 1. Parse functions
    func_pattern = re.compile(r'function\s+([a-zA-Z0-9_]+)\s*\((.*?)\)[^{]*\{', re.MULTILINE)
    matches = list(func_pattern.finditer(contract_text))
    
    nodes_text = []
    func_names = []
    
    if len(matches) > 1:
        for i, match in enumerate(matches):
            fname = match.group(1)
            start_pos = match.start()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(contract_text)
            fbody = contract_text[start_pos:end_pos].strip()
            nodes_text.append(fbody[:1000]) # Cap length
            func_names.append(fname)
    else:
        # Fallback: divide the non-empty lines into about six chunks (at least 5 lines each)
        lines = [line.strip() for line in contract_text.split('\n') if line.strip()]
        chunk_size = max(5, len(lines) // 6)
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i+chunk_size])
            nodes_text.append(chunk[:1000])
            func_names.append(f"block_{i}")

    if not nodes_text:
        nodes_text = [contract_text[:1000]]
        func_names = ["main"]

    # 2. Compute Node Embeddings
    node_embeds = embedder.encode(nodes_text, convert_to_numpy=True, show_progress_bar=False)
    x = torch.tensor(node_embeds, dtype=torch.float32)

    # 3. Construct Edges
    src_nodes, dst_nodes = [], []
    num_nodes = len(nodes_text)
    
    # Sequential / baseline connectivity
    for i in range(num_nodes):
        src_nodes.append(i)
        dst_nodes.append(i) # Self-loop
        if i + 1 < num_nodes:
            src_nodes.extend([i, i + 1])
            dst_nodes.extend([i + 1, i])

    # Call graph connectivity: check if func_name[j] appears inside node_text[i]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and func_names[j] in nodes_text[i]:
                src_nodes.append(i)
                dst_nodes.append(j)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


_GRAPH_CACHE: dict = {}


def _graphs(texts: List[str], labels: List[int], device: str) -> List[Data]:
    """Contract graphs for `texts`; MiniLM node embeddings are frozen, so the graphs are
    built once per process and reused by every fold, repeat and LOSO direction."""
    import hashlib
    key = hashlib.sha256("\x00".join(texts).encode("utf-8", "ignore")).hexdigest()
    if key not in _GRAPH_CACHE:
        print("  Building contract graphs with sentence-transformer embeddings...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        t0_graph = time.time()
        _GRAPH_CACHE[key] = [build_contract_graph(t, embedder, device=device) for t in texts]
        print(f"  Constructed {len(texts)} contract graphs in {time.time()-t0_graph:.1f}s")
    graphs = []
    for g, y in zip(_GRAPH_CACHE[key], labels):
        g = g.clone()
        g.y = torch.tensor(y, dtype=torch.long)
        graphs.append(g)
    return graphs


def run(
    texts: List[str],
    labels: List[int],
    source_datasets: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[GNN] Initializing GNN (Contract-GCN) on device: {device}")
    
    # 1. Feature Extraction / Graph Building (frozen encoder -> cached per process)
    graph_list = _graphs(texts, labels, device)

    fold_results = []

    print(f"[GNN] Running {len(splits)}-Fold Cross Validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        t_fold = time.time()
        
        # Prepare datasets
        train_graphs = [graph_list[i] for i in train_idx]
        test_graphs = [graph_list[i] for i in test_idx]
        
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        test_sources = [source_datasets[i] for i in test_idx]
        
        # Compute class weights
        cw = get_class_weights(train_labels)
        cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

        train_loader = PyGDataLoader(train_graphs, batch_size=16, shuffle=True)
        test_loader = PyGDataLoader(test_graphs, batch_size=32, shuffle=False)

        # Initialize model
        torch.manual_seed(get_seed() + fold_idx)
        model = ContractGCN(in_channels=384, hidden_channels=128, out_channels=NUM_LABELS, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)

        # Training Loop
        model.train()
        epochs = 35
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                batch_preds = out.argmax(dim=-1).cpu().numpy().tolist()
                preds.extend(batch_preds)

        train_time = time.time() - t_fold
        metrics = compute_metrics(test_labels, preds, np.array(test_sources))
        metrics["fold"] = fold_idx
        metrics["train_time_s"] = round(train_time, 2)
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        metrics["test_idx"] = [int(i) for i in test_idx]

        print(f"  Fold {fold_idx}: macro-F1 = {metrics['macro_f1']:.4f} | Acc = {metrics['accuracy']:.4f} | MCC = {metrics['mcc']:.4f} ({train_time:.1f}s)")
        fold_results.append(metrics)

    # Aggregation
    agg = aggregate_folds(fold_results)
    _ev.save_results("gnn", fold_results, agg)
    
    print(f"\n[GNN] Aggregated macro-F1: {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
    return {"model": "gnn", "folds": fold_results, "aggregate": agg}
