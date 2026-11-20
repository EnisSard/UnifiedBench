"""
train/models/deepsolid_model.py
-------------------------------
DeepSolid: dual-branch MLP (Tanh semantic branch, ReLU static branch) on the fused vector.
"""
from train.models.mlp_utils import run_fused_model_cv, DeepSolidHybrid

def run(texts, labels, source_datasets, splits, persist=True, hp=None):
    return run_fused_model_cv("deepsolid", DeepSolidHybrid, texts, labels, source_datasets, splits, persist=persist, hp=hp)
