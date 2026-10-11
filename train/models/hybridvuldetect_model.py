"""
train/models/hybridvuldetect_model.py
--------------------------------------
HybridVulDetect model as per paper.
"""
from train.models.mlp_utils import run_fused_model_cv, HybridVulDetect

def run(texts, labels, source_datasets, splits):
    return run_fused_model_cv("hybridvuldetect", HybridVulDetect, texts, labels, source_datasets, splits)
