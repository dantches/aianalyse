import torch
import json
from pathlib import Path
from vulnopt.models.model import FusionClassifier
from vulnopt.models.encoders import CodeEncoder
from torch.utils.data import DataLoader
from vulnopt.train import CodeVulnDataset, collate_fn
import numpy as np
from sklearn.metrics import classification_report

def load_checkpoint(ckpt_path: Path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    # instantiate model same as train
    code_encoder = CodeEncoder()
    model = FusionClassifier(code_emb_dim=768, node_feat_dim=128)
    model.load_state_dict(ckpt["model_state"])
    return code_encoder, model

def eval_main(ckpt: Path, data_jsonl: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_encoder, model = load_checkpoint(ckpt, device)
    ds = CodeVulnDataset(data_jsonl)
    dl = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    model.to(device).eval()
    for batch in dl:
        codes = batch["codes"]
        node_matrix = torch.tensor(batch["node_matrix"], dtype=torch.long).to(device)
        labels = batch["labels"].numpy().tolist()
        # encode
        inputs = ds.encoder.tokenizer(codes, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ce_out = ds.encoder.model(**inputs)
            if hasattr(ce_out, "pooler_output") and ce_out.pooler_output is not None:
                code_emb = ce_out.pooler_output
            else:
                code_emb = ce_out.last_hidden_state.mean(dim=1)
        # node embedding placeholder
        # We don't have node_emb_table state restored here in this simplified evaluator;
        node_repr = torch.zeros((code_emb.size(0), 128), device=device)
        logits = model(code_emb.to(device), node_repr)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
    print(classification_report(y_true, y_pred, digits=4))
