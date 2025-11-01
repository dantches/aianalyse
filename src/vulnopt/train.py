import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import random
from tqdm import tqdm
from vulnopt.models.encoders import CodeEncoder
from vulnopt.models.model import FusionClassifier
from vulnopt.features.ast_graph import extract_ast_graph_python
import numpy as np

class CodeVulnDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_model="microsoft/codebert-base", max_len=256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.samples.append(j)
        self.encoder = CodeEncoder(tokenizer_model)
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        code = s.get("code", "")
        # encode code with CodeEncoder later in collate for batch efficiency
        tokens, adj = extract_ast_graph_python(code)
        # produce node_feat as simple bag-of-node-type embedding via hashing
        # here: convert node type strings to indices via hash
        node_feats = np.array([hash(t) % 1000 for t in tokens], dtype=np.int64)
        label = int(s.get("label", 0))
        return {"code": code, "node_feats": node_feats, "label": label}

def collate_fn(batch):
    codes = [b["code"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    # create node_repr simple: mean hash embedding -> embedding matrix learned later
    max_nodes = max(len(b["node_feats"]) for b in batch)
    node_matrix = np.zeros((len(batch), max_nodes), dtype=np.int64)
    for i, b in enumerate(batch):
        nf = b["node_feats"]
        node_matrix[i, :len(nf)] = nf
    return {"codes": codes, "node_matrix": node_matrix, "labels": labels}

def train_main(data_jsonl: Path, out_dir: Path, epochs: int = 3, lr: float = 2e-5, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CodeVulnDataset(data_jsonl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    code_encoder = CodeEncoder().model.to(device)
    # node embedding table (for hashed node types)
    node_emb_table = nn.Embedding(1000, 128).to(device)
    model = FusionClassifier(code_emb_dim=code_encoder.config.hidden_size if hasattr(code_encoder, "config") else 768,
                              node_feat_dim=128).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(node_emb_table.parameters()) , lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            codes = batch["codes"]
            node_matrix = torch.tensor(batch["node_matrix"], dtype=torch.long).to(device)
            labels = batch["labels"].to(device)
            # encode codes in-batch
            inputs = ds.encoder.tokenizer(codes, truncation=True, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                ce_out = ds.encoder.model(**inputs)
                if hasattr(ce_out, "pooler_output") and ce_out.pooler_output is not None:
                    code_emb = ce_out.pooler_output
                else:
                    code_emb = ce_out.last_hidden_state.mean(dim=1)
            # node_repr: average embedding per row (ignoring zeros)
            node_feats = node_emb_table(node_matrix)  # (B, max_nodes, D)
            # mask zeros
            mask = (node_matrix != 0).unsqueeze(-1).to(device)
            node_feats = node_feats * mask
            node_sum = node_feats.sum(dim=1)
            node_count = mask.sum(dim=1).clamp(min=1)
            node_repr = node_sum / node_count

            logits = model(code_emb, node_repr)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            pbar.set_postfix(loss=loss.item(), acc=acc)
        # save checkpoint
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "node_emb": node_emb_table.state_dict()
        }, out_dir / f"ckpt_epoch{epoch+1}.pth")
    print("[train] done")
