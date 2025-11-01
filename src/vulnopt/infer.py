import torch
from pathlib import Path
from vulnopt.models.encoders import CodeEncoder
from vulnopt.models.model import FusionClassifier
from vulnopt.features.ast_graph import extract_ast_graph_python
import json

def infer_main(ckpt: Path, path: Path, out: Path, _):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt, map_location=device)
    model = FusionClassifier()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    encoder = CodeEncoder()

    samples = []
    if path.is_file():
        texts = [path.read_text(encoding="utf-8")]
    else:
        texts = [p.read_text(encoding="utf-8") for p in path.rglob("*.py")]

    results = []
    for t in texts:
        tokens, adj = extract_ast_graph_python(t)
        # we will use full file -> break by functions is better (not implemented here)
        code_emb = encoder.encode([t]).to(device)
        node_repr = torch.zeros((1, 128), device=device)
        logits = model(code_emb.to(device), node_repr)
        pred = logits.argmax(dim=1).item()
        prob = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        results.append({"pred": int(pred), "prob": prob, "snippet": t[:300]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[infer] wrote {len(results)} entries to {out}")
