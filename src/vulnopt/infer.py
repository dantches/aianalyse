import json
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import torch

from vulnopt.train import build_model, deserialize_vectorizer


def _normalize_payload(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        return [payload]
    if isinstance(payload, list):
        normalized: List[Mapping[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                normalized.append(item)
            else:
                normalized.append({"code": str(item)})
        return normalized
    raise ValueError("Input JSON must be a mapping or a list of mappings/strings")


def _extract_codes(samples: Sequence[Mapping[str, Any]]) -> List[str]:
    codes: List[str] = []
    for sample in samples:
        if "code" not in sample:
            raise ValueError("Each sample must contain a 'code' field")
        codes.append(str(sample["code"]))
    return codes


def infer_main(ckpt_path: Path, samples_path: Path, out_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_names = list(checkpoint["class_names"])
    input_dim = int(checkpoint["input_dim"])
    vectorizer = deserialize_vectorizer(checkpoint["vectorizer"])

    model = build_model(input_dim=input_dim, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with samples_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = _normalize_payload(payload)
    codes = _extract_codes(samples)

    features = vectorizer.transform(codes).toarray().astype(np.float32)
    batch = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    results = []
    for idx, sample in enumerate(samples):
        results.append(
            {
                "input": sample,
                "predicted_class": class_names[int(preds[idx])],
                "probabilities": {
                    cls: float(prob) for cls, prob in zip(class_names, probs[idx])
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference completed for {len(results)} sample(s). Results saved to {out_path}")
