from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from vulnopt.config import ROOT, DATA_DIR
from vulnopt.train import (
    CodeDataset,
    build_model,
    deserialize_vectorizer,
    evaluate as evaluate_model,
    load_split,
    vectorize_samples,
)


def _resolve_dataset_dir(checkpoint_dataset: str | None, override: Path | None) -> Path:
    if override is not None:
        dataset_dir = override.expanduser()
        if not dataset_dir.is_absolute():
            dataset_dir = (ROOT / dataset_dir).resolve()
        return dataset_dir

    if checkpoint_dataset is None:
        dataset_dir = DATA_DIR / "synthetic_vuln"
    else:
        dataset_dir = Path(checkpoint_dataset)
        if not dataset_dir.is_absolute():
            dataset_dir = (ROOT / dataset_dir).resolve()
    return dataset_dir


def _prepare_datasets(
    dataset_dir: Path, vectorizer_blob: bytes
) -> Tuple[CodeDataset, CodeDataset]:
    vectorizer = deserialize_vectorizer(vectorizer_blob)
    train_samples = load_split(dataset_dir, "train")
    valid_samples = load_split(dataset_dir, "valid")

    train_features, train_labels = vectorize_samples(vectorizer, train_samples)
    valid_features, valid_labels = vectorize_samples(vectorizer, valid_samples)

    return CodeDataset(train_features, train_labels), CodeDataset(valid_features, valid_labels)


def eval_main(ckpt_path: Path, batch_size: int = 128, dataset_dir: Path | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    input_dim = int(checkpoint["input_dim"])
    class_names = list(checkpoint["class_names"])
    vectorizer_blob = checkpoint["vectorizer"]
    checkpoint_dataset = checkpoint.get("dataset_dir")

    resolved_dataset = _resolve_dataset_dir(checkpoint_dataset, dataset_dir)
    train_dataset, valid_dataset = _prepare_datasets(resolved_dataset, vectorizer_blob)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = build_model(input_dim=input_dim, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
    valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)

    print("Evaluation results:")
    print(f"  Train loss: {train_loss:.4f} | accuracy: {train_acc:.4f}")
    print(f"  Valid loss: {valid_loss:.4f} | accuracy: {valid_acc:.4f}")
    print(f"  Dataset dir: {resolved_dataset}")
