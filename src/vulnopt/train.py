import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vulnopt.config import ROOT, DATA_DIR


@dataclass
class CodeSample:
    """A single labeled source snippet."""

    code: str
    label: int
    vulnerability_type: str


class CodeDataset(Dataset):
    """Dataset wrapper for dense feature vectors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have the same number of rows")
        self._features = features.astype(np.float32)
        self._labels = labels.astype(np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self._features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        feature = torch.from_numpy(self._features[index]).float()
        label = torch.tensor(int(self._labels[index]), dtype=torch.long)
        return feature, label


def load_split(dataset_dir: Path, split: str) -> List[CodeSample]:
    """Load a JSONL split into :class:`CodeSample` objects."""

    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split '{split}' not found at {path}")

    samples: List[CodeSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            samples.append(
                CodeSample(
                    code=str(payload["code"]),
                    label=int(payload["label"]),
                    vulnerability_type=str(payload.get("vulnerability_type", "unknown")),
                )
            )
    if not samples:
        raise ValueError(f"Split '{split}' is empty at {path}")
    return samples


def serialize_vectorizer(vectorizer: TfidfVectorizer) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(vectorizer, buffer)
    return buffer.getvalue()


def deserialize_vectorizer(blob: bytes) -> TfidfVectorizer:
    buffer = io.BytesIO(blob)
    return joblib.load(buffer)


def vectorize_samples(
    vectorizer: TfidfVectorizer, samples: Sequence[CodeSample]
) -> Tuple[np.ndarray, np.ndarray]:
    texts = [sample.code for sample in samples]
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    features = vectorizer.transform(texts)
    dense = features.toarray().astype(np.float32)
    return dense, labels


def build_model(input_dim: int, hidden_dim: int = 256, num_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, num_classes),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += features.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def _resolve_dataset_dir(dataset_dir: Path | None) -> Path:
    if dataset_dir is None:
        dataset_dir = DATA_DIR / "synthetic_vuln"
    dataset_dir = dataset_dir.expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    return dataset_dir


def _prepare_vectorizer(train_samples: Sequence[CodeSample], max_features: int) -> TfidfVectorizer:
    texts = [sample.code for sample in train_samples]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        lowercase=False,
        max_features=max_features,
    )
    vectorizer.fit(texts)
    return vectorizer


def train_main(
    out_dir: Path,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    dataset_dir: Path | None = None,
    hidden_dim: int = 256,
    max_features: int = 4096,
) -> None:
    """Train a text classifier that flags vulnerable C snippets."""

    dataset_dir = _resolve_dataset_dir(dataset_dir)
    train_samples = load_split(dataset_dir, "train")
    valid_samples = load_split(dataset_dir, "valid")

    vectorizer = _prepare_vectorizer(train_samples, max_features=max_features)

    train_features, train_labels = vectorize_samples(vectorizer, train_samples)
    valid_features, valid_labels = vectorize_samples(vectorizer, valid_samples)

    train_dataset = CodeDataset(train_features, train_labels)
    valid_dataset = CodeDataset(valid_features, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_dim=train_features.shape[1],
        hidden_dim=hidden_dim,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
        )
        tqdm.write(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

    out_dir = out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "vuln_classifier.pt"
    serialized_vectorizer = serialize_vectorizer(vectorizer)

    dataset_dir_resolved = dataset_dir.resolve()
    try:
        dataset_dir_field = dataset_dir_resolved.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_dir_field = str(dataset_dir_resolved)

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": train_features.shape[1],
            "class_names": ["safe", "vulnerable"],
            "vectorizer": serialized_vectorizer,
            "dataset_dir": dataset_dir_field,
            "max_features": max_features,
            "hidden_dim": hidden_dim,
        },
        checkpoint_path,
    )

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    tqdm.write(f"Training finished. Checkpoint saved to {checkpoint_path}")
