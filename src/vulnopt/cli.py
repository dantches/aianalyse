from pathlib import Path

import typer

from vulnopt import train, evaluate, infer

app = typer.Typer(
    help="VulnOpt — пример проекта для классификации уязвимостей в коде"
)


@app.command()
def train_cmd(
    out: Path = Path("runs/vuln_detector"),
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    dataset: Path | None = typer.Option(
        None, help="Путь к каталогу с JSONL-файлами train/valid/test"
    ),
    hidden_dim: int = 256,
    max_features: int = 4096,
):
    """Запустить обучение классификатора уязвимых фрагментов кода."""

    train.train_main(
        out_dir=out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        dataset_dir=dataset,
        hidden_dim=hidden_dim,
        max_features=max_features,
    )


@app.command()
def evaluate_cmd(
    ckpt: Path,
    batch_size: int = 128,
    dataset: Path | None = typer.Option(None, help="Путь к датасету для оценки"),
):
    """Оценить качество сохранённой модели."""

    evaluate.eval_main(ckpt, batch_size=batch_size, dataset_dir=dataset)


@app.command()
def infer_cmd(ckpt: Path, samples: Path, out: Path):
    """Выполнить инференс по JSON с кодовыми фрагментами."""

    infer.infer_main(ckpt, samples, out)


if __name__ == "__main__":
    app()
