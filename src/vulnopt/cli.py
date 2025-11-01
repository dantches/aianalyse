import typer
from pathlib import Path
from vulnopt.data import cve_map, repo_miner, denoise, nvd_fetch, prepare_dataset
from vulnopt import train, evaluate, infer

app = typer.Typer(help="VulnOpt — нейросеть для анализа и оптимизации кода")

@app.command()
def fetch_nvd(
    years: str,
    out: Path,
    api_key: str = typer.Option(None, "--api-key", "-k", help="NVD API key"),
):
    start, end = [int(x) for x in years.split("-")]
    out.parent.mkdir(parents=True, exist_ok=True)
    nvd_fetch.fetch_nvd(start, end, out, api_key=api_key)

@app.command()
def map_cve(nvd: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    cve_map.map_cve_references(nvd, out)

@app.command()
def mine_repos(map: Path, out: Path, cache: Path = Path("repos/")):
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    repo_miner.mine_repos(map, out, cache)

@app.command()
def prepare(data_json: Path, out_samples: Path):
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    prepare_dataset.build_samples(data_json, out_samples)

@app.command()
def denoise_cmd(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    denoise.denoise_jsonl(inp, out)

@app.command()
def train_cmd(data: Path, out: Path, epochs: int = 3, lr: float = 2e-5):
    out.mkdir(parents=True, exist_ok=True)
    train.train_main(data, out, epochs=epochs, lr=lr)

@app.command()
def evaluate_cmd(ckpt: Path, data: Path):
    evaluate.eval_main(ckpt, data)

@app.command()
def infer_cmd(ckpt: Path, path: Path, out: Path):
    infer.infer_main(ckpt, path, out, None)

if __name__ == "__main__":
    app()
