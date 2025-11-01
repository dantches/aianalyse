from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
NVD_API_KEY = os.getenv("NVD_API_KEY", None)
HF_MODEL = os.getenv("HF_MODEL", "microsoft/codebert-base")  # default
