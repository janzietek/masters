from __future__ import annotations
import os
import sys
import yaml
from typing import List
from .hparams import DEFAULT_HPARAMS

TEMPLATE_NOTE = (
    "# Szablon hiperparametrów per-dataset.\n"
    "# Po kalibracji cross-dataset zaktualizuj poniższe wartości i zamroź.\n"
)

def write_hparams(dataset_name: str, out_dir: str = "hparams"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_name}.yaml")
    if os.path.exists(path):
        print(f"⚠️  Pomijam – plik już istnieje: {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE_NOTE)
        yaml.safe_dump(DEFAULT_HPARAMS, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Utworzono: {path}")

def main():
    if len(sys.argv) < 2:
        print("Użycie: python -m expfs.make_hparams <dataset1> [<dataset2> ...]")
        sys.exit(1)
    for ds in sys.argv[1:]:
        write_hparams(ds)

if __name__ == "__main__":
    main()
