"""Run the comparisons on a Modal GPU, one container per model, in parallel.

    modal run modal_run.py                          # chain task, all five models, seed 1337
    modal run modal_run.py --task shakespeare --steps 3000
    modal run modal_run.py --seed 1 --tag seed1     # a second seed
    MODAL_GPU=A10G modal run modal_run.py ...       # a cheaper GPU (these models are small)

Results land in out/ locally (merged compare_<task>.json and the checkpoints), as if
`python nanoloop_moe.py --mode compare` had run here. Needs a Modal account (`pip install modal`).
"""
import json
import os
import subprocess

import modal

app = modal.App("nanoloop-moe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file("nanoloop_moe.py", "/root/work/nanoloop_moe.py")
)
NAMES = ["standard", "plainloop", "loopmoe_k1", "loopmoe_nomix", "loopmoe"]


@app.function(image=image, gpu=os.environ.get("MODAL_GPU", "H100"), timeout=4 * 3600)
def train_one(name: str, flags: list[str]) -> dict:
    os.chdir("/root/work")
    subprocess.run(["python", "nanoloop_moe.py", "--mode", "compare", "--only", name, *flags], check=True)
    task = flags[flags.index("--task") + 1]
    row = json.load(open(f"out/compare_{task}.json"))[name]
    return {"name": name, "row": row, "ckpt": open(f"out/{task}_{name}.pt", "rb").read()}


@app.local_entrypoint()
def main(task: str = "chain", steps: int = 0, seed: int = 1337, tag: str = "", only: str = "", extra: str = ""):
    flags = ["--task", task, "--seed", str(seed)] + (["--steps", str(steps)] if steps else []) + extra.split()
    names = [n for n in NAMES if not only or n in only.split(",")]
    suffix = f"_{tag}" if tag else ""
    os.makedirs("out", exist_ok=True)
    path = f"out/compare_{task}{suffix}.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    for r in train_one.starmap([(n, flags) for n in names]):
        results[r["name"]] = r["row"]
        open(f"out/{task}_{r['name']}{suffix}.pt", "wb").write(r["ckpt"])
        print(r["name"], {k: v for k, v in r["row"].items() if k != "sweep"})
        json.dump(results, open(path, "w"), indent=1)
