# Mamba-GIR

Mamba-GIR uses residual vector quantization (RVQ) to assign fixed-gallery identifiers and codebook-aware visual prompting (CAVP) to condition a Mamba decoder on a query image. The repository contains dataset preparation, tokenizer training, generator training, evaluation and CAVP ablations.

## Released 790M checkpoints

The release contains one MambaVision-B / Mamba-790M checkpoint per benchmark with its paired identifiers, gallery order, portable image manifest, saved raw beams and reference metrics. Download `mambagir-790m-metadata.tar.gz` and the `*_weights.pt` files for the datasets you want to evaluate. [GitHub release](https://github.com/dhujinyun-netizen/Mamba-GIR/releases/tag/repro-v1)

Extract the metadata archive and place the downloaded weights beside `inference.json` in `artifacts/`. The archive's `metrics/` directory records the measurements from these checkpoints. Each dataset is represented by a single released checkpoint. The three-seed training commands below produce new independent results.

| Dataset | Queries | Gallery | R@1 (%) | R@4 (%) | R@8 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-Shop | 14,218 | 12,612 | 85.62 | 86.68 | 88.18 |
| CUB | 5,794 | 5,994 | 89.64 | 94.27 | 94.43 |
| Cars196 | 8,041 | 8,144 | 92.02 | 95.83 | 95.90 |
| FGVC-Aircraft | 3,333 | 6,667 | 89.38 | 95.35 | 95.44 |
| Four-dataset mean | | | 89.17 | 93.03 | 93.49 |

### Recompute the measurements from the saved beams

This step uses CPU PyTorch, NumPy and Pillow. It recomputes image recall, raw-key Hit@K and MRR, identifier validity, reuse and collision statistics directly from the released raw beams.

```bash
tar -xzf mambagir-790m-metadata.tar.gz
for dataset in inshop cub cars aircraft; do
  python -m mambagir.infer "$dataset" --artifacts artifacts --replay \
    --output-dir "runs/replay/$dataset"
done
```

### Run inference on the benchmark queries

Install a CUDA-compatible build of PyTorch and torchvision first. The release was checked with Python 3.10, PyTorch 2.1.0 with CUDA 11.8, Transformers 4.53.0, timm 1.0.12, mamba-ssm 2.2.4 and causal-conv1d 1.5.0.post8. The released settings use CUDA graphs, BF16 for In-Shop, CUB and Aircraft, and FP32 with TF32 enabled for Cars.

```bash
python -m pip install -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'state-spaces/mamba-790m-hf',
    revision='9822dd4b76af2bd9099b6ce2f19efd8329189a7e',
    allow_patterns=['config.json', 'model.safetensors'],
    local_dir='weights/mamba-790m',
)
PY

python -m mambagir.infer cub --artifacts artifacts \
  --data-root /datasets/CUB_200_2011 --mamba-model weights/mamba-790m \
  --output-dir runs/released/cub
```

Point `--data-root` at the directory from which the paths in `artifacts/manifests/<dataset>.json` are relative. For example, the CUB root contains `images/`, Cars contains `images/cars_train/` and `images/cars_test/`, Aircraft contains `images/` and In-Shop contains `images/img/`. Each model loads its paired `<dataset>_identifiers.pt` and release manifest. The In-Shop inference manifest indexes the official gallery directly. Use `--image FILE` with the same model and dataset arguments to retrieve for custom queries. Benchmark inference writes `metrics.json` and `predictions.pt`.

## Train and evaluate

Use the original dataset images and official metadata. The processed training, gallery and query splits are in [manifests/](manifests/README.md). The released mappings are in `artifacts/mappings/`. Training exports its new identifiers and their ordered gallery members alongside each run.

| Dataset | Tokenizer and generator target pool | Searchable gallery | Evaluation queries |
| --- | ---: | ---: | ---: |
| In-Shop | 25,882 train + 12,612 gallery | 12,612 gallery | 14,218 query |
| CUB | 5,994 train | 5,994 train | 5,794 test |
| Cars196 | 8,144 train | 8,144 train | 8,041 test |
| FGVC-Aircraft | 6,667 train + validation | 6,667 train + validation | 3,333 test |

The In-Shop gallery labels are used for tokenizer supervision, same-identity pairing and the auxiliary class loss. The held-out query split is used during evaluation. `configs/paper.json` specifies the 200-epoch tokenizer, 400-epoch generator, four RVQ stages, 256 codewords per stage and the default conditioning and decoding settings. Supply the official MambaVision-B-21K `mambavision_base_21k.pth.tar` and the local pretrained Mamba directory. [MambaVision-B-21K checkpoint](https://huggingface.co/nvidia/MambaVision-B-21K/resolve/bfe552a588f9f250ea0583951ea4dd5a10a198f8/mambavision_base_21k.pth.tar)

```bash
export CUB_ROOT=/datasets/CUB_200_2011 CARS_ROOT=/datasets/cars196
export AIRCRAFT_ROOT=/datasets/fgvc-aircraft-2013b INSHOP_ROOT=/datasets/InShop
export VISION_CHECKPOINT=/weights/mambavision_base_21k.pth.tar
export MAMBA_MODEL="$PWD/weights/mamba-790m"
export RUN_ROOT=runs/reproduction-main
bash scripts/reproduce.sh plan
bash scripts/reproduce.sh run
```

The first command prints the plan. The second prepares all four splits, trains tokenizer and generator for seeds 3407, 3408 and 3409, evaluates the final configured epoch and aggregates means and sample standard deviations. For individual stages and overrides, use `python -m mambagir --help`. Runs record their configuration and parameter counts. Evaluation records per-run recall, raw-key and collision statistics and the raw beam outputs. To repeat a different decoder scale, use a matching pretrained model directory and a distinct `RUN_ROOT`.

### CAVP ablations

Use the same tokenizer and image roles per dataset and seed. The ablation script generates explicit configurations for the ten conditioning controls, trains each variant, evaluates the four datasets and summarizes per-dataset and equal-weight macro recall.

```bash
python scripts/reproduce_ablations.py --run-root "$RUN_ROOT" \
  --mamba-model "$MAMBA_MODEL" --output runs/ablation-790m
```

Add `--execute` to run the plan. Use `--variants patch_attention levelwise full_cavp` to run the two core comparisons. Generic patch attention uses a common head and alignment summary. Level-wise prompting selects the matching stage summary while keeping the same prediction memory. Full CAVP adds prompt-to-codebook attention. The no-code-memory control removes that attention and uses decoder embeddings for prediction and alignment.

### Additional evaluation commands

- `python -m mambagir.reproduction --help` exports portable splits and paired key mappings from a trained tokenizer.
- `python -m mambagir.diagnostics --help` computes raw-key and ranking diagnostics from saved training-workflow predictions.
- `python -m mambagir.benchmark --help` profiles latency and peak memory for a training-workflow checkpoint.
- `python -m unittest discover -s tests -v` runs the implementation checks.

The release and the training recipes specify their own measurements. Exact reproduction of earlier reported table entries requires the corresponding per-run configurations, checkpoints or raw predictions. The reference Transformer implementation is available through `decoder.family=transformer`. Refer to the original IRGen and Diff-IR publications for their results and protocols. [IRGen code](https://github.com/yakt00/IRGen)

## Licenses

Original Mamba-GIR code is MIT licensed. The extracted MambaVision architecture retains the NVIDIA Source Code License-NC. The released visual encoder states are derived from the NVIDIA MambaVision-B-21K pretrained weights under their upstream CC BY-NC-SA 4.0 terms. See [NOTICE](NOTICE) for attribution. Dataset images are obtained from their original sources. [MambaVision licensing](https://github.com/NVlabs/MambaVision#licenses)
