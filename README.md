# Mamba-GIR

Minimal research implementation of **Mamba-GIR: Structured Visual Conditioning and Identifier Modeling for Generative Image Retrieval**.

**Official splits → semantic tokenizer → residual vector quantization (RVQ) → fixed identifiers → CAVP-conditioned Mamba training → gallery retrieval.**

This release contains source code, configuration and verification only. Dataset images, pretrained weights, trained checkpoints, run artifacts, manuscripts and review correspondence are not included.

## Environment

Checked with Python 3.10, PyTorch 2.1.0 + CUDA 11.8, Transformers 4.53.0 and timm 1.0.12. Install CUDA-compatible PyTorch first:

~~~bash
python -m pip install -r requirements.txt
python -m mambagir --help
~~~

The MambaVision selective-scan dependency needs a compatible CUDA environment. RVQ uses PyTorch; FAISS and PEFT are not required. Main defaults are in configs/paper.json.

Obtain pretrained weights separately. Supply a checkpoint matching the default MambaVision L-21K architecture and a local Hugging Face MambaModel directory. See the [upstream MambaVision project](https://github.com/NVlabs/MambaVision). The decoder checkpoint name is not the total retrieval-model parameter count; the code reports actual total and trainable parameters. With the locally verified L-21K encoder and mamba-790m decoder, the CUB 200-class model has 1,064,165,948 total and 44,569,676 trainable parameters. That example is an executable main-method reference, not an archived instance of the reported 790M full-model configuration.

## 1. Prepare official splits

~~~bash
python -m mambagir prepare-data --dataset cub --root /datasets/CUB_200_2011 --output data/cub.json
python -m mambagir prepare-data --dataset cars --root /datasets/cars196 --output data/cars.json
python -m mambagir prepare-data --dataset aircraft --root /datasets/fgvc-aircraft-2013b --output data/aircraft.json
python -m mambagir prepare-data --dataset inshop --root /datasets/InShop --output data/inshop.json
~~~

| Dataset | Tokenizer/generator target pool | Searchable gallery | Held-out queries |
|---|---:|---:|---:|
| CUB | 5,994 train | 5,994 train | 5,794 test |
| Cars | 8,144 train | 8,144 train | 8,041 test |
| Aircraft | 6,667 train + validation | 6,667 train + validation | 3,333 test |
| In-Shop | 25,882 train + 12,612 gallery | 12,612 gallery | 14,218 query |

In-Shop gallery identity labels participate in tokenizer semantic supervision, positive-pair sampling and the generator's auxiliary class loss. Query images are held out. Images are resized to 256 × 256, then randomly cropped/flipped in training or center-cropped to 224 × 224 during evaluation, using ImageNet normalization.

## 2. Train the tokenizer and construct identifiers

~~~bash
python -m mambagir train-tokenizer \
  --manifest data/cub.json \
  --vision-checkpoint /weights/mambavision_large_21k.pth.tar \
  --output runs/cub/seed3407/tokenizer \
  --seed 3407 --device cuda:0
~~~

This trains the semantic encoder/head and alternates deterministic feature extraction with residual codebook fitting. The exported tokenizer.pt contains the final encoder, frozen codebooks, target identifiers, labels, gallery indices and a unique tokenizer identity. Defaults: 4 RVQ stages, 256 codewords per stage. Supply trusted pretrained files: legacy NVIDIA training metadata requires the original PyTorch loader.

## 3. Train the generator

~~~bash
python -m mambagir train \
  --manifest data/cub.json \
  --tokenizer runs/cub/seed3407/tokenizer/tokenizer.pt \
  --mamba-model /weights/mamba-790m \
  --output runs/cub/seed3407/generator \
  --seed 3407 --device cuda:0
~~~

The frozen encoder supplies 49 patches. CAVP produces 4 prompts per RVQ stage. Codebook memories enter both prompt refinement and the stage-aware head. Trainable components are CAVP, the head, the auxiliary classifier and rank-8 LoRA adapters in x_proj, in_proj, out_proj and embeddings. Same-label targets are sampled independently for every source per epoch, excluding self when another positive exists.

Generator artifacts bind to the exact tokenizer UUID. last.pt is the final scheduled checkpoint; best.pt is chosen by training loss without query-set selection. Adapter-enabled Mamba uses an unfused forward path to preserve LoRA; decoding currently recomputes the prefix without incremental caching.

## 4. Evaluate

~~~bash
python -m mambagir evaluate \
  --checkpoint runs/cub/seed3407/generator/last.pt \
  --manifest data/cub.json \
  --output runs/cub/seed3407/evaluation \
  --device cuda:0 --beam-size 8
~~~

Unconstrained beam search produces raw keys. Gallery resolution removes invalid keys, then expands every distinct valid key in fixed gallery-manifest order without image reranking. Outputs: metrics.json and predictions.pt.

Image Recall@K, raw-key Hit@K/MRR, validity, query reuse, gallery collisions and a within-bucket oracle are separate metrics. The oracle is an upper-bound ordering diagnostic, not a trained reranker.

## 5. Export manifests and identifier mappings

~~~bash
python -m mambagir.reproduction export \
  --manifest data/cub.json \
  --tokenizer runs/cub/seed3407/tokenizer/tokenizer.pt \
  --dataset-root /datasets/CUB_200_2011 \
  --output runs/cub/seed3407/export
~~~

The export contains a relative manifest, every identifier's ordered gallery members, and configuration/tokenizer provenance; no images or weights. For relocation, see:

~~~bash
python -m mambagir.reproduction --help
CUB_ROOT=/datasets/CUB_200_2011 CARS_ROOT=/datasets/cars196 \
AIRCRAFT_ROOT=/datasets/fgvc-aircraft-2013b INSHOP_ROOT=/datasets/InShop \
VISION_CHECKPOINT=/weights/mambavision_large_21k.pth.tar \
MAMBA_MODEL=/weights/mamba-790m bash scripts/reproduce.sh plan
~~~

The script provides a four-dataset, three-seed command plan. Its default prints commands before optional execution.

## Main files

| File | Purpose |
|---|---|
| mambagir/data.py | Splits, preprocessing and positive targets |
| mambagir/tokenizer.py, rvq.py | Semantic objective and residual quantization |
| mambagir/vision.py, _mambavision.py | Global/spatial encoder features |
| mambagir/model.py | CAVP, stage-aware head and losses |
| mambagir/decoder.py | Mamba LoRA and causal Transformer reference |
| mambagir/retrieval.py | Beam search, gallery resolution and metrics |
| mambagir/runtime.py, __main__.py | Training, evaluation and artifacts |
| mambagir/reproduction.py | Portable manifests and key mappings |

Generic patch controls use a shared prompt mean in the prediction/alignment paths; level-wise controls use the matching stage mean. Full CAVP remains stage-aware. The no_code_memory control removes both prompt code attention and RVQ vectors from head/alignment memory. These explicit reference definitions do not constitute independently rerun historical ablations.

## Verification and result provenance

~~~bash
PYTHONDONTWRITEBYTECODE=1 USE_TF=0 MKL_THREADING_LAYER=GNU \
  OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
~~~

Tests cover formulas, active LoRA gradients, causality, data roles, tokenizer identity, beam ancestry, invalid raw positions, collision ordering and a small training-to-retrieval path. Full pretrained integration is opt-in via MAMBAGIR_ENCODER_CHECKPOINT and MAMBAGIR_DECODER_PATH.

The minimal release supplies the main-method reproduction path. Published benchmark values remain the authors' reported results; unit tests and synthetic smoke runs do not regenerate those accuracy values. External baseline implementations and archived experiment artifacts are not bundled.

## License

Original Mamba-GIR code is MIT licensed. The extracted MambaVision architecture retains NVIDIA's Source Code License-NC and its research/evaluation-only use limitation; see NOTICE and licenses/NVIDIA-MambaVision.txt. Datasets, weights and external dependencies retain their respective terms.
