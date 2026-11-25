# Mamba-GIR

Research implementation of **Mamba-GIR: Structured Visual Conditioning and Identifier Modeling for Generative Image Retrieval**.

**Official splits → semantic tokenizer → residual vector quantization (RVQ) → fixed identifiers → CAVP-conditioned Mamba training → gallery retrieval.**

The repository provides source code, dataset split manifests, identifier-to-image mappings, configurations, and commands for training and evaluation.

## Environment

Checked with Python 3.10, PyTorch 2.1.0 + CUDA 11.8, Transformers 4.53.0 and timm 1.0.12. Install CUDA-compatible PyTorch first:

~~~bash
python -m pip install -r requirements.txt
python -m mambagir --help
~~~

The MambaVision selective-scan dependency needs a compatible CUDA environment. RVQ and LoRA modules are implemented in PyTorch. Main defaults are in configs/paper.json.

Supply a checkpoint matching the selected MambaVision architecture and a local Hugging Face MambaModel directory. The default encoder is MambaVision-B; see the [upstream MambaVision project](https://github.com/NVlabs/MambaVision). The 790M designation identifies the Mamba decoder variant. Full-model parameter counts include the visual encoder, decoder, conditioning modules, and prediction heads. Each run records its configuration and actual total/trainable parameter counts.

## 1. Prepare official splits

Processed split manifests for the four benchmarks are in [manifests/](manifests/README.md). Each file specifies image paths, labels, training targets, gallery row indices, and queries. The following commands generate manifests for your local dataset layout:

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
  --vision-checkpoint /weights/mambavision_base_21k.pth.tar \
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

Generator artifacts bind to the tokenizer used during training. last.pt is the final scheduled checkpoint; best.pt is chosen by training loss. Training applies LoRA through the adapter modules.

## 4. Evaluate

~~~bash
python -m mambagir evaluate \
  --checkpoint runs/cub/seed3407/generator/last.pt \
  --manifest data/cub.json \
  --output runs/cub/seed3407/evaluation \
  --device cuda:0 --beam-size 8
~~~

Evaluation restores the adapters and merges them into the decoder weights for inference. Mamba processes the visual prefix and SOS once, then carries convolution and SSM states while consuming one new token per step. Beam selection reorders and duplicates both state types with their parent hypotheses. The Python API supports `beam_search(..., use_cache=False)` for comparison with full-prefix decoding.

Add `--cuda-graph` to CUDA evaluation to replay the complete encoding and beam-search computation with fixed tensor shapes. The runner precomputes the fixed codebook memory and its attention K/V projections once. Each query still computes its own visual features, prompts and decoder states. A smaller final evaluation batch uses the cached eager path. The Python `CUDAGraphBeamSearch(model, example_images, beam_size=8, amp=False)` interface provides the same reusable runner after `model.prepare_for_inference()`; create a new runner after changing model weights, precision or input shape.

Unconstrained beam search produces raw keys. Gallery resolution removes invalid keys, then expands every distinct valid key in fixed gallery-manifest order without image reranking. Outputs: metrics.json and predictions.pt.

The report includes image Recall@K, raw-key Hit@K/MRR, validity, query reuse, gallery collisions, and a within-bucket ordering diagnostic. Raw-key Hit@K/MRR retain invalid candidates at their original beam positions. A key is relevant when at least one of its gallery members has the query label; MRR assigns zero when no relevant key occurs in the beam.

Export aggregate raw-ranking metrics and the candidate-level records from a saved evaluation:

~~~bash
python -m mambagir.diagnostics \
  --predictions runs/cub/seed3407/evaluation/predictions.pt \
  --tokenizer runs/cub/seed3407/tokenizer/tokenizer.pt \
  --manifest data/cub.json \
  --output-dir runs/cub/seed3407/evaluation/diagnostics
~~~

Measure query latency and peak GPU memory for a trained configuration:

~~~bash
python -m mambagir.benchmark \
  --checkpoint runs/cub/seed3407/generator/last.pt \
  --manifest data/cub.json \
  --output runs/cub/seed3407/timing.json \
  --device cuda:0 --batch-size 1 --beam-size 8 \
  --warmup 20 --iterations 100
~~~

Timing covers a resident query batch through visual encoding, conditioning, beam decoding, and gallery resolution. The JSON report records the GPU, precision, warm-up and measured iterations, CUDA-synchronized latency samples, and peak allocated/reserved GPU memory. Image preprocessing and input transfer precede the timed interval.

The benchmark also accepts `--cuda-graph`. Graph preparation occurs once before timed queries and is reported separately as `cuda_graph_setup_ms`; query latency includes input-buffer copying, graph replay, and gallery resolution. Both evaluation and timing retain the checkpoint configuration's autocast setting.

## 5. Export a retrieval index

Released gallery mappings for all four benchmarks are available in [identifiers/](identifiers/README.md), paired with the corresponding [split manifests](manifests/README.md). Each mapping contains the four-symbol identifier and every gallery image assigned to it, including the image path, label, target row, and gallery row. Members of a shared identifier retain their gallery order. Use each assignment with the generator trained on those identifiers.

Export the gallery lookup and dataset manifest for a trained tokenizer with:

~~~bash
python -m mambagir.reproduction export \
  --manifest data/cub.json \
  --tokenizer runs/cub/seed3407/tokenizer/tokenizer.pt \
  --dataset-root /datasets/CUB_200_2011 \
  --output runs/cub/seed3407/export
~~~

The export contains relative image paths, ordered gallery members for each identifier, and the tokenizer configuration. The reproduction script provides commands for all four datasets:

~~~bash
python -m mambagir.reproduction --help
CUB_ROOT=/datasets/CUB_200_2011 CARS_ROOT=/datasets/cars196 \
AIRCRAFT_ROOT=/datasets/fgvc-aircraft-2013b INSHOP_ROOT=/datasets/InShop \
VISION_CHECKPOINT=/weights/mambavision_base_21k.pth.tar \
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
| mambagir/diagnostics.py | Saved raw candidates and ranking diagnostics |
| mambagir/benchmark.py | Query latency and GPU memory measurements |

Generic patch controls use a shared prompt mean in the prediction/alignment paths; level-wise controls use the matching stage mean. Full CAVP remains stage-aware. The no_code_memory control removes both prompt code attention and RVQ vectors from head/alignment memory.

## Verification

~~~bash
PYTHONDONTWRITEBYTECODE=1 USE_TF=0 MKL_THREADING_LAYER=GNU \
  OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
~~~

Tests cover formulas, active LoRA gradients, causality, data roles, tokenizer identity, beam ancestry, invalid raw positions, collision ordering and a small training-to-retrieval path. Full pretrained integration is opt-in via MAMBAGIR_ENCODER_CHECKPOINT and MAMBAGIR_DECODER_PATH.

## License

Original Mamba-GIR code is MIT licensed. The extracted MambaVision architecture retains NVIDIA's Source Code License-NC and its research/evaluation-only use limitation; see NOTICE and licenses/NVIDIA-MambaVision.txt. Datasets, weights and external dependencies retain their respective terms.
