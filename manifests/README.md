# Fixed-gallery split manifests

These manifests are constructed from the official benchmark metadata. `targets` contains the images used for tokenizer and generator training; `gallery_indices` selects the searchable images from that ordered target list; `queries` contains the evaluation images. Labels are assigned from the target pool.

| Manifest | Training targets | Gallery images | Queries | Paths relative to the dataset root |
|---|---:|---:|---:|---|
| `cub.json` | 5,994 | 5,994 | 5,794 | `images/<class>/<image>.jpg` |
| `cars.json` | 8,144 | 8,144 | 8,041 | `images/cars_train/<image>.jpg`, `images/cars_test/<image>.jpg` |
| `aircraft.json` | 6,667 | 6,667 | 3,333 | `images/<image>.jpg` |
| `inshop.json` | 38,494 | 12,612 | 14,218 | `img/<category>/<item>/<image>.jpg` |

For In-Shop, the targets comprise 25,882 training images followed by 12,612 gallery images. The 14,218 query images are held out. The other datasets use their training pool as the gallery; Aircraft combines training and validation images in that pool.

Place the chosen manifest at the root matching the path layout above, then pass that file to the training and evaluation commands. For example:

```bash
cp manifests/cub.json /datasets/CUB_200_2011/retrieval_manifest.json
python -m mambagir train-tokenizer \
  --manifest /datasets/CUB_200_2011/retrieval_manifest.json \
  --vision-checkpoint /weights/mambavision_base_21k.pth.tar \
  --output runs/cub/seed3407/tokenizer \
  --seed 3407 --device cuda:0
```

`python -m mambagir prepare-data --help` describes how to construct a manifest from a local dataset layout. After tokenizer training, `python -m mambagir.reproduction export` exports the identifier-to-image mapping in gallery order with its manifest. The released weights use the manifests and mappings in the paired release archive.
