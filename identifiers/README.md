# Gallery identifier assignments

The four JSON files provide identifier-to-image mappings for the fixed retrieval galleries. Each file is paired with the same dataset's split manifest in [`manifests/`](../manifests/README.md).

Each entry contains a four-symbol `identifier` and its ordered `members`. Every member gives its zero-based `gallery_row`, `target_row` in the manifest, relative image `path`, and `label`. All images sharing an identifier are retained in gallery order.

| Mapping | Gallery images | Distinct identifiers |
|---|---:|---:|
| [cub.json](cub.json) | 5,994 | 5,874 |
| [cars.json](cars.json) | 8,144 | 7,656 |
| [aircraft.json](aircraft.json) | 6,667 | 6,579 |
| [inshop.json](inshop.json) | 12,612 | 4,057 |

`identifier_length` is 4 and `codebook_size` is 256. `source_assignment` names the tokenizer assignment artifact from which the mapping was exported. `target_row` indexes `targets` in the paired manifest, and `gallery_row` indexes its `gallery_indices` list. A member's path is relative to that dataset's image root.

For a generated identifier, locate the entry with the matching four-symbol list and return its members in the stored order. Multiple members can share the same identifier.

```python
import json
from pathlib import Path

mapping = json.loads(Path("identifiers/inshop.json").read_text())
index = {tuple(entry["identifier"]): entry["members"]
         for entry in mapping["entries"]}
# Resolve a predicted identifier with index.get(tuple(predicted_identifier), []).
```

Use the assignments paired with the generator trained on those identifiers. For a new tokenizer training run, the [export command](../README.md#5-export-a-retrieval-index) writes its corresponding assignments in gallery order.
