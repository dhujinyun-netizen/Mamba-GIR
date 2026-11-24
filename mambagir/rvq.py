"""Offline residual vector quantization with seeded PyTorch k-means."""

import torch
from torch import nn


class ResidualVectorQuantizer(nn.Module):
    """Fit each codebook to the residual remaining after the preceding levels.

    Fitting uses float32 k-means with k-means++ initialization on the input device.
    Empty clusters are reseeded from the largest-error rows. Fitted codebooks are
    frozen buffers; encode/reconstruct also execute on the input tensor's device.
    """

    def __init__(self, levels=4, codebook_size=256, seed=3407, iterations=25):
        super().__init__()
        if levels < 1 or codebook_size < 1 or iterations < 1:
            raise ValueError("levels, codebook_size, and iterations must be positive")
        self.levels = int(levels)
        self.codebook_size = int(codebook_size)
        self.seed = int(seed)
        self.iterations = int(iterations)
        self.register_buffer("codebooks", torch.empty(self.levels, self.codebook_size, 0))

    @staticmethod
    def _validate_features(features):
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("features must be a tensor with shape [N, D] and D > 0")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must contain finite floating-point values")

    @staticmethod
    def _nearest(features, centers):
        # Chunk the distance matrix so larger galleries do not require N x V memory.
        codes, errors = [], []
        norms = centers.square().sum(dim=1)
        for chunk in features.split(8192):
            distance = chunk.square().sum(dim=1, keepdim=True) + norms - 2 * chunk @ centers.T
            error, code = distance.min(dim=1)
            codes.append(code)
            errors.append(error.clamp_min(0))
        if not codes:
            return (torch.empty(0, dtype=torch.long, device=features.device),
                    torch.empty(0, dtype=features.dtype, device=features.device))
        return torch.cat(codes), torch.cat(errors)

    def _kmeans(self, features, generator):
        count = features.shape[0]
        centers = torch.empty(self.codebook_size, features.shape[1],
                              dtype=features.dtype, device=features.device)
        first = torch.randint(count, (), generator=generator, device=features.device).item()
        centers[0] = features[first]
        closest = (features - centers[0]).square().sum(dim=1)
        for index in range(1, self.codebook_size):
            if closest.sum() > 0:
                row = torch.multinomial(closest, 1, generator=generator).item()
            else:
                row = torch.randint(count, (), generator=generator, device=features.device).item()
            centers[index] = features[row]
            closest = torch.minimum(closest, (features - centers[index]).square().sum(dim=1))

        for _ in range(self.iterations):
            codes, errors = self._nearest(features, centers)
            counts = torch.bincount(codes, minlength=self.codebook_size)
            updated = torch.zeros_like(centers)
            updated.index_add_(0, codes, features)
            updated /= counts.clamp_min(1).unsqueeze(1)
            empty = (counts == 0).nonzero(as_tuple=True)[0]
            if empty.numel():
                farthest = torch.argsort(errors, descending=True, stable=True)[:empty.numel()]
                updated[empty] = features[farthest]
            if torch.equal(updated, centers):
                break
            centers = updated
        return centers

    @torch.no_grad()
    def fit(self, features: torch.Tensor):
        self._validate_features(features)
        if features.shape[0] < self.codebook_size:
            raise ValueError("RVQ fitting needs at least codebook_size feature rows")
        residual = features.detach().to(dtype=torch.float32).clone()
        generator = torch.Generator(device=features.device).manual_seed(self.seed)
        codebooks = []
        for _ in range(self.levels):
            centers = self._kmeans(residual, generator)
            codes, _ = self._nearest(residual, centers)
            residual = residual - centers[codes]
            codebooks.append(centers)
        self.codebooks = torch.stack(codebooks).to(device=features.device, dtype=features.dtype)
        return self

    def _require_fitted(self):
        if self.codebooks.ndim != 3 or self.codebooks.shape[:2] != (self.levels, self.codebook_size):
            raise ValueError("codebooks shape disagrees with the RVQ configuration")
        if self.codebooks.shape[-1] == 0:
            raise RuntimeError("RVQ must be fitted before encoding or reconstruction")

    @torch.no_grad()
    def encode(self, features):
        self._validate_features(features)
        self._require_fitted()
        if features.shape[1] != self.codebooks.shape[-1]:
            raise ValueError("features and codebooks must have the same feature dimension")
        # Float32 avoids half-precision squared-distance overflow for encoder features.
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        residual = features.detach().to(dtype=dtype).clone()
        codebooks = self.codebooks.to(device=features.device, dtype=dtype)
        assignments = []
        for centers in codebooks:
            codes, _ = self._nearest(residual, centers)
            assignments.append(codes)
            residual = residual - centers[codes]
        return torch.stack(assignments, dim=1)

    @torch.no_grad()
    def reconstruct(self, codes, cumulative=False):
        self._require_fitted()
        if not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.shape[1] != self.levels:
            raise ValueError("codes must have shape [N, levels]")
        if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("codes must contain integer indices")
        if torch.any(codes < 0) or torch.any(codes >= self.codebook_size):
            raise ValueError("codes must be in [0, codebook_size)")
        codebooks = self.codebooks.to(device=codes.device)
        stages = torch.arange(self.levels, device=codes.device)
        selected = codebooks[stages.unsqueeze(0), codes.long()]
        return selected.cumsum(dim=1) if cumulative else selected.sum(dim=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "codebooks"
        incoming = state_dict.get(key)
        if incoming is not None and incoming.ndim == 3 and incoming.shape[:2] == (
                self.levels, self.codebook_size):
            self.codebooks = torch.empty(incoming.shape, device=self.codebooks.device,
                                         dtype=self.codebooks.dtype)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
