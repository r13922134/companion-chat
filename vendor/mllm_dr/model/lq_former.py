from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


QUERY_GROUP_NAMES = ("global", "temporal", "clinical", "aspect")
DEFAULT_QUERY_GROUP_SIZES = {
    "global": 8,
    "temporal": 12,
    "clinical": 8,
    "aspect": 4,
}


def _default_query_group_sizes(num_queries: int) -> dict[str, int]:
    if num_queries == sum(DEFAULT_QUERY_GROUP_SIZES.values()):
        return dict(DEFAULT_QUERY_GROUP_SIZES)
    ratios = (0.25, 0.375, 0.25, 0.125)
    sizes = [int(num_queries * ratio) for ratio in ratios]
    remainder = num_queries - sum(sizes)
    order = (1, 0, 2, 3)
    for idx in order:
        if remainder <= 0:
            break
        sizes[idx] += 1
        remainder -= 1
    return dict(zip(QUERY_GROUP_NAMES, sizes, strict=True))


def _query_group_ids(
    num_queries: int,
    group_sizes: Mapping[str, int] | None,
) -> torch.Tensor:
    sizes = dict(group_sizes or _default_query_group_sizes(num_queries))
    unknown = sorted(set(sizes) - set(QUERY_GROUP_NAMES))
    if unknown:
        raise ValueError(f"Unknown query groups: {unknown}")
    ids: list[int] = []
    for group_idx, name in enumerate(QUERY_GROUP_NAMES):
        size = int(sizes.get(name, 0))
        if size < 0:
            raise ValueError(f"Query group {name!r} has negative size: {size}")
        ids.extend([group_idx] * size)
    if len(ids) != num_queries:
        raise ValueError(
            "query_group_sizes must sum to num_queries. "
            f"Got {len(ids)} grouped queries for num_queries={num_queries}."
        )
    return torch.tensor(ids, dtype=torch.long)


class _PerceiverResamplerLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(query_dim)
        self.cross_memory_norm = nn.LayerNorm(query_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(query_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=self.cross_query_norm(latents),
            key=self.cross_memory_norm(memory),
            value=self.cross_memory_norm(memory),
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        latents = latents + self.dropout(attended)
        attended, _ = self.self_attention(
            query=self.self_norm(latents),
            key=self.self_norm(latents),
            value=self.self_norm(latents),
            need_weights=False,
        )
        latents = latents + self.dropout(attended)
        return latents + self.dropout(self.ffn(self.ffn_norm(latents)))


class LQFormer(nn.Module):
    """Clinical temporal Perceiver resampler for long audio/visual features."""

    def __init__(
        self,
        input_dim: int,
        query_dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 1024,
        output_dim: int = 4096,
        dropout: float = 0.3,
        aspect_embed_dim: int | None = None,
        use_aspect_conditioning: bool = False,
        temporal_pack_segments: int = 3,
        temporal_pack_tokens_per_segment: int = 64,
        query_group_sizes: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if temporal_pack_segments <= 0:
            raise ValueError("temporal_pack_segments must be positive")
        if temporal_pack_tokens_per_segment <= 0:
            raise ValueError("temporal_pack_tokens_per_segment must be positive")
        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)
        self.temporal_pack_segments = int(temporal_pack_segments)
        self.temporal_pack_tokens_per_segment = int(temporal_pack_tokens_per_segment)
        self.use_aspect_conditioning = use_aspect_conditioning

        group_ids = _query_group_ids(self.num_queries, query_group_sizes)
        self.register_buffer("query_group_ids", group_ids, persistent=False)
        self.num_query_groups = len(QUERY_GROUP_NAMES)

        self.input_projection = (
            nn.Identity() if input_dim == query_dim else nn.Linear(input_dim, query_dim)
        )
        self.position_projection = nn.Linear(1, query_dim)
        self.segment_embedding = nn.Embedding(self.temporal_pack_segments, query_dim)
        self.local_depthwise = nn.Conv1d(
            query_dim,
            query_dim,
            kernel_size=3,
            padding=1,
            groups=query_dim,
        )
        self.local_pointwise = nn.Conv1d(query_dim, query_dim, kernel_size=1)
        self.local_norm = nn.LayerNorm(query_dim)
        self.local_dropout = nn.Dropout(dropout)

        self.pack_queries = nn.Parameter(
            torch.empty(
                self.temporal_pack_segments,
                self.temporal_pack_tokens_per_segment,
                query_dim,
            )
        )
        self.pack_query_norm = nn.LayerNorm(query_dim)
        self.pack_memory_norm = nn.LayerNorm(query_dim)
        self.pack_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pack_norm = nn.LayerNorm(query_dim)
        self.pack_dropout = nn.Dropout(dropout)

        self.query_tokens = nn.Parameter(torch.empty(self.num_queries, query_dim))
        self.query_group_embedding = nn.Embedding(self.num_query_groups, query_dim)
        self.aspect_film = None
        if use_aspect_conditioning:
            aspect_embed_dim = aspect_embed_dim or query_dim
            self.aspect_film = nn.Sequential(
                nn.Linear(aspect_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_query_groups * query_dim * 2),
            )

        self.resampler_layers = nn.ModuleList(
            [
                _PerceiverResamplerLayer(
                    query_dim=query_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.pack_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.query_group_embedding.weight, mean=0.0, std=0.02)
        if self.aspect_film is not None:
            final = self.aspect_film[-1]
            if isinstance(final, nn.Linear):
                nn.init.normal_(final.weight, mean=0.0, std=0.01)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _ensure_nonempty_mask(feature_mask: torch.Tensor) -> torch.Tensor:
        feature_mask = feature_mask.bool()
        empty = ~feature_mask.any(dim=1)
        if empty.any():
            feature_mask = feature_mask.clone()
            feature_mask[empty, 0] = True
        return feature_mask

    def _relative_positions(
        self,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_index = feature_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        lengths = feature_mask.long().sum(dim=1, keepdim=True).clamp_min(1)
        denominator = (lengths - 1).clamp_min(1)
        relative = valid_index.float() / denominator.float()
        return relative.masked_fill(~feature_mask, 0.0)

    def _segment_ids(
        self,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        relative = self._relative_positions(feature_mask)
        segment_ids = torch.floor(relative * self.temporal_pack_segments).long()
        segment_ids = segment_ids.clamp(max=self.temporal_pack_segments - 1)
        return segment_ids.masked_fill(~feature_mask, 0)

    def _add_temporal_encoding(
        self,
        memory: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative = self._relative_positions(feature_mask)
        centered = relative.mul(2.0).sub(1.0).unsqueeze(-1)
        segment_ids = self._segment_ids(feature_mask)
        memory = (
            memory
            + self.position_projection(centered.to(dtype=memory.dtype))
            + self.segment_embedding(segment_ids)
        )
        return memory, segment_ids

    def _local_temporal_encode(
        self,
        memory: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = memory * feature_mask.unsqueeze(-1).to(dtype=memory.dtype)
        local = self.local_depthwise(masked.transpose(1, 2))
        local = torch.nn.functional.gelu(local)
        local = self.local_pointwise(local).transpose(1, 2)
        local = self.local_dropout(local)
        memory = self.local_norm(memory + local)
        return memory * feature_mask.unsqueeze(-1).to(dtype=memory.dtype)

    def _segment_key_padding_mask(
        self,
        feature_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        segment_idx: int,
    ) -> torch.Tensor:
        valid = feature_mask & (segment_ids == segment_idx)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty] = feature_mask[empty]
        return ~valid

    def _pack_memory(
        self,
        memory: torch.Tensor,
        feature_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_segments: list[torch.Tensor] = []
        for segment_idx in range(self.temporal_pack_segments):
            queries = self.pack_queries[segment_idx].unsqueeze(0).expand(
                memory.shape[0],
                -1,
                -1,
            )
            attended, _ = self.pack_attention(
                query=self.pack_query_norm(queries),
                key=self.pack_memory_norm(memory),
                value=self.pack_memory_norm(memory),
                key_padding_mask=self._segment_key_padding_mask(
                    feature_mask,
                    segment_ids,
                    segment_idx,
                ),
                need_weights=False,
            )
            packed_segments.append(self.pack_norm(queries + self.pack_dropout(attended)))
        packed = torch.cat(packed_segments, dim=1)
        packed_mask = torch.ones(
            packed.shape[:2],
            dtype=torch.bool,
            device=packed.device,
        )
        return packed, ~packed_mask

    def _typed_queries(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        aspect_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        group_ids = self.query_group_ids.to(device=device)
        group_offsets = self.query_group_embedding(group_ids)
        queries = self.query_tokens + group_offsets
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=device, dtype=dtype)
        if self.aspect_film is None:
            return queries
        if aspect_embeddings is None:
            raise ValueError(
                "aspect_embeddings must be provided when aspect conditioning is enabled"
            )
        if aspect_embeddings.ndim != 2:
            raise ValueError("aspect_embeddings must have shape [batch, dim]")
        if aspect_embeddings.shape[0] != batch_size:
            raise ValueError("aspect_embeddings batch size must match features")
        film = self.aspect_film(
            aspect_embeddings.to(device=device, dtype=queries.dtype)
        )
        film = film.view(batch_size, self.num_query_groups, 2, self.query_dim)
        gamma = film[:, :, 0].index_select(1, group_ids).tanh()
        beta = film[:, :, 1].index_select(1, group_ids)
        return queries * (1.0 + gamma) + beta

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
        aspect_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, seq, dim]")
        features = features.to(dtype=self.query_tokens.dtype)
        memory = self.input_projection(features)
        batch_size, seq_len, _ = memory.shape
        if feature_mask is None:
            feature_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=memory.device,
            )
        else:
            feature_mask = feature_mask.to(device=memory.device).bool()
            if feature_mask.shape != memory.shape[:2]:
                raise ValueError("feature_mask must have shape [batch, seq]")
        feature_mask = self._ensure_nonempty_mask(feature_mask)
        memory, segment_ids = self._add_temporal_encoding(memory, feature_mask)
        memory = self._local_temporal_encode(memory, feature_mask)
        packed_memory, packed_key_padding_mask = self._pack_memory(
            memory,
            feature_mask,
            segment_ids,
        )
        latents = self._typed_queries(
            batch_size=batch_size,
            device=memory.device,
            dtype=memory.dtype,
            aspect_embeddings=aspect_embeddings,
        )
        for layer in self.resampler_layers:
            latents = layer(latents, packed_memory, packed_key_padding_mask)
        return self.projection(latents)
