from __future__ import annotations

from dataclasses import dataclass
import inspect
import sys
import types
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mllm_dr.prompts import PHQ8_CLINICAL_DESCRIPTIONS

from .lq_former import LQFormer


@dataclass
class MLlmDROutput:
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    mse_loss: torch.Tensor | None
    regression_scores: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor
    tract_mse_loss: torch.Tensor | None = None
    tract_scores: torch.Tensor | None = None
    tract_score_probs: torch.Tensor | None = None
    tract_ordinal_loss: torch.Tensor | None = None
    distill_kl_loss: torch.Tensor | None = None


class MLlmDR(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        tokenizer: Any,
        audio_marker: str = "<AudioHere>",
        video_marker: str = "<VideoHere>",
        query_tokens: int = 32,
        query_dim: int = 768,
        lq_hidden_dim: int = 1024,
        lq_layers: int = 4,
        lq_heads: int = 8,
        lq_dropout: float = 0.3,
        llm_embed_dim: int | None = None,
        audio_input_dim: int = 768,
        video_input_dim: int = 2048,
        use_audio_lqformer: bool = True,
        use_video_lqformer: bool = True,
        use_aspect_conditioning: bool = False,
        aspect_descriptions: list[str] | tuple[str, ...] | None = None,
        aspect_embedding_table: torch.Tensor | None = None,
        use_tract_raft: bool = False,
        tract_score_values: list[int] | tuple[int, ...] = (0, 1, 2, 3),
        tract_score_token_prefix: str = " ",
        tract_raft_weight: float = 1.0,
        tract_loss_type: str = "expected_mse",
        tract_ordinal_loss_weight: float = 0.0,
        tract_ordinal_sigma: float = 0.6,
        distill_kl_weight: float = 0.0,
        temporal_pack_segments: int = 3,
        temporal_pack_tokens_per_segment: int = 64,
        query_group_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.audio_marker = audio_marker
        self.video_marker = video_marker
        self.audio_marker_id = tokenizer.convert_tokens_to_ids(audio_marker)
        self.video_marker_id = tokenizer.convert_tokens_to_ids(video_marker)
        self.llm_embed_dim = llm_embed_dim or llm.get_input_embeddings().embedding_dim
        self.use_audio_lqformer = use_audio_lqformer
        self.use_video_lqformer = use_video_lqformer
        self.use_aspect_conditioning = use_aspect_conditioning
        self.use_tract_raft = use_tract_raft
        self.tract_raft_weight = tract_raft_weight
        self.tract_loss_type = str(tract_loss_type or "expected_mse").strip().lower()
        if self.tract_loss_type not in {"expected_mse", "ordinal_emd", "hybrid"}:
            raise ValueError(
                "tract_loss_type must be expected_mse, ordinal_emd, or hybrid"
            )
        self.tract_ordinal_loss_weight = float(tract_ordinal_loss_weight)
        self.tract_ordinal_sigma = max(1e-3, float(tract_ordinal_sigma))
        self.distill_kl_weight = float(distill_kl_weight)
        if use_aspect_conditioning:
            if aspect_embedding_table is None:
                descriptions = aspect_descriptions or PHQ8_CLINICAL_DESCRIPTIONS
                aspect_embedding_table = self._encode_aspect_descriptions(descriptions)
            if aspect_embedding_table.ndim != 2:
                raise ValueError("aspect_embedding_table must have shape [aspects, dim]")
            if aspect_embedding_table.shape[-1] != self.llm_embed_dim:
                raise ValueError(
                    "aspect_embedding_table dim must match the LLM embedding dim"
                )
            self.register_buffer(
                "aspect_embedding_table",
                aspect_embedding_table.detach().float(),
                persistent=True,
            )
        else:
            self.register_buffer(
                "aspect_embedding_table",
                torch.empty(0, self.llm_embed_dim),
                persistent=False,
            )
        self.audio_lqformer = (
            LQFormer(
                input_dim=int(audio_input_dim),
                query_dim=query_dim,
                num_queries=query_tokens,
                num_layers=lq_layers,
                num_heads=lq_heads,
                hidden_dim=lq_hidden_dim,
                output_dim=self.llm_embed_dim,
                dropout=lq_dropout,
                aspect_embed_dim=self.llm_embed_dim,
                use_aspect_conditioning=use_aspect_conditioning,
                temporal_pack_segments=temporal_pack_segments,
                temporal_pack_tokens_per_segment=temporal_pack_tokens_per_segment,
                query_group_sizes=query_group_sizes,
            )
            if use_audio_lqformer
            else None
        )
        self.video_lqformer = (
            LQFormer(
                input_dim=int(video_input_dim),
                query_dim=query_dim,
                num_queries=query_tokens,
                num_layers=lq_layers,
                num_heads=lq_heads,
                hidden_dim=lq_hidden_dim,
                output_dim=self.llm_embed_dim,
                dropout=lq_dropout,
                aspect_embed_dim=self.llm_embed_dim,
                use_aspect_conditioning=use_aspect_conditioning,
                temporal_pack_segments=temporal_pack_segments,
                temporal_pack_tokens_per_segment=temporal_pack_tokens_per_segment,
                query_group_sizes=query_group_sizes,
            )
            if use_video_lqformer
            else None
        )
        if use_tract_raft:
            score_token_ids = self._score_token_ids(
                tokenizer=tokenizer,
                score_values=tract_score_values,
                token_prefix=tract_score_token_prefix,
            )
            self.register_buffer(
                "tract_score_token_ids",
                torch.tensor(score_token_ids, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "tract_score_values",
                torch.tensor(tract_score_values, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "tract_score_token_ids",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "tract_score_values",
                torch.empty(0, dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def _tokenize_fragment(tokenizer: Any, text: str) -> list[int]:
        try:
            encoded = tokenizer(
                text,
                truncation=False,
                padding=False,
                add_special_tokens=False,
                return_tensors=None,
            )
        except TypeError:
            encoded = tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    @classmethod
    def _score_token_ids(
        cls,
        tokenizer: Any,
        score_values: list[int] | tuple[int, ...],
        token_prefix: str,
    ) -> list[int]:
        token_ids: list[int] = []
        for value in score_values:
            text = f"{token_prefix}{int(value)}"
            ids = cls._tokenize_fragment(tokenizer, text)
            if len(ids) != 1:
                fallback = cls._tokenize_fragment(tokenizer, str(int(value)))
                if len(fallback) == 1:
                    ids = fallback
            if len(ids) != 1:
                raise ValueError(
                    "TRACT score candidates must each tokenize to one token. "
                    f"Score {value!r} produced token ids {ids!r}. Set "
                    "model.tract_score_token_prefix for this tokenizer."
                )
            token_ids.append(ids[0])
        return token_ids

    def _encode_aspect_descriptions(
        self,
        descriptions: list[str] | tuple[str, ...],
    ) -> torch.Tensor:
        embedding_layer = self.llm.get_input_embeddings()
        device = embedding_layer.weight.device
        encoded = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids).float()
            mask = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
            pooled = (token_embeddings * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
        return pooled.detach().cpu()

    def _aspect_embeddings_for_batch(
        self,
        aspect_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_aspect_conditioning:
            return None
        if aspect_ids is None:
            raise ValueError(
                "aspect_ids must be provided when aspect conditioning is enabled"
            )
        aspect_ids = aspect_ids.to(device=self.aspect_embedding_table.device).long()
        if aspect_ids.ndim != 1 or aspect_ids.shape[0] != batch_size:
            raise ValueError("aspect_ids must have shape [batch]")
        aspect_embeddings = self.aspect_embedding_table.index_select(0, aspect_ids)
        return aspect_embeddings.to(device=device)

    def freeze_llm(self) -> None:
        for param in self.llm.parameters():
            param.requires_grad = False

    def freeze_lqformers(self) -> None:
        for module in (self.audio_lqformer, self.video_lqformer):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def _iter_llm_related_modules(self):
        queue = [self.llm]
        seen: set[int] = set()
        while queue:
            module = queue.pop(0)
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            yield module
            for attr in ("module", "base_model", "model", "language_model"):
                child = getattr(module, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

    def _llm_needs_input_ids_with_inputs_embeds(self) -> bool:
        for candidate in self._iter_llm_related_modules():
            config = getattr(candidate, "config", None)
            text = " ".join(
                str(value)
                for value in (
                    candidate.__class__.__name__,
                    candidate.__class__.__module__,
                    getattr(config, "model_type", ""),
                    getattr(config, "architectures", ""),
                )
            ).lower()
            if "gemma4" in text:
                return True
        return False

    def _gemma4_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
    ) -> Any | None:
        for candidate in self._iter_llm_related_modules():
            get_per_layer_inputs = getattr(candidate, "get_per_layer_inputs", None)
            if get_per_layer_inputs is None:
                continue
            try:
                return get_per_layer_inputs(input_ids, None)
            except TypeError:
                try:
                    return get_per_layer_inputs(input_ids)
                except TypeError:
                    return get_per_layer_inputs(
                        input_ids=input_ids,
                        inputs_embeds=None,
                    )
        return None

    def _prepare_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        aspect_ids: torch.Tensor | None,
        audio_features: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
        video_features: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        score_token_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        audio_embeds = None
        video_embeds = None
        aspect_embeddings = self._aspect_embeddings_for_batch(
            aspect_ids=aspect_ids,
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )
        if self.audio_lqformer is not None and audio_features is not None:
            audio_embeds = self.audio_lqformer(
                audio_features,
                audio_mask,
                aspect_embeddings=aspect_embeddings,
            )
            audio_embeds = audio_embeds.to(dtype=token_embeds.dtype)
        if self.video_lqformer is not None and video_features is not None:
            video_embeds = self.video_lqformer(
                video_features,
                video_mask,
                aspect_embeddings=aspect_embeddings,
            )
            video_embeds = video_embeds.to(dtype=token_embeds.dtype)

        embed_rows: list[torch.Tensor] = []
        id_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        for i in range(input_ids.shape[0]):
            sample_embeds: list[torch.Tensor] = []
            sample_ids: list[torch.Tensor] = []
            sample_mask: list[torch.Tensor] = []
            sample_labels: list[torch.Tensor] = []
            sample_score_mask: list[torch.Tensor] = []
            for j in range(input_ids.shape[1]):
                if attention_mask[i, j].item() == 0:
                    continue
                token_id = int(input_ids[i, j])
                replacement = None
                if token_id == self.audio_marker_id and audio_embeds is not None:
                    replacement = audio_embeds[i]
                elif token_id == self.video_marker_id and video_embeds is not None:
                    replacement = video_embeds[i]
                if replacement is None:
                    sample_embeds.append(token_embeds[i, j].unsqueeze(0))
                    sample_ids.append(input_ids[i, j].view(1))
                    sample_mask.append(torch.ones(1, device=input_ids.device))
                    if labels is not None:
                        sample_labels.append(labels[i, j].view(1))
                    if score_token_mask is not None:
                        sample_score_mask.append(score_token_mask[i, j].view(1))
                else:
                    sample_embeds.append(replacement)
                    sample_ids.append(
                        torch.full(
                            (replacement.shape[0],),
                            token_id,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        )
                    )
                    sample_mask.append(torch.ones(replacement.shape[0], device=input_ids.device))
                    if labels is not None:
                        sample_labels.append(
                            torch.full(
                                (replacement.shape[0],),
                                -100,
                                dtype=labels.dtype,
                                device=input_ids.device,
                            )
                        )
                    if score_token_mask is not None:
                        sample_score_mask.append(
                            torch.zeros(
                                replacement.shape[0],
                                dtype=score_token_mask.dtype,
                                device=input_ids.device,
                            )
                        )
            embed_rows.append(torch.cat(sample_embeds, dim=0))
            id_rows.append(torch.cat(sample_ids, dim=0))
            mask_rows.append(torch.cat(sample_mask, dim=0))
            if labels is not None:
                label_rows.append(torch.cat(sample_labels, dim=0))
            if score_token_mask is not None:
                score_rows.append(torch.cat(sample_score_mask, dim=0))

        max_len = max(row.shape[0] for row in embed_rows)
        batch = len(embed_rows)
        embeds = token_embeds.new_zeros(batch, max_len, token_embeds.shape[-1])
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        out_input_ids = input_ids.new_full((batch, max_len), pad_token_id)
        masks = attention_mask.new_zeros(batch, max_len)
        out_labels = None
        if labels is not None:
            out_labels = labels.new_full((batch, max_len), -100)
        out_score_mask = None
        if score_token_mask is not None:
            out_score_mask = score_token_mask.new_zeros(batch, max_len)
        for i, row in enumerate(embed_rows):
            length = row.shape[0]
            embeds[i, :length] = row
            out_input_ids[i, :length] = id_rows[i]
            masks[i, :length] = mask_rows[i].to(dtype=masks.dtype)
            if out_labels is not None:
                out_labels[i, :length] = label_rows[i]
            if out_score_mask is not None:
                out_score_mask[i, :length] = score_rows[i]
        return (
            embeds,
            masks,
            out_labels,
            out_score_mask,
            audio_embeds,
            video_embeds,
            out_input_ids,
        )

    def _tract_score_distribution(
        self,
        logits: torch.Tensor,
        score_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.use_tract_raft or score_token_mask is None:
            return None
        score_token_mask = score_token_mask.bool()
        if not score_token_mask.any():
            return None
        has_score = score_token_mask.any(dim=1)
        if not has_score.all():
            return None
        positions = score_token_mask.float().argmax(dim=1).long()
        if (positions <= 0).any():
            return None
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        score_logits = logits.float()[batch_indices, positions - 1]
        candidate_ids = self.tract_score_token_ids.to(device=logits.device)
        candidate_values = self.tract_score_values.to(device=logits.device)
        candidate_logits = score_logits.index_select(dim=-1, index=candidate_ids)
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        scores = (candidate_probs * candidate_values).sum(dim=-1)
        return scores, candidate_probs

    def _ordinal_target_distribution(
        self,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        values = self.tract_score_values.to(device=device).float()
        labels = labels.float().to(device=device).unsqueeze(-1)
        sigma = max(1e-3, float(self.tract_ordinal_sigma))
        logits = -0.5 * ((values.unsqueeze(0) - labels) / sigma) ** 2
        target = torch.softmax(logits, dim=-1)
        return target

    def _ordinal_emd_loss(
        self,
        score_probs: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = self._ordinal_target_distribution(labels, score_probs.device)
        pred_cdf = score_probs.float().cumsum(dim=-1)
        target_cdf = target.float().cumsum(dim=-1)
        losses = (pred_cdf - target_cdf).pow(2).mean(dim=-1)
        if weights is not None:
            weights = weights.float().to(losses.device)
            return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        return losses.mean()

    def _teacher_distribution_kl_loss(
        self,
        score_probs: torch.Tensor,
        teacher_score_probs: torch.Tensor | None,
        teacher_score_prob_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if teacher_score_probs is None or self.distill_kl_weight <= 0:
            return None
        teacher = teacher_score_probs.float().to(score_probs.device)
        if teacher.shape != score_probs.shape:
            return None
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_sample = torch.sum(
            teacher * (teacher.clamp_min(1e-8).log() - score_probs.float().clamp_min(1e-8).log()),
            dim=-1,
        )
        if teacher_score_prob_mask is not None:
            mask = teacher_score_prob_mask.bool().to(per_sample.device)
            if not bool(mask.any()):
                return None
            return per_sample[mask].mean()
        return per_sample.mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        aspect_ids: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        score_token_mask: torch.Tensor | None = None,
        regression_labels: torch.Tensor | None = None,
        regression_weights: torch.Tensor | None = None,
        teacher_score_probs: torch.Tensor | None = None,
        teacher_score_prob_mask: torch.Tensor | None = None,
        use_mse: bool = True,
    ) -> MLlmDROutput:
        (
            inputs_embeds,
            mm_attention_mask,
            mm_labels,
            mm_score_token_mask,
            audio_embeds,
            video_embeds,
            mm_input_ids,
        ) = self._prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            aspect_ids=aspect_ids,
            audio_features=audio_features,
            audio_mask=audio_mask,
            video_features=video_features,
            video_mask=video_mask,
            score_token_mask=score_token_mask,
        )
        llm_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": mm_attention_mask,
            "labels": mm_labels,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if self._llm_needs_input_ids_with_inputs_embeds():
            per_layer_inputs = self._gemma4_per_layer_inputs(mm_input_ids)
            if per_layer_inputs is None:
                raise RuntimeError(
                    "Gemma4 inputs_embeds path requires a language_model with "
                    "get_per_layer_inputs; check the loaded Transformers/PEFT "
                    "model wrapper."
                )
            llm_kwargs["per_layer_inputs"] = per_layer_inputs
        outputs = self.llm(**llm_kwargs)
        hidden = outputs.hidden_states[-1]
        tract_distribution = self._tract_score_distribution(
            logits=outputs.logits,
            score_token_mask=mm_score_token_mask,
        )
        tract_scores = tract_distribution[0] if tract_distribution is not None else None
        tract_score_probs = (
            tract_distribution[1] if tract_distribution is not None else None
        )
        if (
            self.use_tract_raft
            and regression_labels is not None
            and use_mse
            and tract_scores is None
        ):
            raise ValueError(
                "TRACT/CoT-RAFT training requires a non-empty score_token_mask "
                "for every sample. Check model.score_after_rationale and the "
                "tokenizer score format."
            )
        regression_scores = tract_scores
        mse_loss = None
        tract_mse_loss = None
        tract_ordinal_loss = None
        distill_kl_loss = None
        if regression_labels is not None and use_mse:
            if tract_scores is not None:
                loss_terms: list[torch.Tensor] = []
                if self.tract_loss_type in {"expected_mse", "hybrid"}:
                    tract_errors = F.mse_loss(
                        tract_scores.float(),
                        regression_labels.float(),
                        reduction="none",
                    )
                    if regression_weights is not None:
                        weights = regression_weights.float().to(tract_errors.device)
                        tract_mse_loss = (
                            (tract_errors * weights).sum()
                            / weights.sum().clamp_min(1e-6)
                        )
                    else:
                        tract_mse_loss = tract_errors.mean()
                    loss_terms.append(tract_mse_loss)
                if self.tract_loss_type in {"ordinal_emd", "hybrid"}:
                    if tract_score_probs is None:
                        raise ValueError("ordinal TRACT loss requires score probabilities")
                    tract_ordinal_loss = self._ordinal_emd_loss(
                        tract_score_probs,
                        regression_labels,
                        weights=regression_weights,
                    )
                    ordinal_weight = (
                        self.tract_ordinal_loss_weight
                        if self.tract_loss_type == "hybrid"
                        else 1.0
                    )
                    loss_terms.append(ordinal_weight * tract_ordinal_loss)
                if tract_score_probs is not None:
                    distill_kl_loss = self._teacher_distribution_kl_loss(
                        tract_score_probs,
                        teacher_score_probs=teacher_score_probs,
                        teacher_score_prob_mask=teacher_score_prob_mask,
                    )
                    if distill_kl_loss is not None:
                        loss_terms.append(self.distill_kl_weight * distill_kl_loss)
                if loss_terms:
                    mse_loss = self.tract_raft_weight * sum(loss_terms)
        lm_loss = outputs.loss
        loss = lm_loss
        if mse_loss is not None:
            loss = mse_loss if loss is None else loss + mse_loss
        return MLlmDROutput(
            loss=loss,
            lm_loss=lm_loss,
            mse_loss=mse_loss,
            regression_scores=regression_scores,
            logits=outputs.logits,
            hidden_states=hidden,
            tract_mse_loss=tract_mse_loss,
            tract_scores=tract_scores,
            tract_score_probs=tract_score_probs,
            tract_ordinal_loss=tract_ordinal_loss,
            distill_kl_loss=distill_kl_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_ids: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        (
            inputs_embeds,
            mm_attention_mask,
            _,
            _,
            _,
            _,
            mm_input_ids,
        ) = self._prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            aspect_ids=aspect_ids,
            audio_features=audio_features,
            audio_mask=audio_mask,
            video_features=video_features,
            video_mask=video_mask,
        )
        generate_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": mm_attention_mask,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **generation_kwargs,
        }
        if self._llm_needs_input_ids_with_inputs_embeds():
            per_layer_inputs = self._gemma4_per_layer_inputs(mm_input_ids)
            if per_layer_inputs is None:
                raise RuntimeError(
                    "Gemma4 generation with inputs_embeds requires a "
                    "language_model with get_per_layer_inputs; check the loaded "
                    "Transformers/PEFT model wrapper."
                )
            generate_kwargs["per_layer_inputs"] = per_layer_inputs
        return self.llm.generate(**generate_kwargs)


def _quantization_kwargs(training_cfg: dict[str, Any]) -> dict[str, Any]:
    if training_cfg.get("load_in_4bit") or training_cfg.get("load_in_8bit"):
        from transformers import BitsAndBytesConfig

        if training_cfg.get("load_in_4bit"):
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
        }
    return {}


def _forward_accepts_inputs_embeds(model: nn.Module) -> bool:
    signature = inspect.signature(model.forward)
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "inputs_embeds" in signature.parameters


def _lora_target_matches(module_name: str, target_name: str) -> bool:
    return module_name == target_name or module_name.endswith(f".{target_name}")


def _normalize_lora_target_modules(
    model: nn.Module,
    target_modules: list[str],
) -> list[str]:
    named_modules = list(model.named_modules())
    normalized: list[str] = []
    for target_name in target_modules:
        target_name = str(target_name).strip()
        if not target_name:
            continue
        matches = [
            (module_name, module)
            for module_name, module in named_modules
            if _lora_target_matches(module_name, target_name)
        ]
        if any(isinstance(module, nn.Linear) for _, module in matches):
            normalized.append(target_name)
            continue
        nested_target = f"{target_name}.linear"
        nested_matches = [
            (module_name, module)
            for module_name, module in named_modules
            if _lora_target_matches(module_name, nested_target)
        ]
        if (
            matches
            and any(type(module).__name__ == "Gemma4ClippableLinear" for _, module in matches)
            and any(isinstance(module, nn.Linear) for _, module in nested_matches)
        ):
            normalized.append(nested_target)
            continue
        normalized.append(target_name)
    return normalized


def _install_transformers_torch_checkpoint_shim() -> None:
    try:
        import torch.distributed.checkpoint.hf_storage  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import torch.distributed.checkpoint as checkpoint
    except Exception:
        return

    module = types.ModuleType("torch.distributed.checkpoint.hf_storage")

    class _UnavailableHfStorage:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "torch.distributed.checkpoint.hf_storage is unavailable in this "
                "PyTorch build. Upgrade PyTorch before using Transformers "
                "distributed Hugging Face checkpoint storage."
            )

    for name in (
        "HuggingFaceStorageWriter",
        "HuggingFaceStorageReader",
        "QuantizedHuggingFaceStorageReader",
    ):
        existing = getattr(checkpoint, name, None)
        setattr(module, name, existing or _UnavailableHfStorage)
    sys.modules[module.__name__] = module


def _load_text_tokenizer(
    model_name: str,
    *,
    processor_first: bool,
    use_fast: bool,
    pretrained_kwargs: dict[str, Any],
) -> Any:
    _install_transformers_torch_checkpoint_shim()
    from transformers import AutoTokenizer

    if processor_first:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_name,
                **pretrained_kwargs,
            )
            tokenizer = getattr(processor, "tokenizer", processor)
            if hasattr(tokenizer, "add_special_tokens"):
                return tokenizer
        except (OSError, ValueError):
            pass
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            **pretrained_kwargs,
        )
    except (OSError, ValueError) as tokenizer_exc:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_name,
                **pretrained_kwargs,
            )
            tokenizer = getattr(processor, "tokenizer", processor)
        except (OSError, ValueError) as processor_exc:
            raise OSError(
                f"Unable to load tokenizer or processor for {model_name!r}. "
                "Check Hugging Face access/login or set "
                "`model.base_model_name_or_path` to a local model path."
            ) from processor_exc
        if not hasattr(tokenizer, "add_special_tokens"):
            raise OSError(
                f"Processor for {model_name!r} does not expose a text tokenizer."
            ) from tokenizer_exc
        return tokenizer


def load_mllm_dr(config: dict[str, Any]) -> MLlmDR:
    _install_transformers_torch_checkpoint_shim()
    from transformers import AutoModelForCausalLM

    model_cfg = config["model"]
    training_cfg = config.get("training", {})
    model_name = model_cfg["base_model_name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    pretrained_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    tokenizer = _load_text_tokenizer(
        model_name,
        processor_first=bool(model_cfg.get("processor_first", False)),
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
        pretrained_kwargs=pretrained_kwargs,
    )
    special_tokens = [
        model_cfg.get("audio_marker", "<AudioHere>"),
        model_cfg.get("video_marker", "<VideoHere>"),
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if training_cfg.get("bf16") else None
    load_kwargs: dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if training_cfg.get("device_map") is not None:
        load_kwargs["device_map"] = training_cfg.get("device_map")
    if model_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_cfg.get("attn_implementation")
    load_kwargs.update(pretrained_kwargs)
    load_kwargs.update(_quantization_kwargs(training_cfg))
    try:
        llm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except OSError as exc:
        raise OSError(
            f"Unable to load base model {model_name!r}. Check Hugging Face "
            "access/login or point the config at a local model directory."
        ) from exc
    if not _forward_accepts_inputs_embeds(llm):
        raise RuntimeError(
            f"Base model {model_name!r} does not expose an inputs_embeds forward "
            "argument, which is required for multimodal token insertion."
        )
    llm.resize_token_embeddings(len(tokenizer))
    if training_cfg.get("gradient_checkpointing"):
        if hasattr(llm, "config"):
            llm.config.use_cache = False
        llm.gradient_checkpointing_enable()
    if model_cfg.get("llm_embed_dim") is None:
        model_cfg = dict(model_cfg)
        model_cfg["llm_embed_dim"] = llm.get_input_embeddings().embedding_dim

    model = MLlmDR(
        llm=llm,
        tokenizer=tokenizer,
        audio_marker=model_cfg.get("audio_marker", "<AudioHere>"),
        video_marker=model_cfg.get("video_marker", "<VideoHere>"),
        query_tokens=int(model_cfg.get("query_tokens", 32)),
        query_dim=int(model_cfg.get("query_dim", 768)),
        lq_hidden_dim=int(model_cfg.get("lq_hidden_dim", 1024)),
        lq_layers=int(model_cfg.get("lq_layers", 4)),
        lq_heads=int(model_cfg.get("lq_heads", 8)),
        lq_dropout=float(model_cfg.get("lq_dropout", 0.3)),
        temporal_pack_segments=int(model_cfg.get("temporal_pack_segments", 3)),
        temporal_pack_tokens_per_segment=int(
            model_cfg.get("temporal_pack_tokens_per_segment", 64)
        ),
        query_group_sizes=model_cfg.get("query_group_sizes"),
        llm_embed_dim=int(model_cfg["llm_embed_dim"]),
        audio_input_dim=int(model_cfg.get("audio_input_dim", 768)),
        video_input_dim=int(model_cfg.get("video_input_dim", 2048)),
        use_audio_lqformer=bool(model_cfg.get("use_audio_lqformer", True)),
        use_video_lqformer=bool(model_cfg.get("use_video_lqformer", True)),
        use_aspect_conditioning=bool(model_cfg.get("use_aspect_conditioning", False)),
        aspect_descriptions=model_cfg.get("aspect_descriptions"),
        use_tract_raft=bool(model_cfg.get("use_tract_raft", False)),
        tract_score_values=tuple(model_cfg.get("tract_score_values", [0, 1, 2, 3])),
        tract_score_token_prefix=str(
            model_cfg.get("tract_score_token_prefix", " ")
        ),
        tract_raft_weight=float(
            training_cfg.get(
                "tract_raft_weight",
                model_cfg.get("tract_raft_weight", 1.0),
            )
        ),
        tract_loss_type=str(
            training_cfg.get(
                "tract_loss_type",
                model_cfg.get("tract_loss_type", "expected_mse"),
            )
        ),
        tract_ordinal_loss_weight=float(
            training_cfg.get(
                "tract_ordinal_loss_weight",
                model_cfg.get("tract_ordinal_loss_weight", 0.0),
            )
        ),
        tract_ordinal_sigma=float(
            training_cfg.get(
                "tract_ordinal_sigma",
                model_cfg.get("tract_ordinal_sigma", 0.6),
            )
        ),
        distill_kl_weight=float(
            training_cfg.get(
                "distill_kl_weight",
                model_cfg.get("distill_kl_weight", 0.0),
            )
        ),
    )
    if training_cfg.get("stage") == "stage1":
        model.freeze_llm()
    elif training_cfg.get("stage") == "stage2":
        if model_cfg.get("freeze_lqformer_in_stage2", True):
            model.freeze_lqformers()
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("Stage 2 requires peft. Install requirements.txt.") from exc
        if training_cfg.get("load_in_4bit") or training_cfg.get("load_in_8bit"):
            model.llm = prepare_model_for_kbit_training(model.llm)
        lora_cfg = config.get("lora", {})
        modules_to_save = lora_cfg.get("modules_to_save")
        raw_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        if isinstance(raw_target_modules, str):
            target_modules = raw_target_modules
            if raw_target_modules != "all-linear":
                target_modules = _normalize_lora_target_modules(
                    model.llm,
                    [raw_target_modules],
                )
        else:
            target_modules = _normalize_lora_target_modules(
                model.llm,
                list(raw_target_modules),
            )
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, peft_cfg)
    return model
