# src/sticky/models/sjd/sjd_model.py
from __future__ import annotations
from collections.abc import Sequence
from typing import Mapping, Optional

import flax.linen as nn
import jax.numpy as jnp

from .anchors import AnchorTableConfig, TokenAnchors
from .classifier import ContinuousClassifier
from .joint_input import SudokuJointInputProj
from .multi_axis_input import MultiAxisInputProj
from .state_layout import StateLayout
from .sudoku_structural import SudokuStructuralAdapter

Array = jnp.ndarray


class SJD(nn.Module):
    anchor_config: AnchorTableConfig
    learnable_anchors: bool = True
    enable_joint_input: bool = False
    use_noisy_input_bias: bool = False
    # When True, project the cell-only input through an explicit Dense
    # (feature_dim) before handing it to the classifier. This makes the
    # cell-only path go through the same number of input projections as the
    # joint (slack-augmented) path, which routes cell features through
    # SudokuJointInputProj.cell_proj. Use this flag in baseline configs that
    # are compared against the slack-augmented model so the architectures
    # are apples-to-apples; defaults to False to preserve existing
    # checkpoints and the current behavior of cell-only tasks.
    cell_only_input_proj: bool = False

    # Multi-axis state-space partitioning (Part B). When `state_layout` is
    # set, the model exposes a `apply_layout` entry point that consumes a
    # state dict {axis_name: array} and runs the joint sequence through
    # MultiAxisInputProj before the classifier. The legacy `__call__`
    # path is preserved unchanged for cell-only tasks. `use_sudoku_structural`
    # toggles the SudokuStructuralAdapter (row/col/box/group_idx embeddings,
    # init-zero) for layouts that share the Sudoku adjacency. Both default
    # to None / False so existing tasks are byte-identical.
    state_layout: Optional[StateLayout] = None
    use_sudoku_structural: bool = False

    feature_dim: int = 128
    num_heads: int = 12
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)

    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    model_sharding: bool = False
    sequence_backbone: str = "auto"
    sequence_mlp_hidden_dim: int | None = None
    sequence_max_length: int | None = None
    sequence_causal: bool = False
    image_backbone: str = "auto"
    adm_num_res_blocks: int = 2
    adm_attention_resolutions: Sequence[int] = (2, 4, 8)
    adm_num_heads: int = 4
    adm_num_head_channels: int = -1
    adm_num_heads_upsample: int = -1
    adm_conv_resample: bool = True
    adm_use_scale_shift_norm: bool = True
    adm_resblock_updown: bool = False
    adm_use_conv_skip: bool = False
    adm_use_new_attention_order: bool = False

    vocab_size: int = 256

    def setup(self):
        if int(self.anchor_config.vocab_size) != int(self.vocab_size):
            raise ValueError(
                "SJD anchor_config.vocab_size must match model vocab_size, got "
                f"{self.anchor_config.vocab_size} vs {self.vocab_size}."
            )
        self.anchors = TokenAnchors(
            config=self.anchor_config,
            learnable=self.learnable_anchors,
        )
        self.classifier = ContinuousClassifier(
            n_layers=self.n_layers,
            n_dit_layers=self.n_dit_layers,
            dit_num_heads=self.dit_num_heads,
            dit_hidden_size=self.dit_hidden_size,
            ch_mult=self.ch_mult,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            mlp_type=self.mlp_type,
            depth_scaled_init=self.depth_scaled_init,
            cond_type=self.cond_type,
            model_sharding=self.model_sharding,
            sequence_backbone=self.sequence_backbone,
            sequence_mlp_hidden_dim=self.sequence_mlp_hidden_dim,
            sequence_max_length=self.sequence_max_length,
            sequence_causal=self.sequence_causal,
            image_backbone=self.image_backbone,
            adm_num_res_blocks=self.adm_num_res_blocks,
            adm_attention_resolutions=self.adm_attention_resolutions,
            adm_num_heads=self.adm_num_heads,
            adm_num_head_channels=self.adm_num_head_channels,
            adm_num_heads_upsample=self.adm_num_heads_upsample,
            adm_conv_resample=self.adm_conv_resample,
            adm_use_scale_shift_norm=self.adm_use_scale_shift_norm,
            adm_resblock_updown=self.adm_resblock_updown,
            adm_use_conv_skip=self.adm_use_conv_skip,
            adm_use_new_attention_order=self.adm_use_new_attention_order,
        )
        if self.use_noisy_input_bias:
            self.noisy_input_bias = self.param(
                "noisy_input_bias",
                nn.initializers.zeros,
                (int(self.anchor_config.anchor_dim),),
                jnp.float32,
            )
        if self.enable_joint_input:
            self.joint_input_proj = SudokuJointInputProj(
                feature_dim=self.feature_dim
            )
        if self.cell_only_input_proj:
            self.cell_only_input_dense = nn.Dense(
                int(self.feature_dim), name="cell_only_input_proj"
            )
        if self.state_layout is not None:
            anchored = self.state_layout.anchored_axes
            for axis in anchored:
                if int(axis.embedding_dim) != int(self.vocab_size):
                    raise ValueError(
                        f"SJD.state_layout: anchored axis {axis.name!r} has "
                        f"embedding_dim={axis.embedding_dim}, but model "
                        f"vocab_size={self.vocab_size}. They must match for "
                        "the per-anchor logit head to align with the table."
                    )
            self.multi_axis_input = MultiAxisInputProj(
                layout=self.state_layout,
                feature_dim=self.feature_dim,
            )
            if self.use_sudoku_structural:
                self.sudoku_structural = SudokuStructuralAdapter(
                    feature_dim=self.feature_dim
                )

    def embed(self, token_ids: Array) -> Array:
        return self.anchors(token_ids)

    def anchor_table(self) -> Array:
        return self.anchors.table_float()

    def __call__(
        self,
        y_t: Array,
        t: Array,
        *,
        cond=None,
        anchor_token_ids: Array | None = None,
        noisy_position_mask: Array | None = None,
        slack_y_t: Array | None = None,
        state: Mapping[str, Array] | None = None,
        train: bool = False,
    ):
        if anchor_token_ids is not None:
            _ = self.embed(anchor_token_ids)
        if state is not None:
            return self.apply_layout(
                state, t, cond=cond, train=train
            )
        if slack_y_t is None:
            cell_input = y_t
            if self.use_noisy_input_bias:
                if noisy_position_mask is None:
                    noisy_mask = jnp.zeros(cell_input.shape[:-1], dtype=jnp.bool_)
                else:
                    noisy_mask = jnp.asarray(noisy_position_mask, dtype=jnp.bool_)
                    if noisy_mask.shape != cell_input.shape[:-1]:
                        raise ValueError(
                            "noisy_position_mask must match y_t without the "
                            f"feature dimension, got {noisy_mask.shape} vs "
                            f"{cell_input.shape[:-1]}."
                        )
                cell_input = cell_input + noisy_mask[..., None].astype(
                    cell_input.dtype
                ) * self.noisy_input_bias.astype(cell_input.dtype)
            if self.cell_only_input_proj:
                cell_input = self.cell_only_input_dense(cell_input)
            return self.classifier(cell_input, t=t, cond=cond, train=train)
        if not self.enable_joint_input:
            raise ValueError(
                "slack_y_t was provided but enable_joint_input=False on SJD."
            )
        z = self.joint_input_proj(y_t, slack_y_t)
        return self.classifier(z, t=t, cond=cond, train=train)

    def apply_layout(
        self,
        state: Mapping[str, Array],
        t: Array,
        *,
        cond=None,
        train: bool = False,
    ):
        """Multi-axis forward path. Returns (per_axis_logits, aux) where
        `per_axis_logits` is a dict {axis_name: (B, site_count, vocab_size)}
        with one entry per anchored axis (axes that contribute to NLL).
        """
        if self.state_layout is None:
            raise ValueError(
                "SJD.apply_layout requires `state_layout` to be set on the "
                "module. Pass a StateLayout to construct the multi-axis path."
            )
        structural = (
            self.sudoku_structural() if self.use_sudoku_structural else None
        )
        z = self.multi_axis_input(state, structural_offsets=structural)
        logits, aux = self.classifier(z, t=t, cond=cond, train=train)
        per_axis = {}
        for axis in self.state_layout.axes:
            if axis.contributes_to_nll:
                sl = self.state_layout.slice_of(axis.name)
                per_axis[axis.name] = logits[:, sl, :]
        return per_axis, aux
