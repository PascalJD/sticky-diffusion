from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from sticky.rng import ensure_prng_key, make_rng

Array = jnp.ndarray
LOGGER = logging.getLogger(__name__)
_WARNED_MIXED_ANCHOR_CONFIG_IDS: set[int] = set()
_LEGACY_ANCHOR_KEYS = (
    "anchor_dim",
    "anchor_init",
    "learnable_anchors",
    "anchors_init_std",
    "anchor_seed",
    "anchor_order_weight",
    "anchor_residual_weight",
    "anchor_projection_seed",
    "anchor_transform",
    "anchor_center_columns",
    "anchor_whiten",
    "anchor_whiten_eps",
    "anchor_equalize_row_norms",
    "anchor_target_row_norm",
    "anchor_scale",
)
_LEGACY_ANCHOR_FAMILY_ALIASES = {
    "ordered_random_residual": "ordered_normal",
    "thermometer_projected": "thermometer",
}


def _ordered_scalar_table(vocab_size: int, dtype: jnp.dtype) -> Array:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size == 1:
        return jnp.zeros((1, 1), dtype=dtype)

    idx = jnp.arange(vocab_size, dtype=dtype)
    denom = jnp.asarray(vocab_size - 1, dtype=dtype)
    values = -1.0 + (2.0 * idx / denom)
    return values[:, None]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return cfg[key]
    except Exception:
        return default


def _cfg_has(cfg: Any, key: str) -> bool:
    missing = object()
    return _cfg_get(cfg, key, missing) is not missing


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _unwrap_model_cfg(cfg: Any) -> Any:
    maybe_model = _cfg_get(cfg, "model", None)
    if maybe_model is None:
        return cfg
    if _cfg_get(maybe_model, "name", None) is not None:
        return maybe_model
    return cfg


def _anchor_cfg(model_cfg: Any) -> Any:
    anchor_cfg = _cfg_get(model_cfg, "anchor", None)
    return anchor_cfg if anchor_cfg is not None else None


def _warn_if_mixed_anchor_config(model_cfg: Any) -> None:
    anchor_cfg = _anchor_cfg(model_cfg)
    if anchor_cfg is None:
        return

    flat_keys_present = [key for key in _LEGACY_ANCHOR_KEYS if _cfg_has(model_cfg, key)]
    if not flat_keys_present:
        return

    cfg_id = id(model_cfg)
    if cfg_id in _WARNED_MIXED_ANCHOR_CONFIG_IDS:
        return
    _WARNED_MIXED_ANCHOR_CONFIG_IDS.add(cfg_id)
    LOGGER.warning(
        "Detected both nested `model.anchor.*` config and legacy flat anchor fields "
        "(%s). Nested anchor values take precedence over overlapping flat fields.",
        ", ".join(sorted(flat_keys_present)),
    )


def _resolve_anchor_value(
    model_cfg: Any,
    *,
    nested_key: str,
    flat_key: str,
    default: Any,
) -> Any:
    anchor_cfg = _anchor_cfg(model_cfg)
    if (anchor_cfg is not None) and _cfg_has(anchor_cfg, nested_key):
        return _cfg_get(anchor_cfg, nested_key)
    if _cfg_has(model_cfg, flat_key):
        return _cfg_get(model_cfg, flat_key)
    return default


def _resolve_anchor_transform_value(
    model_cfg: Any,
    *,
    nested_key: str,
    flat_cfg_key: str,
    flat_scalar_key: str,
    default: Any,
) -> Any:
    anchor_cfg = _anchor_cfg(model_cfg)
    nested_transform_cfg = (
        _cfg_get(anchor_cfg, "transform", None) if anchor_cfg is not None else None
    )
    if (nested_transform_cfg is not None) and _cfg_has(nested_transform_cfg, nested_key):
        return _cfg_get(nested_transform_cfg, nested_key)

    flat_transform_cfg = _cfg_get(model_cfg, flat_cfg_key, None)
    if (flat_transform_cfg is not None) and _cfg_has(flat_transform_cfg, nested_key):
        return _cfg_get(flat_transform_cfg, nested_key)

    if _cfg_has(model_cfg, flat_scalar_key):
        return _cfg_get(model_cfg, flat_scalar_key)
    return default


def _normalize_anchor_family(value: Any) -> str:
    family = str(value).lower()
    return _LEGACY_ANCHOR_FAMILY_ALIASES.get(family, family)


def _ensure_positive_int(name: str, value: int) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _resolve_random_key(
    *,
    seed: int | None,
    rng: Array | None,
    context: str,
) -> Array:
    if seed is not None:
        return make_rng(int(seed))
    if rng is not None:
        return ensure_prng_key(rng)
    raise ValueError(
        f"{context} requires either an explicit seed in the anchor config or "
        "an RNG key supplied at build time."
    )


def _fix_qr_signs(q: Array, r: Array) -> Array:
    diag = jnp.diag(r)
    signs = jnp.where(diag < 0, -1.0, 1.0).astype(q.dtype)
    return q * signs[None, :]


def _orthogonal_projection_matrix(
    *,
    input_dim: int,
    output_dim: int,
    key: Array,
    dtype: jnp.dtype,
) -> Array:
    input_dim = _ensure_positive_int("input_dim", input_dim)
    output_dim = _ensure_positive_int("output_dim", output_dim)

    if output_dim <= input_dim:
        gaussian = jax.random.normal(key, (input_dim, output_dim), dtype=dtype)
        q, r = jnp.linalg.qr(gaussian, mode="reduced")
        return _fix_qr_signs(q, r)

    gaussian = jax.random.normal(key, (output_dim, input_dim), dtype=dtype)
    q, r = jnp.linalg.qr(gaussian, mode="reduced")
    q = _fix_qr_signs(q, r)
    return jnp.swapaxes(q, 0, 1)


def _build_normal_table(
    *,
    vocab_size: int,
    anchor_dim: int,
    init_std: float,
    seed: int | None,
    rng: Array | None,
    dtype: jnp.dtype,
) -> Array:
    key = _resolve_random_key(seed=seed, rng=rng, context="normal anchors")
    return (
        jax.random.normal(key, (vocab_size, anchor_dim), dtype=dtype)
        * jnp.asarray(float(init_std), dtype=dtype)
    )


def _build_ordered_normal_table(
    *,
    vocab_size: int,
    anchor_dim: int,
    init_std: float,
    seed: int | None,
    rng: Array | None,
    order_weight: float,
    residual_weight: float,
    dtype: jnp.dtype,
) -> Array:
    if anchor_dim < 2:
        raise ValueError(
            "ordered_normal anchors require anchor_dim >= 2, "
            f"got anchor_dim={anchor_dim}."
        )
    ordered = (
        _ordered_scalar_table(vocab_size, dtype)
        * jnp.asarray(float(order_weight), dtype=dtype)
    )
    residual = _build_normal_table(
        vocab_size=vocab_size,
        anchor_dim=anchor_dim - 1,
        init_std=init_std,
        seed=seed,
        rng=rng,
        dtype=dtype,
    )
    residual = residual * jnp.asarray(float(residual_weight), dtype=dtype)
    return jnp.concatenate([ordered, residual], axis=1)


def _build_thermometer_table(
    *,
    vocab_size: int,
    anchor_dim: int,
    projection_seed: int | None,
    seed: int | None,
    rng: Array | None,
    dtype: jnp.dtype,
) -> Array:
    ids = jnp.arange(vocab_size)
    thermometer = (ids[None, :] <= ids[:, None]).astype(dtype)
    thermometer = thermometer - jnp.mean(thermometer, axis=0, keepdims=True)

    key = _resolve_random_key(
        seed=projection_seed if projection_seed is not None else seed,
        rng=rng,
        context="thermometer anchors",
    )
    projection = _orthogonal_projection_matrix(
        input_dim=vocab_size,
        output_dim=anchor_dim,
        key=key,
        dtype=dtype,
    )
    return thermometer @ projection


@dataclass(frozen=True)
class AnchorTransformConfig:
    center_columns: bool = False
    whiten: bool = False
    whiten_eps: float = 1e-5
    equalize_row_norms: bool = False
    target_row_norm: float | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class AnchorTableConfig:
    family: str
    vocab_size: int
    anchor_dim: int
    init_std: float = 1.0
    seed: int | None = None
    order_weight: float = 1.0
    residual_weight: float = 1.0
    projection_seed: int | None = None
    pretrained_path: str | None = None
    transform: AnchorTransformConfig = field(default_factory=AnchorTransformConfig)
    # CDCD §3.2: when True, every read of the anchor table returns
    # (table / ||table||_row) * sqrt(anchor_dim). Backprop flows through the
    # normalization, so the underlying parameter is free to vary while the
    # anchors the model ever sees stay on the d-sphere of radius sqrt(d).
    # When enabled, the AnchorTransformConfig knobs `equalize_row_norms`,
    # `target_row_norm`, and `scale` become redundant.
    normalize_at_use: bool = False


@dataclass(frozen=True)
class AnchorsSpec:
    """Backward-compatible minimal description of an anchor table."""

    vocab_size: int
    anchor_dim: int


@dataclass(frozen=True)
class AnchorTableViews:
    """Anchor table snapshots for debugging and offline inspection."""

    raw: Array
    transformed: Array
    final: Array


def anchor_table_config_from_mapping(
    cfg: Any,
    *,
    vocab_size: int | None = None,
) -> AnchorTableConfig:
    model_cfg = _unwrap_model_cfg(cfg)
    _warn_if_mixed_anchor_config(model_cfg)

    resolved_vocab_size = vocab_size
    if resolved_vocab_size is None:
        resolved_vocab_size = _cfg_get(model_cfg, "vocab_size", None)
    if resolved_vocab_size is None:
        raise ValueError(
            "vocab_size must be provided either explicitly or in the model config."
        )

    return AnchorTableConfig(
        family=_normalize_anchor_family(
            _resolve_anchor_value(
                model_cfg,
                nested_key="family",
                flat_key="anchor_init",
                default="normal",
            )
        ),
        vocab_size=_ensure_positive_int("vocab_size", resolved_vocab_size),
        anchor_dim=_ensure_positive_int(
            "anchor_dim",
            int(
                _resolve_anchor_value(
                    model_cfg,
                    nested_key="dim",
                    flat_key="anchor_dim",
                    default=1,
                )
            ),
        ),
        init_std=float(
            _resolve_anchor_value(
                model_cfg,
                nested_key="init_std",
                flat_key="anchors_init_std",
                default=1.0,
            )
        ),
        seed=_maybe_int(
            _resolve_anchor_value(
                model_cfg,
                nested_key="seed",
                flat_key="anchor_seed",
                default=None,
            )
        ),
        order_weight=float(
            _resolve_anchor_value(
                model_cfg,
                nested_key="order_weight",
                flat_key="anchor_order_weight",
                default=1.0,
            )
        ),
        residual_weight=float(
            _resolve_anchor_value(
                model_cfg,
                nested_key="residual_weight",
                flat_key="anchor_residual_weight",
                default=1.0,
            )
        ),
        projection_seed=_maybe_int(
            _resolve_anchor_value(
                model_cfg,
                nested_key="projection_seed",
                flat_key="anchor_projection_seed",
                default=None,
            )
        ),
        pretrained_path=(
            lambda v: None if v is None else str(v)
        )(
            _resolve_anchor_value(
                model_cfg,
                nested_key="pretrained_path",
                flat_key="anchor_pretrained_path",
                default=None,
            )
        ),
        transform=AnchorTransformConfig(
            center_columns=bool(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="center_columns",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_center_columns",
                    default=False,
                )
            ),
            whiten=bool(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="whiten",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_whiten",
                    default=False,
                )
            ),
            whiten_eps=float(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="whiten_eps",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_whiten_eps",
                    default=1e-5,
                )
            ),
            equalize_row_norms=bool(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="equalize_row_norms",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_equalize_row_norms",
                    default=False,
                )
            ),
            target_row_norm=_maybe_float(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="target_row_norm",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_target_row_norm",
                    default=None,
                )
            ),
            scale=float(
                _resolve_anchor_transform_value(
                    model_cfg,
                    nested_key="scale",
                    flat_cfg_key="anchor_transform",
                    flat_scalar_key="anchor_scale",
                    default=1.0,
                )
            ),
        ),
        normalize_at_use=bool(
            _resolve_anchor_value(
                model_cfg,
                nested_key="normalize_at_use",
                flat_key="anchor_normalize_at_use",
                default=False,
            )
        ),
    )


def anchor_learnable_from_mapping(cfg: Any, *, default: bool = True) -> bool:
    model_cfg = _unwrap_model_cfg(cfg)
    _warn_if_mixed_anchor_config(model_cfg)
    return bool(
        _resolve_anchor_value(
            model_cfg,
            nested_key="learnable",
            flat_key="learnable_anchors",
            default=default,
        )
    )


def apply_anchor_transforms(
    table: Array,
    transform: AnchorTransformConfig,
    *,
    include_scale: bool = True,
) -> Array:
    table = jnp.asarray(table, dtype=jnp.float32)
    if table.ndim != 2:
        raise ValueError(
            f"Anchor tables must have rank 2, got shape={tuple(table.shape)}."
        )

    if table.shape[1] <= 1:
        out = table
        scale = float(transform.scale)
        if include_scale and (scale != 1.0):
            out = out * jnp.asarray(scale, dtype=out.dtype)
        return out

    if float(transform.whiten_eps) <= 0.0:
        raise ValueError(
            f"transform.whiten_eps must be positive, got {transform.whiten_eps}."
        )
    if (transform.target_row_norm is not None) and (
        float(transform.target_row_norm) < 0.0
    ):
        raise ValueError(
            "transform.target_row_norm must be non-negative when provided, got "
            f"{transform.target_row_norm}."
        )

    out = table
    if transform.center_columns:
        out = out - jnp.mean(out, axis=0, keepdims=True)

    if transform.whiten:
        cov = (out.T @ out) / jnp.asarray(max(1, out.shape[0]), dtype=out.dtype)
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        eigvals = jnp.maximum(eigvals, 0.0)
        inv_sqrt = jax.lax.rsqrt(
            eigvals + jnp.asarray(float(transform.whiten_eps), dtype=out.dtype)
        )
        whitener = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T
        out = out @ whitener

    if transform.equalize_row_norms:
        row_norms = jnp.linalg.norm(out, axis=1, keepdims=True)
        if transform.target_row_norm is None:
            target_norm = jnp.mean(row_norms)
        else:
            target_norm = jnp.asarray(
                float(transform.target_row_norm), dtype=out.dtype
            )
        out = out * (
            target_norm / jnp.maximum(row_norms, jnp.asarray(1e-12, dtype=out.dtype))
        )

    scale = float(transform.scale)
    if include_scale and (scale != 1.0):
        out = out * jnp.asarray(scale, dtype=out.dtype)

    return out


def _build_anchor_base_table(
    config: AnchorTableConfig,
    *,
    dtype: jnp.dtype = jnp.float32,
    rng: Array | None = None,
) -> Array:
    family = _normalize_anchor_family(config.family)
    vocab_size = _ensure_positive_int("vocab_size", int(config.vocab_size))
    anchor_dim = _ensure_positive_int("anchor_dim", int(config.anchor_dim))

    if family == "ordered_scalar":
        if anchor_dim != 1:
            raise ValueError(
                "ordered_scalar anchors require anchor_dim=1, "
                f"got anchor_dim={anchor_dim}."
            )
        table = _ordered_scalar_table(vocab_size, dtype)
    elif family == "normal":
        table = _build_normal_table(
            vocab_size=vocab_size,
            anchor_dim=anchor_dim,
            init_std=config.init_std,
            seed=config.seed,
            rng=rng,
            dtype=dtype,
        )
    elif family == "ordered_normal":
        table = _build_ordered_normal_table(
            vocab_size=vocab_size,
            anchor_dim=anchor_dim,
            init_std=config.init_std,
            seed=config.seed,
            rng=rng,
            order_weight=config.order_weight,
            residual_weight=config.residual_weight,
            dtype=dtype,
        )
    elif family == "thermometer":
        table = _build_thermometer_table(
            vocab_size=vocab_size,
            anchor_dim=anchor_dim,
            projection_seed=config.projection_seed,
            seed=config.seed,
            rng=rng,
            dtype=dtype,
        )
    elif family == "pretrained":
        from sticky.models.sjd.pretrained_anchors import load_pretrained_anchor_table

        if config.pretrained_path is None:
            raise ValueError(
                "anchor.pretrained_path must be set when anchor.family='pretrained'."
            )
        table = load_pretrained_anchor_table(
            path=config.pretrained_path,
            vocab_size=vocab_size,
            anchor_dim=anchor_dim,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unknown anchor initializer {config.family!r}. Expected one of: "
            "normal, ordered_normal, ordered_scalar, thermometer, pretrained."
        )

    expected_shape = (vocab_size, anchor_dim)
    if tuple(table.shape) != expected_shape:
        raise ValueError(
            f"Anchor family {config.family!r} produced shape {tuple(table.shape)}, "
            f"expected {expected_shape}."
        )
    return table.astype(dtype)


def build_anchor_table_views(
    config: AnchorTableConfig,
    *,
    dtype: jnp.dtype = jnp.float32,
    rng: Array | None = None,
) -> AnchorTableViews:
    raw = _build_anchor_base_table(config, dtype=dtype, rng=rng)
    transformed = apply_anchor_transforms(
        raw,
        config.transform,
        include_scale=False,
    )
    final = apply_anchor_transforms(
        raw,
        config.transform,
        include_scale=True,
    )
    expected_shape = (int(config.vocab_size), int(config.anchor_dim))
    for stage_name, table in (
        ("raw", raw),
        ("transformed", transformed),
        ("final", final),
    ):
        if tuple(table.shape) != expected_shape:
            raise ValueError(
                f"Anchor {stage_name} table has shape {tuple(table.shape)}, "
                f"expected {expected_shape}."
            )
    return AnchorTableViews(
        raw=raw.astype(dtype),
        transformed=transformed.astype(dtype),
        final=final.astype(dtype),
    )


def build_anchor_table(
    config: AnchorTableConfig,
    *,
    dtype: jnp.dtype = jnp.float32,
    rng: Array | None = None,
) -> Array:
    views = build_anchor_table_views(config, dtype=dtype, rng=rng)
    table = views.final

    expected_shape = (int(config.vocab_size), int(config.anchor_dim))
    if tuple(table.shape) != expected_shape:
        raise ValueError(
            "Anchor transform pipeline changed the anchor table shape from "
            f"{expected_shape} to {tuple(table.shape)}."
        )
    return table.astype(dtype)


def build_anchor_table_from_config(
    cfg: Any,
    *,
    vocab_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
    rng: Array | None = None,
) -> Array:
    config = anchor_table_config_from_mapping(cfg, vocab_size=vocab_size)
    return build_anchor_table(config, dtype=dtype, rng=rng)


def make_anchor_initializer(
    *,
    name: str | None = None,
    vocab_size: int | None = None,
    anchor_dim: int | None = None,
    init_std: float = 1.0,
    config: AnchorTableConfig | None = None,
):
    if config is None:
        if name is None or vocab_size is None or anchor_dim is None:
            raise ValueError(
                "make_anchor_initializer requires either `config` or the legacy "
                "`name`, `vocab_size`, and `anchor_dim` arguments."
            )
        config = AnchorTableConfig(
            family=_normalize_anchor_family(name),
            vocab_size=int(vocab_size),
            anchor_dim=int(anchor_dim),
            init_std=float(init_std),
        )

    expected = (int(config.vocab_size), int(config.anchor_dim))

    def _init(key, shape, dtype=jnp.float32):
        if tuple(shape) != expected:
            raise ValueError(
                f"anchor initializer expected shape {expected}, got {tuple(shape)}."
            )
        return build_anchor_table(config, dtype=dtype, rng=key)

    return _init


class TokenAnchors(nn.Module):
    """Learnable (or fixed) anchor lookup table."""

    config: AnchorTableConfig
    learnable: bool = True
    learnable_log_w: bool = False
    log_w_init: Any = None
    # When set, log_w_float() clamps the recentered log_w into this range.
    # Hazard-likelihood is unidentified at the optimum (corrected appendix
    # rem:hazard-flat), so keep this loose: [-3, 3] => w in [~0.05, 20]. A
    # tight clamp can pin the very signal the experiment measures. Defaults
    # to None (no clamp) for back-compat; learned_e2e configs wire this to
    # hazard_weighting.clip via the SJD model factory.
    log_w_clip: tuple[float, float] | None = None

    def _normalize(self, table: Array) -> Array:
        """Apply CDCD-style L2-normalize-then-scale-by-sqrt(d), if enabled.

        Pure JAX function: differentiable when called from a tracing context,
        so gradients on the underlying parameter flow through the normalization
        as required by CDCD §3.2.
        """
        if not self.config.normalize_at_use:
            return table
        norms = jnp.linalg.norm(table, axis=-1, keepdims=True)
        unit = table / jnp.maximum(norms, 1e-12)
        scale = jnp.sqrt(jnp.asarray(table.shape[-1], dtype=table.dtype))
        return unit * scale

    def _recenter_log_w(self, log_w: Array) -> Array:
        # β(t) absorbs additive shifts to log_w, leaving the parameter free
        # to drift along an unidentifiable direction. Subtract the mean every
        # forward use so gradients only push log_w along identifiable
        # directions.
        log_w = log_w.astype(jnp.float32)
        return log_w - jnp.mean(log_w)

    @nn.compact
    def __call__(self, ids: Array) -> Array:
        table = self.param(
            "table",
            make_anchor_initializer(config=self.config),
            (int(self.config.vocab_size), int(self.config.anchor_dim)),
            jnp.float32,
        )
        if not self.learnable:
            table = jax.lax.stop_gradient(table)
        table = self._normalize(table)
        if self.learnable_log_w:
            log_w_init = self.log_w_init

            def _log_w_initializer(key, shape, dtype=jnp.float32):
                del key
                if log_w_init is not None:
                    return jnp.asarray(log_w_init, dtype=dtype).reshape(shape)
                return jnp.zeros(shape, dtype=dtype)

            self.param(
                "log_w",
                _log_w_initializer,
                (int(self.config.vocab_size),),
                jnp.float32,
            )
        return jnp.take(table, ids, axis=0)

    def table_float(self) -> Array:
        table = self.get_variable("params", "table")
        if not self.learnable:
            table = jax.lax.stop_gradient(table)
        table = self._normalize(table)
        return table.astype(jnp.float32)

    def log_w_float(self) -> Array | None:
        if not self.learnable_log_w:
            return None
        log_w = self.get_variable("params", "log_w")
        log_w = self._recenter_log_w(log_w)
        # Single chokepoint clamp: everything downstream (loss, sampling,
        # metrics, log_lambda_hat quadrature) reads through this accessor.
        # Without the clamp, the raw param can drift to overflow w = exp(log_w)
        # in float32 and produce NaNs via 0 * inf in the prior term, even with
        # prior_strength=0. (See plan v3 H1+H2.)
        if self.log_w_clip is not None:
            lo, hi = self.log_w_clip
            log_w = jnp.clip(log_w, float(lo), float(hi))
        return log_w


@struct.dataclass
class AnchorTable:
    """Frozen view of an anchor table for sampling."""

    table_float: Array
    log_w: Array | None = None

    @property
    def L(self) -> int:
        return int(self.table_float.shape[0])

    @property
    def d(self) -> int:
        return int(self.table_float.shape[1])

    @property
    def effective_log_w(self) -> Array:
        # Frequency-weighted hazard: log of the per-anchor weight w(a) such that
        # lambda_t(a) = beta(t) * exp(log_w[a]). Default zeros => w(a) == 1, i.e.
        # the anchor-agnostic baseline lambda_t(a) = beta(t).
        if self.log_w is None:
            return jnp.zeros((int(self.table_float.shape[0]),), dtype=jnp.float32)
        return jnp.asarray(self.log_w, dtype=jnp.float32)


def clamp_known_state(
    *,
    y: Array,
    committed: Array,
    k_idx: Array,
    known_mask: Array | None,
    known_idx: Array | None,
    a_table: Array,
) -> tuple[Array, Array, Array]:
    if known_mask is None or known_idx is None:
        return y, committed, k_idx
    known_vec = a_table[jnp.asarray(known_idx, dtype=jnp.int32)]
    y = jnp.where(known_mask[..., None], known_vec, y)
    committed = jnp.where(known_mask, True, committed)
    k_idx = jnp.where(known_mask, known_idx, k_idx)
    return y, committed, k_idx
