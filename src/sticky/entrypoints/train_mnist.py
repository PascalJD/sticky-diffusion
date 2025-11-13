from __future__ import annotations

import os
import time
from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig


def configure_runtime(cfg: DictConfig):
    """
    Set environment flags BEFORE importing JAX or any module that imports JAX.
    """
    # device selection
    dev = str(cfg.runtime.device).lower()
    if dev in ("cpu", "gpu"):
        os.environ.setdefault("JAX_PLATFORMS", dev)
    # safer memory behavior on GPU
    if cfg.runtime.xla_preallocate is False:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if cfg.runtime.xla_mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(cfg.runtime.xla_mem_fraction))


def maybe_init_wandb(cfg: DictConfig):
    if not bool(cfg.wandb.enabled):
        return None
    import wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.get("tags", [])),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


@hydra.main(config_path="../../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 0) Print config
    print(OmegaConf.to_yaml(cfg))

    # 1) Runtime env must be set BEFORE any JAX import
    configure_runtime(cfg)

    # 2) Now import JAX/Flax/Optax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training import train_state
    from flax.serialization import to_bytes

    # Data & forward mechanics
    from sticky.data.mnist_discrete import load_mnist_split
    from sticky.core.anchors import AnalogBitsAnchors
    from sticky.core.sde_vp import make_beta
    from sticky.core.hazard import make_hazard_early
    from sticky.core.jump import GaussianJump

    # Models
    from sticky.models.classifier_heads import AllocatorHead
    from sticky.models.intensity_head import IntensityHead

    # Train step (CE + DHM)
    from sticky.trainers.train_step import DualTrainState, ForwardProcess, train_step

    # Reverse sampler for eval
    from sticky.core.sampler import SamplerConfig, reverse_sample

    def _make_grid(imgs_hw, cols: int = 8, pad: int = 2):
        """
        imgs_hw: (B,H,W) in [0,1] -> (H_grid, W_grid) float32 in [0,1]
        """
        imgs = np.asarray(imgs_hw)
        b, h, w = imgs.shape
        cols = max(1, int(cols))
        rows = int(np.ceil(b / cols))
        grid_h = rows * h + (rows - 1) * pad
        grid_w = cols * w + (cols - 1) * pad
        grid = np.ones((grid_h, grid_w), dtype=np.float32)  # white padding
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= b:
                    break
                y0 = r * (h + pad)
                x0 = c * (w + pad)
                grid[y0:y0 + h, x0:x0 + w] = imgs[idx]
                idx += 1
        return grid

    def _to_uint8(img_01: np.ndarray) -> np.ndarray:
        return np.clip(np.round(img_01 * 255.0), 0, 255).astype(np.uint8)

    def log_eval_images(step, rng, state, anchors, beta, wb_run, batch_np):
        """
        Produce sample grid + ground-truth grids, and log sampler metrics.
        """
        if wb_run is None:
            return

        import wandb

        # Config with safe fallbacks
        eval_every = int(getattr(cfg.training, "eval_every", 500))
        num_samples = int(getattr(cfg.training, "num_samples", 36))
        grid_cols = int(getattr(cfg.training, "grid_cols", 6))

        sampler_dict = dict(
            T=float(cfg.forward.beta.T),
            n_steps=int(getattr(getattr(cfg, "sampler", {}), "n_steps", 250)),
            alloc_mode=str(getattr(getattr(cfg, "sampler", {}), "alloc_mode", "argmax")),
            score_from_classifier=bool(getattr(getattr(cfg, "sampler", {}), "score_from_classifier", True)),
            score_scale=float(getattr(getattr(cfg, "sampler", {}), "score_scale", 1.0)),
            force_classify_at_end=bool(getattr(getattr(cfg, "sampler", {}), "force_classify_at_end", True)),
            init_std=float(getattr(getattr(cfg, "sampler", {}), "init_std", 1.0)),
            eps_denom=float(getattr(getattr(cfg, "sampler", {}), "eps_denom", 1e-12)),
            rng_fold=int(getattr(getattr(cfg, "sampler", {}), "rng_fold", 4)),
        )
        scfg = SamplerConfig(**sampler_dict)
        H = W = 28

        rng, ksample = jax.random.split(rng)
        res = reverse_sample(
            ksample,
            state.cls.params,
            state.haz.params,
            state.cls.apply_fn,
            state.haz.apply_fn,
            anchors=anchors,
            beta=beta,
            shape_hw=(H, W),
            B=num_samples,
            cfg=scfg,
        )
        # Convert final anchor indices to displayable pixels
        k_final = np.asarray(res.k) 
        samples_01 = np.asarray((k_final + 0.5) / float(anchors.L), dtype=np.float32)
        samples_grid = _make_grid(samples_01, cols=grid_cols, pad=2)

        b = min(batch_np.shape[0], num_samples)
        gt = batch_np[:b].reshape(b, H, W)
        gt_grid = _make_grid(np.asarray(gt), cols=grid_cols, pad=2)
        # quantized to L bins (bin centers)
        k_gt = np.clip(np.floor(gt * float(anchors.L)).astype(np.int32), 0, anchors.L - 1)
        gtq_01 = (k_gt.astype(np.float32) + 0.5) / float(anchors.L)
        gtq_grid = _make_grid(gtq_01, cols=grid_cols, pad=2)

        sm = {k: float(np.asarray(v)) for k, v in res.metrics.items()}
        wandb.log(
            {
                "samples/grid": wandb.Image(_to_uint8(samples_grid), caption="SJD samples"),
                "examples/gt_grid": wandb.Image(_to_uint8(gt_grid), caption="MNIST ground truth"),
                "examples/gt_quantized_grid": wandb.Image(_to_uint8(gtq_grid), caption=f"Quantized (L={anchors.L})"),
                **{f"samples/{k}": v for k, v in sm.items()},
            },
            step=step,
        )

    # 3) Data iterator (choose backend)
    if cfg.dataset.name == "mnist_numpy":
        from sticky.data.mnist_numpy import iter_mnist_split
        train_iter = iter_mnist_split(
            split="train",
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            seed=cfg.seed,
            drop_remainder=cfg.dataset.drop_remainder,
            data_dir=cfg.dataset.get("data_dir"),
        )
    elif cfg.dataset.name == "mnist_discrete":
        # TF/TFDS-based loader (requires a TensorFlow install compatible with ml-dtypes)
        # This is giving me a headache
        from sticky.data.mnist_discrete import load_mnist_split
        train_iter = load_mnist_split(
            split="train",
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            seed=cfg.seed,
            drop_remainder=cfg.dataset.drop_remainder,
            num_workers=int(getattr(cfg.training, "num_workers", 0)),
        )
    else:
        raise ValueError(f"Unknown dataset.name={cfg.dataset.name!r}")
    print(f"[jax] backend={jax.default_backend()}  devices={jax.devices()}")

    # 4) Anchors & forward process
    anchors = AnalogBitsAnchors(L=int(cfg.dataset.L), d=None, gray=bool(cfg.dataset.gray_code))
    L = anchors.L
    d = anchors.d
    print(f"[anchors] L={L}, d={d}, gray={anchors.gray}")

    beta = make_beta(cfg.forward.beta.beta_min, cfg.forward.beta.beta_max, T=cfg.forward.beta.T)
    hazard = make_hazard_early(beta, kappa=cfg.forward.hazard.kappa)
    jump = GaussianJump(std=cfg.forward.jump.std, clip=cfg.forward.jump.clip)
    fwd = ForwardProcess(beta=beta, hazard=hazard, jump=jump, T=cfg.forward.beta.T)

    # 5) Models + states
    def make_optimizers():
        def mk(lr: float):
            return optax.chain(
                optax.clip_by_global_norm(cfg.optim.grad_clip),
                optax.adamw(lr, weight_decay=cfg.optim.weight_decay),
            )
        return mk(cfg.optim.lr_cls), mk(cfg.optim.lr_haz)

    def create_train_states(rng):
        H = W = 28
        cls_model = AllocatorHead(
            L=L,
            ch=cfg.model.allocator.ch,
            depth=cfg.model.allocator.depth,
            num_groups=cfg.model.allocator.num_groups,
            temb_dim=cfg.model.allocator.temb_dim,
            tfeat_dim=cfg.model.allocator.tfeat_dim,
        )
        haz_model = IntensityHead(
            ch=cfg.model.intensity.ch,
            depth=cfg.model.intensity.depth,
            num_groups=cfg.model.intensity.num_groups,
            temb_dim=cfg.model.intensity.temb_dim,
            tfeat_dim=cfg.model.intensity.tfeat_dim,
            min_lambda=cfg.model.intensity.min_lambda,
        )
        key1, key2 = jax.random.split(rng)
        y_dummy = jnp.zeros((1, H, W, d), dtype=jnp.float32)
        t_dummy = jnp.zeros((1,), dtype=jnp.float32)

        params_cls = cls_model.init(key1, y_dummy, t_dummy)
        params_haz = haz_model.init(key2, y_dummy, t_dummy)

        opt_cls, opt_haz = make_optimizers()
        state = DualTrainState(
            cls=train_state.TrainState.create(apply_fn=cls_model.apply, params=params_cls, tx=opt_cls),
            haz=train_state.TrainState.create(apply_fn=haz_model.apply, params=params_haz, tx=opt_haz),
        )
        return state

    # RNG
    rng = jax.random.PRNGKey(cfg.seed)
    rng, key_init = jax.random.split(rng)
    state = create_train_states(key_init)

    # 6) JIT train step (closed over anchors & forward process)
    step_fn = jax.jit(partial(
        train_step,
        anchors=anchors,
        fwd=fwd,
        L=L,
        ce_weight=1.0,
        dhm_weight=1.0,
    ))

    # 7) Wandb & loop
    wb_run = maybe_init_wandb(cfg)
    log_every = int(cfg.training.log_every)
    ckpt_every = int(cfg.training.ckpt_every)
    eval_every = int(getattr(cfg.training, "eval_every", 500))
    out_dir = os.getcwd()  # hydra run dir

    def save_ckpt(state, step: int):
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        path_cls = os.path.join(out_dir, "checkpoints", f"cls_step{step}.params")
        path_haz = os.path.join(out_dir, "checkpoints", f"haz_step{step}.params")
        with open(path_cls, "wb") as f:
            f.write(to_bytes(state.cls.params))
        with open(path_haz, "wb") as f:
            f.write(to_bytes(state.haz.params))
        print(f"[ckpt] saved: {path_cls}, {path_haz}")

    t0 = time.time()
    for step, batch_np in enumerate(train_iter, start=1):
        if step > int(cfg.training.max_steps):
            break

        batch = jnp.asarray(batch_np, dtype=jnp.float32)  # (B, 28*28)
        rng, sub = jax.random.split(rng)
        state, metrics = step_fn(sub, state, batch)

        if (step % log_every) == 0:
            m = {k: float(v) for k, v in metrics.items()}
            m["step"] = step
            m["lr_cls"] = float(cfg.optim.lr_cls)
            m["lr_haz"] = float(cfg.optim.lr_haz)
            dt = time.time() - t0
            m["sec_per_step"] = dt / log_every
            t0 = time.time()
            print(
                f"[step {step:6d}] total={m['loss_total']:.4f}  ce={m['loss_ce']:.4f}  dhm={m['loss_dhm']:.4f}  "
                f"acc@1={m['acc_top1_event']:.3f}  lam_mae={m['lam_mae']:.4f}  frac_event={m['frac_event']:.3f}  s/it={m['sec_per_step']:.3f}"
            )
            if wb_run is not None:
                import wandb
                wandb.log(m, step=step)
                
        if (wb_run is not None) and ((step % eval_every) == 0):
            log_eval_images(step, rng, state, anchors, beta, wb_run, np.asarray(batch_np))

        if (step % ckpt_every) == 0:
            save_ckpt(state, step)

    save_ckpt(state, step)
    print("[done] training finished.")

if __name__ == "__main__":
    main()