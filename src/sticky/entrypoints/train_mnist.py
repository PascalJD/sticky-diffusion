# src/sticky/entrypoints/train_mnist.py
from __future__ import annotations

import os
import time
from functools import partial

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from sticky.entrypoints.eval_utils import log_mnist_eval_images
from sticky.entrypoints.runtime_utils import configure_runtime, maybe_init_wandb, save_flax_params


@hydra.main(config_path="../../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Must run BEFORE importing JAX
    configure_runtime(cfg)

    # Now import JAX/Flax/Optax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    # Core
    from sticky.core.anchors import AnalogBitsAnchors
    from sticky.core.hazard import make_hazard_early
    from sticky.core.sde_vp import make_beta

    # Model + training
    from sticky.models.sticky_head import StickyHead
    from sticky.trainers.train_step import ForwardProcess, StickyTrainState, train_step

    # Sampler (eval)
    from sticky.core.sampler import SamplerConfig

    # 1) Data iterator
    from sticky.data.mnist_discrete import load_mnist_split
    train_iter = load_mnist_split(
        split="train",
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        seed=cfg.seed,
        drop_remainder=cfg.dataset.drop_remainder,
        num_workers=int(cfg.training.get("num_workers", 0)),
    )

    print(f"[jax] backend={jax.default_backend()}  devices={jax.devices()}")

    # 2) Anchors & forward process
    anchors = AnalogBitsAnchors(
        L=int(cfg.dataset.L),
        d=None,
        gray=bool(cfg.dataset.gray_code),
    )
    L = anchors.L
    d = anchors.d
    print(f"[anchors] L={L}, d={d}, gray={anchors.gray}")

    beta = instantiate(cfg.forward.beta)
    hazard = instantiate(cfg.forward.hazard, beta=beta)
    jump = instantiate(cfg.forward.jump, beta=beta)
    fwd = ForwardProcess(beta=beta, hazard=hazard, jump=jump, T=float(cfg.forward.beta.T))

    # 3) Model + optimizer + state
    def make_optimizer() -> optax.GradientTransformation:
        lr = float(cfg.optim.get("lr", cfg.optim.lr_cls))
        return optax.chain(
            optax.clip_by_global_norm(float(cfg.optim.grad_clip)),
            optax.adamw(lr, weight_decay=float(cfg.optim.weight_decay)),
        )

    def create_train_state(rng: jax.random.PRNGKey) -> StickyTrainState:
        H = W = 28
        model = StickyHead(
            L=L,
            ch=cfg.model.allocator.ch,
            depth=cfg.model.allocator.depth,
            num_groups=cfg.model.allocator.num_groups,
            temb_dim=cfg.model.allocator.temb_dim,
            tfeat_dim=cfg.model.allocator.tfeat_dim,
            min_rho=cfg.model.intensity.min_rho,
            max_rho=cfg.model.intensity.max_rho,
            use_confidence=cfg.model.intensity.use_confidence,
        )
        y_dummy = jnp.zeros((1, H, W, d), dtype=jnp.float32)
        t_dummy = jnp.zeros((1,), dtype=jnp.float32)
        params = model.init(rng, y_dummy, t_dummy)
        tx = make_optimizer()
        return StickyTrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_rng = jax.random.PRNGKey(int(cfg.seed))
    eval_rng = jax.random.PRNGKey(int(cfg.seed) + 1)

    train_rng, key_init = jax.random.split(train_rng)
    state = create_train_state(key_init)

    # 4) JIT train step
    use_dhm = bool(cfg.training.get("use_dhm", False))
    ce_w = float(cfg.training.ce_weight)
    dhm_w = float(cfg.training.dhm_weight) if use_dhm else 0.0

    step_fn = jax.jit(
        partial(
            train_step,
            anchors=anchors,
            fwd=fwd,
            L=L,
            ce_weight=ce_w,
            dhm_weight=dhm_w,
            use_dhm=use_dhm,
        )
    )


    # 5) Sampler config
    sampler_kwargs = OmegaConf.to_container(cfg.sampler, resolve=True) if "sampler" in cfg else {}
    sampler_kwargs = dict(sampler_kwargs)  # ensure mutable
    sampler_cfg = SamplerConfig(**sampler_kwargs)

    # 6) Wandb + loop
    wb_run = maybe_init_wandb(cfg)
    log_every = int(cfg.training.log_every)
    eval_every = int(cfg.training.get("eval_every", 500))
    num_samples = int(cfg.training.get("num_samples", 36))
    grid_cols = int(cfg.training.get("grid_cols", 6))
    should_save_ckpt = bool(cfg.training.should_save_ckpt)
    ckpt_every = int(cfg.training.ckpt_every)

    out_dir = os.getcwd()  # hydra run dir

    t0 = time.time()
    for step, batch_np in enumerate(train_iter, start=1):
        if step > int(cfg.training.max_steps):
            break

        batch = jnp.asarray(batch_np, dtype=jnp.float32)  # (B, 28*28)

        train_rng, sub = jax.random.split(train_rng)
        state, metrics = step_fn(sub, state, batch)

        if (step % log_every) == 0:
            m = {k: float(v) for k, v in metrics.items()}
            dt = time.time() - t0
            m["time/sec_per_step"] = dt / log_every
            t0 = time.time()
            m["optim/step"] = step
            m["optim/lr"] = float(cfg.optim.get("lr", cfg.optim.lr_cls))

            # Make printing robust if a metric is missing
            total = m.get("loss/total", float("nan"))
            ce = m.get("loss/CE", float("nan"))
            dhm = m.get("loss/DHM", 0.0)

            if use_dhm:
                print(
                    f"[step {step:6d}] loss={total:.4f}  CE={ce:.4f}  DHM={dhm:.4f}  "
                    f"acc@1={m.get('CE/acc_top1_event', float('nan')):.3f}  "
                    f"lam_mae={m.get('DHM/lam_mae', float('nan')):.4f}  "
                    f"clip={m.get('DHM/log_ratio_clip_frac', float('nan')):.3f}  "
                    f"rho_logstd={m.get('DHM/rho_hat_log_std_spatial', float('nan')):.3f}  "
                    f"rho_PRED_logstd={m.get('DHM/rho_pred_log_std_spatial', float('nan')):.3f}  "
                    f"s/it={m['time/sec_per_step']:.3f}"
                )
            else:
                print(
                    f"[step {step:6d}] loss={total:.4f}  CE={ce:.4f}  "
                    f"acc@1={m.get('CE/acc_top1_event', float('nan')):.3f}  "
                    f"s/it={m['time/sec_per_step']:.3f}"
                )

            if wb_run is not None:
                import wandb
                wandb.log(m, step=step)

        if (wb_run is not None) and ((step % eval_every) == 0):
            eval_rng, sub = jax.random.split(eval_rng)
            log_mnist_eval_images(
                step=step,
                key=sub,
                state=state,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                sampler_cfg=sampler_cfg,
                num_samples=num_samples,
                grid_cols=grid_cols,
                batch_np=np.asarray(batch_np),
                wb_run=wb_run,
                shape_hw=(28, 28),
            )

        if should_save_ckpt and (step % ckpt_every) == 0:
            path = save_flax_params(state.params, out_dir, step)
            print(f"[ckpt] saved: {path}")

    # final ckpt
    if should_save_ckpt:
        path = save_flax_params(state.params, out_dir, step)
        print(f"[ckpt] saved: {path}")
        print("[done] training finished.")


if __name__ == "__main__":
    main()
