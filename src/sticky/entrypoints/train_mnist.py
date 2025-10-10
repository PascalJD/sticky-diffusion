# src/sticky/entrypoints/train_mnist.py
from __future__ import annotations
import os, math
import hydra
from omegaconf import DictConfig, OmegaConf
import jax, jax.numpy as jnp
from flax.training import train_state as flax_state
import optax
import wandb

from sticky.core.sde_vp import make_beta_linear
from sticky.core.hazard import make_lambda
from sticky.core.precondition import Affine, joint_mixture_moments, sym_whitener
from sticky.core.anchors import PoinAnchors, CoordAnchors
from sticky.core.eventgen import build_events_coord

from sticky.data.mnist_discrete import load_mnist_split, make_bins, dataset_stats

from sticky.models.score_nets import MLPScore1D, SmallConvScore
from sticky.models.classifier_heads import CoordClassifier
from sticky.models.intensity_head import IntensityNet
from sticky.trainers.train_loop import TrainState, make_optimizer, train_step_coord

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project=cfg.logging.project, 
        name=cfg.logging.name, 
        mode=cfg.logging.mode
    )

    key = jax.random.PRNGKey(cfg.seed)
    T = cfg.sde.T
    beta_fn = make_beta_linear(cfg.sde.beta_min, cfg.sde.beta_max, T)
    c = cfg.hazard.c
    p_survive = float(survival_prob(beta_fn, c, T))
    print(f"[sticky] P(no unstick by T) = {p_survive:.3e} (target: <<1)")
    if p_survive > 1e-3:
        print("[sticky][warn] Survival prob > 1e-3; consider increasing hazard c or T.")

    # vectorized MNIST to (B,784) in [0,1]
    train_iter = load_mnist_split(
        'train', cfg.train.batch_size, shuffle=True, seed=cfg.seed
    )
    # estimate stats for preconditioning & CoordAnchors mean fill
    mu_d, Sig_d = dataset_stats(
        load_mnist_split('train', 512, shuffle=True, seed=cfg.seed)
    )
    d = mu_d.shape[0]
    bins = make_bins(cfg.dataset.L_bins)
    # anchors: per-pixel bins (factorized)
    log_m0 = jnp.full((d, bins.shape[0]), -jnp.log(bins.shape[0]))  # uniform per-pixel
    # affine preconditioning (joint with w)
    mu_A = mu_d  # simple choice; or compute anchor moments explicitly
    Sig_A = jnp.eye(d)
    mu, Sig = joint_mixture_moments(
        mu_d, Sig_d, mu_A, Sig_A, cfg.precond.omega
    )
    P = sym_whitener(Sig, eps=1e-4)
    aff = Affine(P, mu)

    anchors = CoordAnchors(
        d=d, 
        bins=bins, 
        log_m0=log_m0, 
        sigma_R=cfg.anchors.sigma_R, 
        mu_rest=aff.fwd(mu_d)
    )

    # models
    if cfg.model.score_arch == "conv":
        score = SmallConvScore(channels=cfg.model.score_ch) 
    else:
        score = MLPScore1D(hidden=cfg.model.score_hidden)
    # For vectorized input, use MLP-based classifier
    classifier = CoordClassifier(
        d=d, L=bins.shape[0], channels=cfg.model.cls_ch, tdim=64
    )
    intensity  = IntensityNet(hidden=cfg.model.int_hidden)

    # init params
    rng1, rng2, rng3, key = jax.random.split(key, 4)
    bsize = cfg.train.batch_size
    dummy_x = jnp.zeros((bsize, d))
    dummy_t = jnp.zeros((bsize,))
    params_score = score.init(
        rng1, 
        dummy_x.reshape(
            bsize, 
            int(math.sqrt(d)), 
            int(math.sqrt(d)), 
            1
        ) if cfg.model.score_arch=="conv" else dummy_x, dummy_t, True
    )['params']
    params_cls = classifier.init(rng2, dummy_x, dummy_t, True)['params']
    params_int = intensity.init(rng3, dummy_x, dummy_t, True)['params']

    # offline events
    Mevt = cfg.train.M_events
    key, kev = jax.random.split(key)
    tau, y_evt, marks = build_events_coord(
        kev, d, bins.shape[0], log_m0, anchors, beta_fn, c, T, Mevt
    )

    step_fn = train_step_coord
    is_coord = True

    # Optimizer & TrainState
    opt = make_optimizer(cfg.train.lr)
    opt_state = opt.init((params_score, params_cls, params_int))
    state = TrainState(params_score, params_cls, params_int, opt, opt_state)

    # Training loop
    step = 0
    wall_t0 = time.time()
    for epoch in range(cfg.train.epochs):
        for batch in train_iter:
            x = jnp.array(batch)  # (B,784)
            x = aff.fwd(x)
            key, kt = jax.random.split(key)
            t = jax.random.uniform(
                kt, (x.shape[0],), minval=cfg.train.eps, maxval=T
            )

            # events minibatch
            idx = (step * cfg.train.batch_size) % y_evt.shape[0]
            ev_slice = slice(idx, idx+cfg.train.batch_size)
            event_y = y_evt[ev_slice]
            event_t = tau[ev_slice]
            marks_b = marks[ev_slice]

            state, loss, logs = step_fn(
                state, 
                key, 
                x, 
                t, 
                beta_fn,
                (lambda params, xx, tt, tr: score.apply(params, xx.reshape(xx.shape[0],28,28,1) if cfg.model.score_arch=="conv" else xx, tt, tr)),
                classifier.apply, 
                intensity.apply,
                event_y, 
                event_t, 
                marks_b,
                T, 
                cfg.loss.alpha_s, 
                cfg.loss.alpha_c, 
                cfg.loss.alpha_l
            )
                        if step % cfg.logging.log_every == 0:
                elapsed = time.time() - wall_t0
                sps = (step + 1) / max(elapsed, 1e-6)
                # Console
                print(
                    f"[epoch {epoch} | step {step}] "
                    f"loss={float(loss):.4f} "
                    f"dsm={float(logs.get('L_dsm', logs.get('loss_dsm', jnp.nan))):.4f} "
                    f"cls={float(logs.get('L_cls', logs.get('loss_cls', jnp.nan))):.4f} "
                    f"int={float(logs.get('L_int', logs.get('loss_int', jnp.nan))):.4f} "
                    f"| sps={sps:.1f}"
                )

                # W&B scalar logs
                wandb.log(
                    {
                        'loss': float(loss),
                        'loss_dsm': float(logs.get('L_dsm', logs.get('loss_dsm', jnp.nan))),
                        'loss_cls': float(logs.get('L_cls', logs.get('loss_cls', jnp.nan))),
                        'loss_int': float(logs.get('L_int', logs.get('loss_int', jnp.nan))),
                        'steps_per_sec': sps,
                        'epoch': epoch,
                        'step': step,
                    }
                )

            if step % cfg.logging.sample_every == 0:
                with jax.disable_jit():  # keep it simple; sampling batch is small
                    ns = int(cfg.logging.n_samples)
                    key, ksam = jax.random.split(key)
                    # sticky sampling in precond space
                    xw_samp, t_hist, v_hist = sticky_sample_coord(
                        ksam, ns, d, T, beta_fn,
                        # pass wrappers consistent with training call-sites
                        score_apply=lambda params, xx, tt, tr: score.apply(params, xx.reshape(xx.shape[0],28,28,1) if cfg.model.score_arch=="conv" else xx, tt, tr),
                        cls_apply=classifier.apply,
                        int_apply=intensity.apply,
                        params_score=state.params_score,
                        params_cls=state.params_cls,
                        params_int=state.params_int,
                        anchors=anchors,
                        steps=int(cfg.sampling.steps),
                        eps=float(cfg.sampling.eps),
                        track_index= (14 * 28 + 14),  # center pixel (zero-based)
                    )
                    # invert preconditioning and clamp to [0,1] for display
                    xr = jax.vmap(aff.inv)(xw_samp)
                    xr = jnp.clip(xr, 0.0, 1.0)
                    xr_img = np.asarray(xr.reshape(ns, 28, 28))

                    # also log a few *real* examples from current batch (before whitening)
                    xr_true = np.asarray(jnp.clip(aff.inv(x), 0.0, 1.0).reshape(x.shape[0], 28, 28))[:ns]

                    # W&B: image panels
                    wandb.log({
                        "samples": [wandb.Image(xr_img[i], caption=f"synth {i}") for i in range(min(ns, 16))],
                        "reals": [wandb.Image(xr_true[i], caption=f"real {i}") for i in range(min(ns, 16))],
                        "traj_center_pix": wandb.plot.line_series(
                            xs=list(np.asarray(t_hist))[::-1],
                            ys=[list(np.asarray(v_hist))[::-1]],
                            keys=["x[center]"],
                            title="Center-pixel trajectory (reverse time)",
                            xname="t"
                        ),
                        "step": step,
                        "epoch": epoch,
                    })

            step += 1

    wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main()