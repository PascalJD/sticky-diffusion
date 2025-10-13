from __future__ import annotations
import os, math, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # friendlier GPU memory behavior
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hydra
from omegaconf import DictConfig, OmegaConf
import jax, jax.numpy as jnp
import numpy as np
import optax
import wandb


from sticky.core.sde_vp import make_beta_linear
from sticky.core.hazard import survival_prob
from sticky.core.precondition import Affine, joint_mixture_moments, sym_whitener
from sticky.core.anchors import CoordAnchors
from sticky.core.eventgen import sample_events_coord_jit
from sticky.core.sampler import sticky_sample_coord

from sticky.data.mnist_discrete import load_mnist_split, make_bins, dataset_stats

from sticky.models.score_nets import MLPScore1D, SmallConvScore
from sticky.models.classifier_heads import CoordClassifier
from sticky.models.intensity_head import IntensityNet
from sticky.trainers.train_loop import TrainState, make_optimizer, train_step_coord

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project=cfg.experiment.logging.project, 
        name=cfg.experiment.logging.name, 
    )

    key = jax.random.PRNGKey(cfg.seed)
    T = float(cfg.sde.T)
    beta_fn = make_beta_linear(cfg.sde.beta_min, cfg.sde.beta_max, T)
    c = float(cfg.hazard.c)
    p_survive = float(survival_prob(beta_fn, c, T))
    print(f"[sticky] P(no unstick by T) = {p_survive:.3e} (target: ≪ 1)")
    if p_survive > 1e-3:
        print("[sticky][warn] Survival prob > 1e-3; consider increasing hazard c and/or T).")

    # Data: vectorized MNIST (B, 784) in [0,1]
    train_iter = load_mnist_split(
        'train', cfg.experiment.train.batch_size, shuffle=True, seed=cfg.seed
    )
    # Stats for preconditioning & CoordAnchors mean fill
    mu_d, Sig_d = dataset_stats(
        load_mnist_split('train', 512, shuffle=True, seed=cfg.seed)
    )
    d = int(mu_d.shape[0])
    bins = make_bins(cfg.dataset.L_bins)
    log_m0 = jnp.full((d, bins.shape[0]), -jnp.log(bins.shape[0]))

    # Preconditioning
    mu_A = mu_d
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

    # Models
    score = SmallConvScore(
        channels=cfg.model.score_ch
        ) if cfg.model.score_arch == "conv" \
            else MLPScore1D(hidden=cfg.model.score_hidden)
    classifier = CoordClassifier(
        d=d, L=bins.shape[0], channels=cfg.model.cls_ch, tdim=64
    )
    intensity = IntensityNet(hidden=cfg.model.int_hidden)

    # Init params
    rng1, rng2, rng3, key = jax.random.split(key, 4)
    bsize = cfg.experiment.train.batch_size
    dummy_x = jnp.zeros((bsize, d))
    dummy_t = jnp.zeros((bsize,))
    params_score = score.init(
        rng1,
        dummy_x.reshape(bsize, 28, 28, 1) if cfg.model.score_arch=="conv" else dummy_x,
        dummy_t,
        True
    )['params']
    params_cls = classifier.init(rng2, dummy_x, dummy_t, True)['params']
    params_int = intensity.init(rng3, dummy_x, dummy_t, True)['params']

    def score_apply_vec(variables, xx, tt, train: bool):
        if cfg.model.score_arch == "conv":
            out = score.apply(
                variables,
                xx.reshape(xx.shape[0], 28, 28, 1),
                tt, train
            )
            out = out.reshape(xx.shape[0], -1)
        else:
            out = score.apply(variables, xx, tt, train)
            if out.ndim != 2:
                out = out.reshape(out.shape[0], -1)
        return out
    # Optimizer & TrainState
    opt = make_optimizer(cfg.experiment.train.lr)
    opt_state = opt.init((params_score, params_cls, params_int))
    state = TrainState(params_score, params_cls, params_int, opt, opt_state)

    # Training loop
    step = 0
    wall_t0 = time.time()
    for epoch in range(cfg.experiment.train.epochs):
        for batch in train_iter:
            x = jnp.array(batch)       # (B, 784) in [0,1]
            x = aff.fwd(x)
            key, kt, kev = jax.random.split(key, 3)
            t = jax.random.uniform(
                kt, (x.shape[0],), minval=cfg.experiment.train.eps, maxval=T
            )

            # online reversed events: one per data sample in this minibatch
            tau, y_evt, marks = sample_events_coord_jit(
                kev, M=int(x.shape[0]),
                log_m0=log_m0, 
                bins=bins, 
                mu_rest=anchors.mu_rest, 
                sigma_R=anchors.sigma_R,
                beta_fn=beta_fn, c=c, T=T
            )

            # one step
            state, loss, logs = train_step_coord(
                state,
                key,
                x, 
                t, 
                beta_fn,
                score_apply_vec,
                classifier.apply, 
                intensity.apply,
                y_evt, 
                tau, 
                marks,
                T, 
                cfg.loss.alpha_s, 
                cfg.loss.alpha_c, 
                cfg.loss.alpha_l
            )

            if step % cfg.experiment.logging.log_every == 0:
                elapsed = time.time() - wall_t0
                sps = (step + 1) / max(elapsed, 1e-6)
                print(
                    f"[epoch {epoch} | step {step}] "
                    f"loss={float(loss):.4f} "
                    f"dsm={float(logs.get('L_dsm', logs.get('loss_dsm', jnp.nan))):.4f} "
                    f"cls={float(logs.get('L_cls', logs.get('loss_cls', jnp.nan))):.4f} "
                    f"int={float(logs.get('L_int', logs.get('loss_int', jnp.nan))):.4f} "
                    f"| sps={sps:.1f}"
                )
                wandb.log({
                    'loss': float(loss),
                    'loss_dsm': float(logs.get('L_dsm', logs.get('loss_dsm', jnp.nan))),
                    'loss_cls': float(logs.get('L_cls', logs.get('loss_cls', jnp.nan))),
                    'loss_int': float(logs.get('L_int', logs.get('loss_int', jnp.nan))),
                    'steps_per_sec': sps,
                    'epoch': epoch, 'step': step,
                })

            # periodic sampling for logging (small batch, no JIT pressure)
            if step % cfg.experiment.logging.sample_every == 0:
                ns = int(cfg.experiment.logging.n_samples)
                key, ksam = jax.random.split(key)
                xw_samp, t_hist, v_hist = sticky_sample_coord(
                    ksam,
                    ns,
                    d,
                    T,
                    beta_fn,
                    score_apply=score_apply_vec,
                    cls_apply=classifier.apply,
                    int_apply=intensity.apply,
                    params_score=state.params_score,
                    params_cls=state.params_cls,
                    params_int=state.params_int,
                    anchors=anchors,
                    steps=int(cfg.experiment.sampling.steps),
                    eps=float(cfg.experiment.sampling.eps),
                    track_index=(14*28+14),
                )
                xr = jax.vmap(aff.inv)(xw_samp)
                xr = jnp.clip(xr, 0.0, 1.0)
                xr_img = np.asarray(xr.reshape(ns, 28, 28))
                xr_true = np.asarray(jnp.clip(aff.inv(x), 0.0, 1.0).reshape(x.shape[0], 28, 28))[:ns]
                wandb.log({
                    "samples": [wandb.Image(xr_img[i], caption=f"synth {i}") for i in range(min(ns, 16))],
                    "reals": [wandb.Image(xr_true[i], caption=f"real {i}")  for i in range(min(ns, 16))],
                    "traj_center_pix": wandb.plot.line_series(
                        xs=list(np.asarray(t_hist))[::-1],
                        ys=[list(np.asarray(v_hist))[::-1]],
                        keys=["x[center]"], title="Center-pixel trajectory (reverse time)", xname="t"),
                    "step": step, "epoch": epoch,
                })
            step += 1

    wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main()