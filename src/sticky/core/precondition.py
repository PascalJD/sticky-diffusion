# src/sticky/core/precondition.py
from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple

Array = jnp.ndarray

def sym_whitener(cov: Array, eps: float=1e-5) -> Array:
    w, v = jnp.linalg.eigh(cov + eps * jnp.eye(cov.shape[-1]))
    return (v / jnp.sqrt(jnp.clip(w, 1e-10))) @ v.T

def joint_mixture_moments(
    mu_d: Array, Sig_d: Array, mu_A: Array, Sig_A: Array, omega: float
) -> Tuple[Array, Array]:
    mu = (1-omega)*mu_d + omega*mu_A
    delta = (mu_d - mu_A).reshape(-1,1)
    Sig = (1-omega)*Sig_d + omega*Sig_A + (1-omega)*omega*(delta @ delta.T)
    return mu, Sig

class Affine:
    def __init__(self, P: Array, mu: Array):
        self.P = P
        self.mu = mu
        self.Pinv = jnp.linalg.inv(P)
    def fwd(self, x: Array) -> Array:
        return (x - self.mu) @ self.P.T
    def inv(self, xw: Array) -> Array:
        return xw @ self.Pinv.T + self.mu