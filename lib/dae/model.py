"""lib/dae/model.py — architettura DAE Keras con loss composita (repov6, riuso da v5).

Architettura simmetrica a 5 layer:
  n_features -> exp -> mid -> btl -> mid -> exp -> n_features

Output split: primi n_continuous lineari (target N(0,1) post-scaler); ultimi n_binary
sigmoid (target {0,1}). Corruzione (rumore gaussiano) attiva solo a training=True.

Loss (TRAINING): (1/n_cont)·sum MSE(continue) + (1/n_bin)·sum BCE(binarie).
La BCE e' SOLO loss di training: il punteggio di anomalia (lib.dae.evaluate) e' MSE
sulle sole binarie, senza BCE (lib.utils.SCORE_DEFINITION).
"""
from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Concatenate, Dense, GaussianNoise, Input

log = logging.getLogger(__name__)


def build_dae(
    n_features: int,
    btl: int,
    exp: int,
    n_continuous: int,
    n_binary: int,
    continuous_idx: list[int],
    binary_idx: list[int],
    l2_reg: float = 1e-6,
    noise_std: float = 0.10,
    noise_type: str = "gaussian",
    loss_continuous: str = "mse",
    *,
    mid: int,
) -> tuple[Model, Model]:
    """Costruisce il modello DAE Keras (non compilato) e il sotto-modello encoder.
    Ritorna (model, encoder) NON compilati."""
    if n_continuous + n_binary != n_features:
        raise ValueError(
            f"n_continuous + n_binary ({n_continuous}+{n_binary}) != n_features ({n_features})")
    if len(continuous_idx) != n_continuous or len(binary_idx) != n_binary:
        raise ValueError("indici incoerenti con n_continuous/n_binary")
    if noise_type != "gaussian":
        raise ValueError(f"noise_type ignoto: {noise_type}")
    if loss_continuous != "mse":
        raise ValueError(f"loss_continuous ignota: {loss_continuous}")
    if mid <= 0:
        raise ValueError(f"mid deve essere > 0: {mid}")

    inputs = Input(shape=(n_features,), name="input")
    x = GaussianNoise(noise_std, name="corruption")(inputs)

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Encoder: input -> exp -> mid -> btl
    x = Dense(exp, activation="relu", kernel_regularizer=reg, name="enc_exp")(x)
    x = Dense(mid, activation="relu", kernel_regularizer=reg, name="enc_mid")(x)
    btl_out = Dense(btl, activation="relu", kernel_regularizer=reg, name="bottleneck")(x)

    # Decoder: btl -> mid -> exp -> output split
    x = btl_out
    x = Dense(mid, activation="relu", kernel_regularizer=reg, name="dec_mid")(x)
    x = Dense(exp, activation="relu", kernel_regularizer=reg, name="dec_exp")(x)

    cont_out = Dense(n_continuous, activation="linear", name="out_continuous")(x)
    bin_out = Dense(n_binary, activation="sigmoid", name="out_binary")(x)
    output = Concatenate(name="output")([cont_out, bin_out])

    model = Model(inputs=inputs, outputs=output, name="dae")
    encoder = Model(inputs=inputs, outputs=btl_out, name="encoder")

    arch = f"input -> exp({exp}) -> mid({mid}) -> btl({btl}) -> mid({mid}) -> exp({exp})"
    log.info("build_dae: %s | n_cont=%d n_bin=%d noise=%s sigma=%.3f loss_cont=%s l2=%g",
             arch, n_continuous, n_binary, noise_type, noise_std, loss_continuous, l2_reg)
    return model, encoder


def make_dae_loss(continuous_idx: list[int], binary_idx: list[int],
                  loss_continuous: str = "mse"):
    """Loss composita MSE (continue) + BCE (binarie). Ritorna callable
    (y_true, y_pred) -> loss per-batch shape (batch,)."""
    if loss_continuous != "mse":
        raise ValueError(f"loss_continuous ignota: {loss_continuous}")
    cont_idx_tf = tf.constant(continuous_idx, dtype=tf.int32)
    bin_idx_tf = tf.constant(binary_idx, dtype=tf.int32)
    n_cont = float(len(continuous_idx))
    n_bin = float(len(binary_idx))
    eps = 1e-7

    def dae_loss(y_true, y_pred):
        y_true_c = tf.gather(y_true, cont_idx_tf, axis=-1)
        y_pred_c = tf.gather(y_pred, cont_idx_tf, axis=-1)
        y_true_b = tf.gather(y_true, bin_idx_tf, axis=-1)
        y_pred_b = tf.gather(y_pred, bin_idx_tf, axis=-1)
        cont_term = tf.reduce_sum(tf.square(y_true_c - y_pred_c), axis=-1) / n_cont
        y_pred_b = tf.clip_by_value(y_pred_b, eps, 1.0 - eps)
        bce = -(y_true_b * tf.math.log(y_pred_b) + (1.0 - y_true_b) * tf.math.log(1.0 - y_pred_b))
        bin_term = tf.reduce_sum(bce, axis=-1) / n_bin
        return cont_term + bin_term

    dae_loss.__name__ = f"dae_loss_{loss_continuous}_bce"
    return dae_loss
