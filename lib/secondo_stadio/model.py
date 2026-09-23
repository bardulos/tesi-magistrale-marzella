"""lib/secondo_stadio/model.py — builder MLP unico del secondo stadio (ceiling e PA).

Funnel geometrico: n_layers hidden Dense-ReLU con larghezza max(1, int(h1*ratio**k)),
L2 su ogni kernel hidden, Dropout dopo ogni hidden (solo se dropout>0), uscita Dense(1, sigmoid)
con loss BCE non-from-logits a valle. TF importato lazy (vincolo di progetto).
Port unificato di repo_v5 pseudo_anomalie.build_pa_mlp:24-45 == repo_v3 mlp_utils.build_mlp:36-59.
"""
from __future__ import annotations

from lib.secondo_stadio import constants as C


def funnel_widths(h1: int, n_layers: int, ratio: float) -> list[int]:
    """Larghezze degli hidden layer (troncamento intero, minimo 1): contratto legacy."""
    return [max(1, int(int(h1) * (float(ratio) ** k))) for k in range(int(n_layers))]


def h1_from_config(config: dict, n_in: int = C.PHI_DIM) -> int:
    """Larghezza del PRIMO hidden dal config: `config['h1']` diretto (P1/ceiling e legacy) oppure
    derivato da `config['exp']` (PA, redesign 2026-07-03: exp = fattore di espansione, h1 =
    round(exp*N_in)). Un solo grado di liberta' espresso in due unita'."""
    if "h1" in config:
        return int(config["h1"])
    return int(round(float(config["exp"]) * int(n_in)))


def build_mlp(params: dict, input_dim: int = C.PHI_DIM):
    """MLP binario su phi. params: h1, n_layers, ratio, dropout, l2_reg."""
    import tensorflow as tf  # noqa: F401 (lazy, vincolo progetto)
    from tensorflow.keras import Sequential, layers, regularizers

    dropout = float(params["dropout"])
    l2_reg = float(params["l2_reg"])

    model = Sequential(name="s2_mlp")
    model.add(layers.Input(shape=(input_dim,)))
    for k, width in enumerate(funnel_widths(params["h1"], params["n_layers"], params["ratio"])):
        model.add(layers.Dense(width, activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               name=f"dense_{k + 1}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"drop_{k + 1}"))
    model.add(layers.Dense(1, activation="sigmoid", name="out"))
    return model


def count_mlp_params(h1: int, n_layers: int, ratio: float,
                     input_dim: int = C.PHI_DIM) -> int:
    """Numero di parametri in forma chiusa (colonna n_par delle tabelle di reporting):
    per ogni Dense, in*out pesi + out bias; uscita a 1 neurone."""
    n, prev = 0, int(input_dim)
    for width in funnel_widths(h1, n_layers, ratio):
        n += prev * width + width
        prev = width
    n += prev + 1
    return n
