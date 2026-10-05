"""Builder MLP del secondo stadio con funnel geometrico e uscita sigmoid."""
from __future__ import annotations

from lib.secondo_stadio import constants as C


def funnel_widths(h1: int, n_layers: int, ratio: float) -> list[int]:
    """Larghezze degli hidden layer (troncamento intero, minimo 1): contratto legacy."""
    return [max(1, int(int(h1) * (float(ratio) ** k))) for k in range(int(n_layers))]


def h1_from_config(config: dict, n_in: int = C.PHI_DIM) -> int:
    """Legge h1 dal config o lo deriva dal fattore di espansione `exp`."""
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
    """Numero di parametri Dense, inclusi i bias e il neurone di uscita."""
    n, prev = 0, int(input_dim)
    for width in funnel_widths(h1, n_layers, ratio):
        n += prev * width + width
        prev = width
    n += prev + 1
    return n
