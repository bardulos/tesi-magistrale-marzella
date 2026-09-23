"""lib/dae/callback.py — early-stopping su AUC-PR con restore-best (repov6).

Sostituisce MccCallback di v5. A ogni epoca calcola l'Average Precision su val
(via ap_fn, che usa il modello corrente), traccia l'epoca di AUC-PR massima, fa
early-stop con patience e RIPRISTINA i pesi dell'epoca di AUC-PR massima (restore-best:
qualunque cosa accada dopo il picco, si tengono i pesi del picco -> patience alto
privo di rischio). Applica anche il filtro non-apprendimento ancorato alla prevalenza.

La metrica PER-SEME del trial = best_ap (= AUC-PR al restore-best). Niente top-3.
"""
from __future__ import annotations

import math

import tensorflow as tf

from lib.dae.objective import is_non_learning


class PrAucCallback(tf.keras.callbacks.Callback):
    def __init__(self, ap_fn, patience: int, prevalence: float, min_epochs: int,
                 margin: float, _test_inject: bool = False):
        super().__init__()
        self.ap_fn = ap_fn                  # callable() -> float: AUC-PR su val dal modello corrente
        self.patience = int(patience)
        self.prevalence = float(prevalence)
        self.min_epochs = int(min_epochs)
        self.margin = float(margin)
        self._test_inject = bool(_test_inject)
        self._inject_ap = None              # usato solo dai test (al posto di ap_fn)
        self.history_ap: list[float] = []
        self.best_ap: float = -math.inf
        self.best_epoch: int = -1
        self.best_weights = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.filter_reason = None

    def _current_ap(self):
        return self._inject_ap if self._test_inject else self.ap_fn()

    def on_epoch_end(self, epoch, logs=None):
        ap = self._current_ap()
        self.history_ap.append(ap)

        if ap is not None and math.isfinite(float(ap)) and float(ap) > self.best_ap:
            self.best_ap = float(ap)
            self.best_epoch = int(epoch)
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1

        if is_non_learning(self.history_ap, self.prevalence, self.min_epochs, self.margin):
            self.filter_reason = "non_learning"
            self.stopped_epoch = int(epoch)
            self.model.stop_training = True
            return

        if self.wait >= self.patience:
            self.stopped_epoch = int(epoch)
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
