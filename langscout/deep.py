"""Deep-learning benchmarks (optional - needs TensorFlow, not needed to run the app).

Two models are provided:
* ``WordSimpleRNN`` - the architecture of the original project (word tokens -> SimpleRNN).
* ``CharCNN``       - a character-level convolutional network, the deep-learning counterpart of
                      the character n-gram idea that wins on this task.
Both expose ``fit`` / ``predict_proba`` / ``classes_`` like scikit-learn estimators.
"""
from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .config import SEED
from .preprocess import clean_text

_PUNCT = ".,;:!?\"'()[]{}«»“”‘’…—–-\u0964\u0965\u061f\u060c"


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


class _KerasTextModel:
    name = "keras"

    def __init__(self, epochs: int = 20, batch_size: int = 64, patience: int = 3):
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.encoder = LabelEncoder()
        self.history = None

    # -- subclass hooks ---------------------------------------------------------------
    def _fit_vocab(self, texts): raise NotImplementedError
    def _encode(self, texts): raise NotImplementedError
    def _build(self, n_classes): raise NotImplementedError

    # -- public API -------------------------------------------------------------------
    @property
    def classes_(self):
        return self.encoder.classes_

    def fit(self, texts, labels, val_texts, val_labels):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(SEED)
        y = self.encoder.fit_transform(labels)
        yv = self.encoder.transform(val_labels)
        self._fit_vocab(texts)
        self.model = self._build(len(self.classes_))
        weights = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=y)
        self.history = self.model.fit(
            self._encode(texts), y,
            validation_data=(self._encode(val_texts), yv),    # VALIDATION set (never the test set)
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            class_weight=dict(enumerate(weights)),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                        restore_best_weights=True)],
        )
        return self

    def predict_proba(self, texts):
        return self.model.predict(self._encode(list(texts)), verbose=0, batch_size=256)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(1)]

    @property
    def epochs_trained(self) -> int:
        return len(self.history.history["loss"]) if self.history else 0


class CharCNN(_KerasTextModel):
    name = "Char-level CNN (Keras)"

    def __init__(self, maxlen: int = 200, max_chars: int = 600, **kw):
        super().__init__(**kw)
        self.maxlen, self.max_chars = maxlen, max_chars

    def _fit_vocab(self, texts):
        counts = Counter(ch for t in texts for ch in clean_text(t).lower())
        self.vocab = {ch: i + 2 for i, (ch, _) in enumerate(counts.most_common(self.max_chars))}  # 0=pad 1=unk

    def _encode(self, texts):
        X = np.zeros((len(texts), self.maxlen), dtype="int32")
        for i, t in enumerate(texts):
            ids = [self.vocab.get(ch, 1) for ch in clean_text(t).lower()[: self.maxlen]]
            X[i, : len(ids)] = ids
        return X

    def _build(self, n_classes):
        from tensorflow.keras import Input, Model, layers
        inp = Input(shape=(self.maxlen,))
        x = layers.Embedding(len(self.vocab) + 2, 64)(inp)
        pooled = [layers.GlobalMaxPooling1D()(layers.Conv1D(128, k, activation="relu")(x)) for k in (2, 3, 5)]
        x = layers.Dropout(0.3)(layers.Concatenate()(pooled))
        out = layers.Dense(n_classes, activation="softmax")(x)
        model = Model(inp, out)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class WordSimpleRNN(_KerasTextModel):
    name = "Word-level SimpleRNN (original design)"

    def __init__(self, maxlen: int = 50, num_words: int = 15000, **kw):
        super().__init__(**kw)
        self.maxlen, self.num_words = maxlen, num_words

    @staticmethod
    def _tokens(text):
        return [w for w in (tok.strip(_PUNCT) for tok in clean_text(text).lower().split()) if w]

    def _fit_vocab(self, texts):
        counts = Counter(w for t in texts for w in self._tokens(t))
        self.vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(self.num_words - 2))}

    def _encode(self, texts):
        X = np.zeros((len(texts), self.maxlen), dtype="int32")
        for i, t in enumerate(texts):
            ids = [self.vocab.get(w, 1) for w in self._tokens(t)][-self.maxlen:]
            if ids:
                X[i, -len(ids):] = ids          # pre-padding: the RNN's last step sees a real token
        return X

    def _build(self, n_classes):
        from tensorflow.keras import Input, Model, layers
        inp = Input(shape=(self.maxlen,))
        x = layers.Embedding(self.num_words, 64)(inp)
        x = layers.Dropout(0.3)(layers.SimpleRNN(64)(x))
        out = layers.Dense(n_classes, activation="softmax")(x)
        model = Model(inp, out)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
