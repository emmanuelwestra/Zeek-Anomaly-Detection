from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.svm import OneClassSVM
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def matrix(value: np.ndarray, *, width: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("Model input must be a two-dimensional matrix")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"Expected {width} features, found {result.shape[1]}")
    if not np.isfinite(result).all():
        raise ValueError("Model input must contain only finite values")
    return result


@dataclass
class OCSVM:
    kernel: str = "rbf"
    gamma: str | float = 0.02
    nu: float = 0.001
    cache_size: float = 200
    tolerance: float = 0.001
    shrinking: bool = False
    scoring_threads: int = 1
    random_state: int = 42
    model_: OneClassSVM | None = None

    def fit(self, x: np.ndarray) -> OCSVM:
        values = matrix(x)
        if not len(values):
            raise ValueError("Cannot fit OC-SVM on no rows")
        self.model_ = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
            cache_size=self.cache_size,
            tol=self.tolerance,
            shrinking=self.shrinking,
        ).fit(values)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("OC-SVM must be fitted before scoring")
        values = matrix(x, width=self.model_.n_features_in_)
        if not len(values):
            return np.empty(0)
        workers = min(max(1, int(self.scoring_threads)), len(values))
        if workers == 1:
            return -self.model_.score_samples(values)
        chunks = [chunk for chunk in np.array_split(values, workers) if len(chunk)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return -np.concatenate(list(pool.map(self.model_.score_samples, chunks)))

    @property
    def support_vectors(self) -> int:
        return 0 if self.model_ is None else int(len(self.model_.support_))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> OCSVM:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value


def _activation(name: str) -> type[nn.Module]:
    choices = {"relu": nn.ReLU, "gelu": nn.GELU}
    try:
        return choices[name.casefold()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


def _block(
    input_dim: int,
    output_dim: int,
    activation: type[nn.Module],
    dropout: float,
    normalization: str,
) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
    if normalization == "layernorm":
        layers.append(nn.LayerNorm(output_dim))
    elif normalization not in {"none", ""}:
        raise ValueError(f"Unsupported normalization: {normalization}")
    layers.append(activation())
    if dropout:
        layers.append(nn.Dropout(dropout))
    return layers


class _Network(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        hidden_dims: list[int],
        latent_dim: int,
        activation: str,
        dropout: float,
        normalization: str,
    ) -> None:
        super().__init__()
        activation_class = _activation(activation)
        encoder_dims = [int(schema["input_width"]), *hidden_dims, latent_dim]
        encoder: list[nn.Module] = []
        for input_dim, output_dim in zip(encoder_dims, encoder_dims[1:], strict=False):
            encoder.extend(_block(input_dim, output_dim, activation_class, dropout, normalization))
        self.encoder = nn.Sequential(*encoder)
        decoder_dims = [latent_dim, *reversed(hidden_dims)]
        decoder: list[nn.Module] = []
        for input_dim, output_dim in zip(decoder_dims, decoder_dims[1:], strict=False):
            decoder.extend(_block(input_dim, output_dim, activation_class, dropout, normalization))
        self.decoder = nn.Sequential(*decoder)
        final_width = decoder_dims[-1]
        self.numeric_head = nn.Linear(final_width, int(schema["numeric_count"]))
        self.categorical_heads = nn.ModuleList(
            nn.Linear(final_width, int(group["stop"]) - int(group["start"]))
            for group in schema["categorical_groups"]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        decoded = self.decoder(self.encoder(x))
        return self.numeric_head(decoded), [head(decoded) for head in self.categorical_heads]


@dataclass
class MixedAutoencoder:
    reconstruction_schema: dict[str, Any]
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 32
    activation: str = "relu"
    normalization: str = "layernorm"
    dropout: float = 0.0
    categorical_loss: str = "cross_entropy"
    training_numeric_weight: float = 0.5
    scoring_numeric_weight: float = 0.1
    epochs: int = 80
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 12
    internal_validation_fraction: float = 0.05
    random_state: int = 42
    device: str = "auto"
    model_: _Network | None = None
    history_: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("training_numeric_weight", "scoring_numeric_weight"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.categorical_loss != "cross_entropy":
            raise ValueError("This project supports only grouped categorical cross-entropy")
        if not 0 <= self.internal_validation_fraction < 1:
            raise ValueError("internal_validation_fraction must be in [0, 1)")
        self._validate_schema()

    def fit(self, x: np.ndarray) -> MixedAutoencoder:
        values = matrix(x, width=int(self.reconstruction_schema["input_width"]))
        if not len(values):
            raise ValueError("Cannot fit autoencoder on no rows")
        self._validate_one_hot(values)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model_ = self._build().to(self._device())
        order = np.random.default_rng(self.random_state).permutation(len(values))
        validation_rows = int(len(values) * self.internal_validation_fraction)
        validation = torch.from_numpy(values[order[:validation_rows]])
        train = torch.from_numpy(values[order[validation_rows:]])
        loader = DataLoader(
            TensorDataset(train),
            batch_size=min(self.batch_size, len(train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            train_losses = self._train_epoch(loader, optimizer)
            validation_losses = self._losses(validation) if len(validation) else train_losses
            self.history_.append({
                "epoch": float(epoch),
                "train_loss": train_losses[0],
                "train_numeric_loss": train_losses[1],
                "train_categorical_loss": train_losses[2],
                "validation_loss": validation_losses[0],
                "validation_numeric_loss": validation_losses[1],
                "validation_categorical_loss": validation_losses[2],
            })
            if validation_losses[0] < best_loss:
                best_loss = validation_losses[0]
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to(self._device()).eval()
        return self

    def components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Autoencoder must be fitted before scoring")
        values = matrix(x, width=int(self.reconstruction_schema["input_width"]))
        self._validate_one_hot(values)
        if not len(values):
            empty = np.empty(0, dtype=np.float32)
            return {"numeric": empty, "categorical": empty, "total": empty}
        tensors = torch.from_numpy(values)
        numeric_parts: list[np.ndarray] = []
        categorical_parts: list[np.ndarray] = []
        for start in range(0, len(tensors), self.batch_size):
            _, numeric, categorical = self._component_tensors(tensors[start : start + self.batch_size].to(self._device()))
            numeric_parts.append(numeric.detach().cpu().numpy())
            categorical_parts.append(categorical.detach().cpu().numpy())
        numeric = np.concatenate(numeric_parts).astype(np.float32)
        categorical = np.concatenate(categorical_parts).astype(np.float32)
        total = combine_ae_components(numeric, categorical, self.scoring_numeric_weight)
        return {"numeric": numeric, "categorical": categorical, "total": total}

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return self.components(x)["total"]

    def save(self, path: str | Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Autoencoder must be fitted before saving")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 1,
            "schema": self.reconstruction_schema,
            "config": self._config(),
            "state_dict": self.model_.cpu().state_dict(),
            "history": self.history_,
        }, output)
        self.model_.to(self._device())

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> MixedAutoencoder:
        artifact = torch.load(path, map_location="cpu")
        if artifact.get("version") != 1:
            raise ValueError("Unsupported autoencoder artifact version")
        config = dict(artifact["config"])
        if device is not None:
            config["device"] = device
        value = cls(reconstruction_schema=artifact["schema"], **config)
        value.model_ = value._build()
        value.model_.load_state_dict(artifact["state_dict"])
        value.model_.to(value._device()).eval()
        value.history_ = list(artifact.get("history", []))
        return value

    def _build(self) -> _Network:
        return _Network(
            self.reconstruction_schema,
            self.hidden_dims,
            self.latent_dim,
            self.activation,
            self.dropout,
            self.normalization,
        )

    def _component_tensors(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model_ is None:
            raise RuntimeError("Autoencoder is not initialized")
        numeric_prediction, category_predictions = self.model_(batch)
        count = int(self.reconstruction_schema["numeric_count"])
        numeric = torch.mean((batch[:, :count] - numeric_prediction) ** 2, dim=1)
        losses = []
        for group, logits in zip(self.reconstruction_schema["categorical_groups"], category_predictions, strict=True):
            target = batch[:, int(group["start"]) : int(group["stop"])].argmax(dim=1)
            losses.append(F.cross_entropy(logits, target, reduction="none"))
        categorical = torch.stack(losses, dim=1).mean(dim=1)
        total = self.training_numeric_weight * numeric + (1 - self.training_numeric_weight) * categorical
        return total, numeric, categorical

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> tuple[float, float, float]:
        assert self.model_ is not None
        self.model_.train()
        totals = np.zeros(3)
        rows = 0
        for (batch,) in loader:
            batch = batch.to(self._device())
            optimizer.zero_grad(set_to_none=True)
            losses = self._component_tensors(batch)
            losses[0].mean().backward()
            optimizer.step()
            totals += np.array([float(loss.mean().detach().cpu()) for loss in losses]) * len(batch)
            rows += len(batch)
        return tuple(float(value) for value in totals / rows)

    def _losses(self, batch: torch.Tensor) -> tuple[float, float, float]:
        assert self.model_ is not None
        self.model_.eval()
        with torch.inference_mode():
            losses = self._component_tensors(batch.to(self._device()))
        return tuple(float(loss.mean().cpu()) for loss in losses)

    def _validate_one_hot(self, values: np.ndarray) -> None:
        for group in self.reconstruction_schema["categorical_groups"]:
            block = values[:, int(group["start"]) : int(group["stop"])]
            if not np.allclose(block.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"Categorical group {group['name']!r} is not one-hot")

    def _validate_schema(self) -> None:
        schema = self.reconstruction_schema
        if int(schema.get("version", -1)) != 1 or int(schema.get("numeric_count", 0)) < 1:
            raise ValueError("Invalid reconstruction schema")
        offset = int(schema["numeric_count"])
        for group in schema.get("categorical_groups", []):
            if int(group["start"]) != offset or int(group["stop"]) <= offset:
                raise ValueError("Categorical groups must be contiguous")
            offset = int(group["stop"])
        if offset != int(schema.get("input_width", -1)):
            raise ValueError("Reconstruction schema does not cover the input")

    def _device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def _config(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "hidden_dims", "latent_dim", "activation", "normalization", "dropout",
                "categorical_loss", "training_numeric_weight", "scoring_numeric_weight", "epochs",
                "batch_size", "learning_rate", "weight_decay", "patience",
                "internal_validation_fraction", "random_state", "device",
            )
        }


def combine_ae_components(numeric: np.ndarray, categorical: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("AE scoring weight must be between 0 and 1")
    return (weight * np.asarray(numeric) + (1 - weight) * np.asarray(categorical)).astype(np.float32)
