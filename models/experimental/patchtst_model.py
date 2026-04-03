import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import get_model_settings

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

    class PatchTSTModule(nn.Module):
        def __init__(
            self,
            input_dim: int,
            patch_len: int,
            stride: int,
            d_model: int,
            n_heads: int,
            num_encoder_layers: int,
            d_ff: int,
            dropout: float,
            seq_len: int,
        ) -> None:
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.num_patches = (seq_len - patch_len) // stride + 1
            self.input_projection = nn.Linear(patch_len * input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.head = nn.Linear(d_model * self.num_patches, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            patches = []
            for i in range(self.num_patches):
                start = i * self.stride
                end = start + self.patch_len
                patch = x[:, start:end, :].reshape(batch_size, -1)
                patches.append(patch)
            patches_tensor = torch.stack(patches, dim=1)
            patches_embedded = self.input_projection(patches_tensor)
            encoded = self.encoder(patches_embedded)
            encoded = self.dropout(encoded)
            flat = encoded.reshape(batch_size, -1)
            return self.head(flat).squeeze(-1)

except ImportError:
    TORCH_AVAILABLE = False
    PatchTSTModule = None


class PatchTSTForecaster:
    def __init__(self) -> None:
        settings = get_model_settings()
        self._params = settings.experimental.params.copy()
        self._name = settings.experimental.name
        self._model = None
        self._device = "cpu"

    def train(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        val_sequences: np.ndarray,
        val_targets: np.ndarray,
    ) -> dict[str, float]:
        if not TORCH_AVAILABLE:
            logger.warning("torch_not_available_skipping_patchtst")
            return {"rmse": float("inf"), "mae": float("inf")}

        torch.manual_seed(self._params["random_state"])
        self._model = PatchTSTModule(
            input_dim=self._params["input_dim"],
            patch_len=self._params["patch_len"],
            stride=self._params["stride"],
            d_model=self._params["d_model"],
            n_heads=self._params["n_heads"],
            num_encoder_layers=self._params["num_encoder_layers"],
            d_ff=self._params["d_ff"],
            dropout=self._params["dropout"],
            seq_len=self._params["seq_len"],
        ).to(self._device)

        train_dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(targets),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._params["batch_size"],
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._params["learning_rate"])
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self._params["epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "patchtst_epoch",
                    epoch=epoch + 1,
                    loss=epoch_loss / len(train_loader),
                )

        self._model.eval()
        with torch.no_grad():
            val_tensor = torch.FloatTensor(val_sequences).to(self._device)
            predictions = self._model(val_tensor).cpu().numpy()

        rmse = float(np.sqrt(np.mean((val_targets - predictions) ** 2)))
        mae = float(np.mean(np.abs(val_targets - predictions)))
        metrics = {"rmse": rmse, "mae": mae}
        logger.info("trained_patchtst_model", **metrics)
        return metrics

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE or self._model is None:
            return np.zeros(len(sequences))
        self._model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(sequences).to(self._device)
            return self._model(tensor).cpu().numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
        else:
            with open(path, "wb") as f:
                pickle.dump(None, f)
        logger.info("saved_patchtst_model", path=path)

    def load(self, path: str) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("torch_not_available_cannot_load_patchtst")
            return
        self._model = PatchTSTModule(
            input_dim=self._params["input_dim"],
            patch_len=self._params["patch_len"],
            stride=self._params["stride"],
            d_model=self._params["d_model"],
            n_heads=self._params["n_heads"],
            num_encoder_layers=self._params["num_encoder_layers"],
            d_ff=self._params["d_ff"],
            dropout=self._params["dropout"],
            seq_len=self._params["seq_len"],
        ).to(self._device)
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
        logger.info("loaded_patchtst_model", path=path)

    @property
    def name(self) -> str:
        return self._name


def create_sequences(
    df: pd.DataFrame, feature_columns: list[str], target_col: str, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for _, group in df.groupby("engine_id"):
        features = group[feature_columns].values
        target = group[target_col].values
        for i in range(len(features) - seq_len):
            sequences.append(features[i : i + seq_len])
            targets.append(target[i + seq_len - 1])
    return np.array(sequences), np.array(targets)
