import math
import mlflow
import numpy as np
import torch
import torch.nn as nn
import os

from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from models.mlp import MLP
from scripts.cleanup import build_mix_sound_pipeline


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MixDataset(Dataset):
    """
    Denoising autoencoder dataset.

    Training  (eval_mode=False): randomly masks a fraction of observed sounds each call.
    Validation (eval_mode=True) : masks are fixed at construction time so metrics are
                                  comparable across epochs.
    Target is always the full (unmasked) confidence vector.
    """

    def __init__(self, X: np.ndarray, mask_ratio: float = 0.3, eval_mode: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask_ratio = mask_ratio
        self.eval_mode = eval_mode

        if eval_mode:
            self._masks = self._precompute_masks(X)

    def _precompute_masks(self, X: np.ndarray) -> List[torch.Tensor]:
        rng = np.random.default_rng(seed=42)
        masks = []
        for row in X:
            nonzero = np.where(row > 0)[0]
            n_mask = max(1, int(len(nonzero) * self.mask_ratio))
            mask_idx = rng.choice(nonzero, size=n_mask, replace=False)
            masks.append(torch.tensor(mask_idx, dtype=torch.long))
        return masks

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target = self.X[idx].clone()
        x = target.clone()

        if self.eval_mode:
            x[self._masks[idx]] = 0.0
        else:
            nonzero = x.nonzero(as_tuple=True)[0]
            n_mask = max(1, int(len(nonzero) * self.mask_ratio))
            mask_idx = nonzero[torch.randperm(len(nonzero))[:n_mask]]
            x[mask_idx] = 0.0

        return x, target


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stops training when val_loss has not improved by min_delta for patience epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class ModelCheckpoint:
    """Saves model weights whenever val_loss improves."""

    def __init__(self, path: str):
        self.path = path
        self._best = float("inf")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def step(self, model: nn.Module, val_loss: float) -> bool:
        if val_loss < self._best:
            self._best = val_loss
            torch.save(model.state_dict(), self.path)
            return True
        return False


class LRFinder:
    """
    Leslie Smith's learning rate range test.

    Sweeps LR exponentially from start_lr to end_lr over num_iter steps,
    tracks exponentially smoothed loss, and suggests the LR at the point
    of steepest descent. Model and optimizer state are restored afterwards.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_iter: int = 100,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        smooth_beta: float = 0.95,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_iter = num_iter
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.smooth_beta = smooth_beta

    def find(self, loader: DataLoader) -> Tuple[float, List[float], List[float]]:
        saved_model = {k: v.clone() for k, v in self.model.state_dict().items()}
        saved_optim = self.optimizer.state_dict()

        mult = (self.end_lr / self.start_lr) ** (1.0 / self.num_iter)
        lr = self.start_lr
        smoothed_loss: Optional[float] = None
        best_loss = float("inf")
        lrs: List[float] = []
        losses: List[float] = []

        self.model.train()
        loader_iter = iter(loader)

        for _ in tqdm(range(self.num_iter), desc="LR Finder", leave=False):
            try:
                x, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, target = next(loader_iter)

            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            x, target = x.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), target)
            loss.backward()
            self.optimizer.step()

            l = loss.item()
            smoothed_loss = (
                l
                if smoothed_loss is None
                else self.smooth_beta * smoothed_loss + (1 - self.smooth_beta) * l
            )
            lrs.append(lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            if smoothed_loss > 4 * best_loss:
                break

            lr *= mult

        self.model.load_state_dict(saved_model)
        self.optimizer.load_state_dict(saved_optim)

        grad = np.gradient(losses)
        suggested_lr = float(lrs[int(np.argmin(grad))])
        print(f"[LRFinder] Suggested LR: {suggested_lr:.2e}")
        return suggested_lr, lrs, losses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ndcg_at_k(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    scores = []
    for i in range(len(pred)):
        top_k = np.argsort(pred[i])[::-1][:k]
        dcg = sum(target[i, idx] / math.log2(r + 2) for r, idx in enumerate(top_k))
        ideal_vals = np.sort(target[i])[::-1][:k]
        idcg = float(np.sum(ideal_vals / np.log2(np.arange(2, len(ideal_vals) + 2))))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores))


def _hit_rate_at_k(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    hits = sum(
        bool(set(np.argsort(pred[i])[::-1][:k]) & set(np.where(target[i] > 0)[0]))
        for i in range(len(pred))
    )
    return hits / len(pred)


def _novelty(pred: np.ndarray, sound_popularity: np.ndarray, k: int) -> float:
    scores = []
    for i in range(len(pred)):
        top_k = np.argsort(pred[i])[::-1][:k]
        pop = np.clip(sound_popularity[top_k], 1e-10, 1.0)
        scores.append(float(np.mean(-np.log2(pop))))
    return float(np.mean(scores))


def _category_diversity(pred: np.ndarray, sound_ids: List[str], k: int) -> float:
    """Fraction of unique sound categories (prefix) in each top-K list, averaged over mixes."""
    scores = []
    for i in range(len(pred)):
        top_k = np.argsort(pred[i])[::-1][:k]
        categories = {sound_ids[j].split(".")[0] for j in top_k}
        scores.append(len(categories) / k)
    return float(np.mean(scores))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    sound_ids: List[str],
    sound_popularity: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    return {
        f"ndcg_{k}": _ndcg_at_k(pred, target, k),
        "hit_rate_5": _hit_rate_at_k(pred, target, 5),
        "novelty": _novelty(pred, sound_popularity, k),
        "diversity": _category_diversity(pred, sound_ids, k),
        "mse": float(np.mean((pred - target) ** 2)),
    }


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    sound_ids: List[str],
    sound_popularity: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_pred, all_target = [], []

    for x, target in loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        all_pred.append(pred.detach().cpu().numpy())
        all_target.append(target.cpu().numpy())

    pred_np = np.concatenate(all_pred)
    target_np = np.concatenate(all_target)
    train_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(pred_np, target_np, sound_ids, sound_popularity)
    return train_loss, metrics


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    sound_ids: List[str],
    sound_popularity: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_pred, all_target = [], []

    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)
            total_loss += criterion(pred, target).item() * len(x)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())

    pred_np = np.concatenate(all_pred)
    target_np = np.concatenate(all_target)
    val_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(pred_np, target_np, sound_ids, sound_popularity)
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    sound_ids: List[str],
    sound_popularity: np.ndarray,
    n_epochs: int = 50,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint: Optional[ModelCheckpoint] = None,
) -> Dict[str, list]:
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, sound_ids, sound_popularity
        )
        val_loss, val_metrics = val_epoch(
            model, val_loader, criterion, device, sound_ids, sound_popularity
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        mlflow.log_metrics(
            {"train/loss": train_loss, "val/loss": val_loss}
            | {f"train/{k}": v for k, v in train_metrics.items()}
            | {f"val/{k}": v for k, v in val_metrics.items()},
            step=epoch,
        )

        val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        tqdm.write(
            f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | {val_str}"
        )

        if checkpoint and checkpoint.step(model, val_loss):
            tqdm.write(f"           -> checkpoint saved (val_loss={val_loss:.4f})")

        if early_stopping and early_stopping.step(val_loss):
            tqdm.write(f"           -> early stopping at epoch {epoch}")
            break

    return history


def main(cfg: Dict) -> Tuple[nn.Module, Dict[str, list]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    X, sound_to_idx, idx_to_sound = build_mix_sound_pipeline(
        listening_file_path=cfg["LISTENING_EVENT_FILE_PATH"],
        sound_id_file_path=cfg["SOUNDS_LISTENING_IDS_PATH"],
    )
    sound_ids = [idx_to_sound[i] for i in range(len(idx_to_sound))]
    n_mixes, n_sounds = X.shape
    print(f"Matrix shape : {n_mixes} mixes x {n_sounds} sounds")

    split = int(cfg["TRAIN_SPLIT"] * n_mixes)
    X_train, X_val = X[:split], X[split:]

    # sound popularity from training set only (fraction of mixes each sound appears in)
    sound_popularity = (X_train > 0).mean(axis=0)

    mask_ratio = cfg.get("MASK_RATIO", 0.3)
    train_ds = MixDataset(X_train, mask_ratio=mask_ratio, eval_mode=False)
    val_ds = MixDataset(X_val, mask_ratio=mask_ratio, eval_mode=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=2)

    model = MLP(
        input_size=n_sounds,
        output_size=n_sounds,
        hidden_layers=cfg.get("HIDDEN_LAYERS", [512, 256, 512]),
        activation_fn=nn.ReLU,
        dropout=cfg.get("DROPOUT", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # find optimal LR before training
    lr_finder = LRFinder(model, optimizer, criterion, device, num_iter=100)
    suggested_lr, _, _ = lr_finder.find(train_loader)
    for pg in optimizer.param_groups:
        pg["lr"] = suggested_lr

    early_stopping = EarlyStopping(patience=cfg.get("PATIENCE", 5))
    checkpoint_path = cfg.get("CHECKPOINT_PATH", "checkpoints/best_model.pt")
    checkpoint = ModelCheckpoint(path=checkpoint_path)

    # initialize MLflow — fall back to local ./mlruns if the remote server is unreachable
    try:
        mlflow.set_tracking_uri(cfg.get("MLFLOW_URI", "http://10.0.0.1:5000"))
        mlflow.set_experiment(cfg.get("MLFLOW_EXPERIMENT", "sounds-recommender"))
    except Exception as e:
        print(f"[MLflow] Remote server unreachable ({e}), falling back to local ./mlruns")
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(cfg.get("MLFLOW_EXPERIMENT", "sounds-recommender"))
    print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    with mlflow.start_run():
        mlflow.log_params(
            {
                **{
                    k: v
                    for k, v in cfg.items()
                    if k not in ("LISTENING_EVENT_FILE_PATH", "SOUNDS_LISTENING_IDS_PATH")
                },
                "suggested_lr": suggested_lr,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_sounds": n_sounds,
            }
        )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            sound_ids=sound_ids,
            sound_popularity=sound_popularity,
            n_epochs=cfg.get("N_EPOCHS", 50),
            early_stopping=early_stopping,
            checkpoint=checkpoint,
        )

        # load best weights before logging
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        mlflow.pytorch.log_model(model, artifact_path="model")

    return model, history


if __name__ == "__main__":
    load_dotenv()
    cfg = {
        # "LISTENING_EVENT_FILE_PATH": "/Users/emulie/Downloads/script_job_67b7f56b753852fd2a3f35baed75edbc_0.csv",
        "LISTENING_EVENT_FILE_PATH": "/Users/emulie/Downloads/bq-results-20260325-195823-1774468728937.csv",  # for Feb 1st to March 1st
        "SOUNDS_LISTENING_IDS_PATH": "data/sounds_ids.json",
        "TRAIN_SPLIT": 0.8,
        "BATCH_SIZE": 256,
        "MASK_RATIO": 0.3,
        "HIDDEN_LAYERS": [512, 128, 512],
        "DROPOUT": 0.3,
        "PATIENCE": 5,
        "N_EPOCHS": 100,
        "CHECKPOINT_PATH": "checkpoints/best_model.pt",
        "MLFLOW_URI": os.environ.get("MLFLOW_URI"),
        "MLFLOW_EXPERIMENT": os.environ.get("MLFLOW_EXPERIMENT", "sounds-recommender"),
    }
    model, history = main(cfg=cfg)
