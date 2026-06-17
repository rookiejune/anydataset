from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anydatasets import AnyIterableDataset, DatasetSpec, Task
from anydatasets.adapters.base import DatasetAdapter
from anydatasets.cache import CacheManifest


class TorchVisionMNISTAdapter(DatasetAdapter):
    def prepare(self, spec: DatasetSpec, cache: CacheManifest):
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:
            raise ImportError("MNIST example requires torchvision.") from exc

        split = spec.split or "train"
        train = split == "train"
        root = Path(spec.options.get("root", cache.cache_path / "torchvision"))
        return datasets.MNIST(
            root=str(root),
            train=train,
            transform=transforms.ToTensor(),
            download=True,
        )

    def iter_samples(self, manifest) -> Iterator[dict]:
        for image, label in manifest:
            yield {"image": image, "label": int(label)}


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        patch_dim = patch_size * patch_size
        patch_count = (image_size // patch_size) ** 2

        self.patch_projection = nn.Linear(patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(images.shape[0], 1, -1, self.patch_size * self.patch_size)
        patches = patches.squeeze(1)

        tokens = self.patch_projection(patches)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / max(1, self.count)


def build_dataset(split: str, batch_size: int, data_dir: Path, cache_dir: Path) -> AnyIterableDataset:
    key = f"mnist-{split}"
    dataset_map = {
        key: DatasetSpec(
            source="torchvision",
            path="MNIST",
            name=key,
            split=split,
            adapter=TorchVisionMNISTAdapter(),
            options={"root": str(data_dir)},
        )
    }
    return AnyIterableDataset(
        datasets=[key],
        task=Task.IMAGE_CLASSIFICATION,
        batch_size=batch_size,
        dataset_map=dataset_map,
        cache_dir=str(cache_dir),
        shuffle=False,
        drop_last=(split == "train"),
    )


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.torch_tse_path:
        sys.path.insert(0, str(Path(args.torch_tse_path).expanduser()))
    from torch_tse import TSEOptimizer

    def tse_filter(name: str, parameter: torch.nn.Parameter) -> bool:
        if not parameter.requires_grad or parameter.ndim < 2:
            return False
        if "norm" in name or name.endswith("bias"):
            return False
        if name in {"cls_token", "position_embedding"}:
            return False
        return True

    base = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return TSEOptimizer(
        base,
        module=model,
        weight=args.tse_weight,
        gap=args.tse_gap,
        power=args.tse_power,
        parameter_filter=tse_filter,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = RunningAverage()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch.images.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.update(float(loss.item()), labels.numel())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += labels.numel()
            if batch_index + 1 >= max_batches:
                break

    return losses.value, correct / max(1, total)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.log_dir).expanduser() / args.run_name / args.optimizer
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    train_dataset = build_dataset("train", args.batch_size, Path(args.data_dir), Path(args.cache_dir))
    test_dataset = build_dataset("test", args.batch_size, Path(args.data_dir), Path(args.cache_dir))
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=0, pin_memory=device.type == "cuda")

    model = PatchTransformerClassifier(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    writer.add_text("config/optimizer", args.optimizer)
    writer.add_scalar("config/learning_rate", args.lr, 0)
    if args.optimizer == "tse":
        writer.add_scalar("config/tse_weight", args.tse_weight, 0)
        writer.add_scalar("config/tse_gap", args.tse_gap, 0)

    model.train()
    step = 0
    start = time.time()
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            images = batch.images.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            step += 1
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/accuracy", float((logits.argmax(dim=-1) == labels).float().mean().item()), step)
            if hasattr(optimizer, "last_regularizer_loss") and optimizer.last_regularizer_loss is not None:
                writer.add_scalar("train/tse_regularizer", optimizer.last_regularizer_loss, step)

            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss, val_acc = evaluate(model, test_loader, device, args.val_batches)
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/accuracy", val_acc, step)

            if step >= args.max_steps:
                break

    writer.add_scalar("train/seconds", time.time() - start, step)
    writer.flush()
    writer.close()
    print(f"{args.optimizer} finished: steps={step}, log_dir={run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Adam and TSE on MNIST with a Transformer classifier.")
    parser.add_argument("--optimizer", choices=["adam", "tse"], required=True)
    parser.add_argument("--run-name", default="mnist-transformer")
    parser.add_argument("--log-dir", default="runs/mnist_tse_vs_adam")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--cache-dir", default=".cache/anydatasets")
    parser.add_argument("--torch-tse-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tse-weight", type=float, default=1e-4)
    parser.add_argument("--tse-gap", type=float, default=1.0 - 0.5772156649015329)
    parser.add_argument("--tse-power", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
