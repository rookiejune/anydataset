from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from anydataset import (
    AnyDataset,
    ImageItem,
    ImageOptKey,
    ImageView,
    Modality,
    Role,
    Source,
    Spec,
    Task,
)


DATASETS = {
    "mnist": {
        "torchvision_name": "MNIST",
        "image_size": 28,
        "image_channels": 1,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashion_mnist": {
        "torchvision_name": "FashionMNIST",
        "image_size": 28,
        "image_channels": 1,
        "num_classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "kmnist": {
        "torchvision_name": "KMNIST",
        "image_size": 28,
        "image_channels": 1,
        "num_classes": 10,
        "mean": (0.1918,),
        "std": (0.3483,),
    },
    "cifar10": {
        "torchvision_name": "CIFAR10",
        "image_size": 32,
        "image_channels": 3,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}


class TorchVisionImageDataset(AnyDataset):
    def prepare(self):
        if self._dataset is not None:
            return self._dataset

        try:
            from torchvision import datasets, transforms
        except ImportError as exc:
            raise ImportError("This example requires torchvision.") from exc

        self.cache_manager.prepare(self.spec)
        split = self.spec.split or "train"
        train = split == "train"
        root = Path(self.spec.load_options["root"])
        dataset_name = self.spec.load_options["torchvision_name"]
        mean = tuple(self.spec.load_options["mean"])
        std = tuple(self.spec.load_options["std"])

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dataset_cls = getattr(datasets, dataset_name)
        self._dataset = dataset_cls(
            root=str(root),
            train=train,
            transform=transform,
            download=True,
        )
        return self._dataset

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {
            (Role.DEFAULT, Modality.IMAGE): ImageItem(
                views={ImageView.PIXEL: image},
                optional={ImageOptKey.LABEL: int(label)},
            )
        }


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        image_size: int,
        image_channels: int,
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
        patch_dim = image_channels * patch_size * patch_size
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
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(images.shape[0], -1, images.shape[1] * self.patch_size * self.patch_size)

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


class ChainedOptimizer:
    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        self.optimizers = optimizers
        self.last_regularizer_loss = None

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    @property
    def state(self):
        merged = {}
        for optimizer in self.optimizers:
            merged.update(optimizer.state)
        return merged

    @property
    def defaults(self):
        return self.optimizers[0].defaults

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        if closure is not None:
            loss = closure()
        else:
            loss = None

        for optimizer in self.optimizers:
            optimizer.step()
        return loss


def is_tse_matrix_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    if not parameter.requires_grad or parameter.ndim != 2:
        return False
    if "norm" in name or name.endswith("bias"):
        return False
    return True


def build_dataset(
    dataset_name: str,
    split: str,
    data_dir: Path,
    cache_dir: Path,
) -> AnyDataset:
    config = DATASETS[dataset_name]
    return TorchVisionImageDataset(
        spec=Spec(
            source=Source.LOCAL,
            path=str(data_dir / dataset_name),
            split=split,
            load_options={
                "root": str(data_dir / dataset_name),
                "torchvision_name": config["torchvision_name"],
                "mean": config["mean"],
                "std": config["std"],
            },
        ),
        cache_dir=str(cache_dir),
    )


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == "muon":
        return build_muon_optimizer(args, model)

    if args.torch_tse_path:
        sys.path.insert(0, str(Path(args.torch_tse_path).expanduser()))
    from torch_tse import TSEOptimizer

    if args.optimizer == "tse":
        base = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "tse_muon":
        base = build_muon_optimizer(args, model)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return TSEOptimizer(
        base,
        module=model,
        weight=args.tse_weight,
        gap=args.tse_gap,
        power=args.tse_power,
        mode=args.tse_mode,
        projection_strength=args.tse_projection_strength,
        projection_interval=args.tse_projection_interval,
        spectral_lr=args.tse_spectral_lr,
        spectral_momentum=args.tse_spectral_momentum,
        parameter_filter=is_tse_matrix_parameter,
    )


def build_muon_optimizer(args: argparse.Namespace, model: nn.Module) -> ChainedOptimizer:
    matrix_params = []
    auxiliary_params = []
    for name, parameter in model.named_parameters():
        if is_tse_matrix_parameter(name, parameter):
            matrix_params.append(parameter)
        elif parameter.requires_grad:
            auxiliary_params.append(parameter)

    if not matrix_params:
        raise ValueError("Muon requires at least one 2D matrix parameter.")

    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.Muon(
            matrix_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            nesterov=not args.no_muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
        )
    ]
    if auxiliary_params:
        optimizers.append(
            torch.optim.Adam(
                auxiliary_params,
                lr=args.aux_lr,
                weight_decay=args.aux_weight_decay,
            )
        )

    return ChainedOptimizer(*optimizers)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = RunningAverage()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            item = batch.sample[(Role.DEFAULT, Modality.IMAGE)]
            images = item.views[ImageView.PIXEL].to(device, non_blocking=True)
            labels = item.optional[ImageOptKey.LABEL].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.update(float(loss.item()), labels.numel())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += labels.numel()
            if batch_index + 1 >= max_batches:
                break

    return losses.value, correct / max(1, total)


def train(args: argparse.Namespace) -> None:
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_config = DATASETS[args.dataset]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.run_name or f"{args.dataset}-transformer"
    optimizer_label = args.optimizer
    if args.optimizer in {"tse", "tse_muon"}:
        optimizer_label = f"{args.optimizer}_{args.tse_mode}"
    run_dir = Path(args.log_dir).expanduser() / run_name / optimizer_label
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    train_dataset = build_dataset(
        args.dataset,
        "train",
        Path(args.data_dir),
        Path(args.cache_dir),
    )
    test_dataset = build_dataset(
        args.dataset,
        "test",
        Path(args.data_dir),
        Path(args.cache_dir),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=Task.IMAGE_CLASSIFICATION.collate_fn(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=Task.IMAGE_CLASSIFICATION.collate_fn(),
    )

    model = PatchTransformerClassifier(
        image_size=dataset_config["image_size"],
        image_channels=dataset_config["image_channels"],
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        num_classes=dataset_config["num_classes"],
        dropout=args.dropout,
    ).to(device)
    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    writer.add_text("config/dataset", args.dataset)
    writer.add_text("config/optimizer", args.optimizer)
    writer.add_text("config/optimizer_label", optimizer_label)
    writer.add_scalar("config/learning_rate", args.lr, 0)
    writer.add_scalar("config/batch_size", args.batch_size, 0)
    writer.add_scalar("config/max_steps", args.max_steps, 0)
    if args.optimizer in {"muon", "tse_muon"}:
        writer.add_scalar("config/aux_learning_rate", args.aux_lr, 0)
        writer.add_scalar("config/muon_momentum", args.muon_momentum, 0)
        writer.add_scalar("config/muon_ns_steps", args.muon_ns_steps, 0)
    if args.optimizer in {"tse", "tse_muon"}:
        writer.add_text("config/tse_mode", args.tse_mode)
        writer.add_scalar("config/tse_weight", args.tse_weight, 0)
        writer.add_scalar("config/tse_gap", args.tse_gap, 0)
        writer.add_scalar("config/tse_projection_strength", args.tse_projection_strength, 0)
        writer.add_scalar("config/tse_projection_interval", args.tse_projection_interval, 0)
        writer.add_scalar("config/tse_spectral_lr", args.tse_spectral_lr, 0)
        writer.add_scalar("config/tse_spectral_momentum", args.tse_spectral_momentum, 0)

    model.train()
    step = 0
    start = time.time()
    last_time = start
    last_step = 0
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            item = batch.sample[(Role.DEFAULT, Modality.IMAGE)]
            images = item.views[ImageView.PIXEL].to(device, non_blocking=True)
            labels = item.optional[ImageOptKey.LABEL].to(device, non_blocking=True)

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

            if step % args.time_every == 0 or step == args.max_steps:
                now = time.time()
                elapsed = now - start
                recent_steps = max(1, step - last_step)
                writer.add_scalar("time/seconds_elapsed", elapsed, step)
                writer.add_scalar("time/seconds_per_step_mean", elapsed / step, step)
                writer.add_scalar("time/seconds_per_step_recent", (now - last_time) / recent_steps, step)
                if device.type == "cuda":
                    writer.add_scalar("time/max_cuda_memory_gb", torch.cuda.max_memory_allocated() / 1e9, step)
                last_time = now
                last_step = step

            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss, val_acc = evaluate(model, test_loader, device, args.val_batches)
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/accuracy", val_acc, step)

            if step >= args.max_steps:
                break

    elapsed = time.time() - start
    writer.add_scalar("time/seconds_total", elapsed, step)
    writer.add_scalar("time/seconds_per_step_total", elapsed / max(1, step), step)
    writer.flush()
    writer.close()
    print(
        f"{args.optimizer} finished: steps={step}, seconds={elapsed:.3f}, "
        f"seconds_per_step={elapsed / max(1, step):.6f}, log_dir={run_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Adam and TSE on image classification with a Transformer.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--optimizer", choices=["adam", "tse", "muon", "tse_muon"], required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-dir", default="runs/image_transformer_tse_vs_adam")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default=".cache/anydataset")
    parser.add_argument("--torch-tse-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--time-every", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--aux-lr", type=float, default=3e-4)
    parser.add_argument("--aux-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"], default="original")
    parser.add_argument("--no-muon-nesterov", action="store_true")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tse-weight", type=float, default=1e-4)
    parser.add_argument("--tse-gap", type=float, default=1.0 - 0.5772156649015329)
    parser.add_argument("--tse-power", type=float, default=1.0)
    parser.add_argument(
        "--tse-mode",
        choices=["regularizer", "projection", "spectral_momentum", "periodic_projection"],
        default="regularizer",
    )
    parser.add_argument("--tse-projection-strength", type=float, default=1.0)
    parser.add_argument("--tse-projection-interval", type=int, default=100)
    parser.add_argument("--tse-spectral-lr", type=float, default=0.1)
    parser.add_argument("--tse-spectral-momentum", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
