"""Individual labeled student-teacher.

One named experiment trains one collection for a requested number of epochs.
The student encoder, consistency head, and task head take the label loss.
A consistency loss matches the two consistency heads. The teacher receives no
gradient. Its encoder and consistency head take one exponential-moving-average
step from the student at the end of the epoch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

TEACHER_MOMENTUM = 0.9


@dataclass
class Observation:
    """One image and that collection's own labels.

    A missing label is an unmentioned observation and counts as negative.
    None is an uncertain observation: it is left out of the loss and out of
    that label's ROC-AUC. 0 is negative. 1 is positive.
    `image` is the resized original the teacher sees. `augmented` is the
    student view. When `augmented` is omitted, the student sees `image`.
    """

    image: torch.Tensor
    labels: dict[str, Optional[float]]
    augmented: Optional[torch.Tensor] = None


@dataclass
class Collection:
    """Images, labels, a fit list, and a validation list for one collection."""

    name: str
    label_names: tuple[str, ...]
    fit: tuple[Observation, ...]
    validation: tuple[Observation, ...]


class Student(nn.Module):
    """Encoder, consistency head, and the current collection's task head."""

    def __init__(self, image_size: int, n_labels: int, feature_dim: int = 8, consistency_dim: int = 4):
        super().__init__()
        self.encoder = nn.Linear(image_size, feature_dim)
        self.consistency_head = nn.Linear(feature_dim, consistency_dim)
        self.task_head = nn.Linear(consistency_dim, n_labels)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images.flatten(1))
        embedding = self.consistency_head(features)
        logits = self.task_head(embedding)
        return embedding, logits

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(images)
            probabilities = torch.sigmoid(logits)
        self.train(was_training)
        return probabilities


class Teacher(nn.Module):
    """Teacher encoder and consistency head. The task head is not part of the teacher."""

    def __init__(self, image_size: int, feature_dim: int = 8, consistency_dim: int = 4):
        super().__init__()
        self.encoder = nn.Linear(image_size, feature_dim)
        self.consistency_head = nn.Linear(feature_dim, consistency_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images.flatten(1))
        return self.consistency_head(features)


def initial_student_teacher(collection: Collection, seed: int) -> tuple[Student, Teacher]:
    """Untrained student, and a teacher that starts as a copy of that student."""
    image_size = _image_size(collection)
    torch.manual_seed(seed)
    student = Student(image_size, len(collection.label_names))
    teacher = Teacher(image_size)
    teacher.encoder.load_state_dict(student.encoder.state_dict())
    teacher.consistency_head.load_state_dict(student.consistency_head.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return student, teacher


def run_experiment(
    name: str,
    schedule: str,
    run_list: Sequence[Collection],
    root: str | Path,
    epochs: int = 1,
    cycle_order: Optional[Sequence[str]] = None,
    pretrained_start: bool = False,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 2,
    seed: int = 0,
) -> list[dict]:
    """Train an individual labeled student-teacher and return its metrics record.

    A new experiment name starts clean. Resume continues `name` from the last
    finished epoch and keeps the metrics record. The checkpoint is the student
    and teacher from the epoch with the highest validation headline score.
    """
    if schedule != "individual":
        raise ValueError("schedule must be individual")
    if cycle_order is not None:
        raise ValueError("an individual run has no cycle order")
    if pretrained_start:
        raise NotImplementedError("pretrained start is not part of this experiment")
    if len(run_list) != 1:
        raise ValueError("an individual run list has one collection")
    _require_experiment_name(name)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    collection = run_list[0]
    _require_collection(collection)
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    resume_path = directory / "resume.pth"
    checkpoint_path = directory / "checkpoint.pth"
    metrics_path = directory / "metrics.json"

    student, teacher = initial_student_teacher(collection, seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    finished_epoch = -1
    best_headline = float("-inf")
    metrics: list[dict] = []

    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        finished_epoch = int(saved["epoch"])
        best_headline = float(saved["best_headline"])
        metrics = list(saved["metrics"])

    while finished_epoch + 1 < epochs:
        epoch = finished_epoch + 1
        training_loss = _train_epoch(student, teacher, optimizer, collection, batch_size, generator)
        validation_loss, probabilities = _validate(student, teacher, collection)
        _ema_update(teacher, student)
        row = _metrics_row(collection, probabilities, epoch, training_loss, validation_loss)
        metrics.append(row)
        finished_epoch = epoch
        if row["headline_score"] > best_headline:
            best_headline = row["headline_score"]
            _atomic_torch_save(_checkpoint_payload(student, teacher, epoch, best_headline, collection.name), checkpoint_path)
        _write_metrics(metrics_path, metrics)
        _atomic_torch_save(
            {
                "student": _cpu_state(student),
                "teacher": _cpu_state(teacher),
                "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(),
                "epoch": finished_epoch,
                "best_headline": best_headline,
                "metrics": metrics,
            },
            resume_path,
        )

    return metrics


def _train_epoch(student, teacher, optimizer, collection, batch_size, generator) -> float:
    dataset = _ObservationDataset(collection.fit, collection.label_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    student.train()
    total = 0.0
    count = 0
    for images, augmented, targets, mask in loader:
        optimizer.zero_grad()
        loss = _objective(student, teacher, augmented, images, targets, mask)
        loss.backward()
        optimizer.step()
        batch_count = int(images.shape[0])
        total += float(loss.detach()) * batch_count
        count += batch_count
    return total / count


def _validate(student, teacher, collection) -> tuple[float, torch.Tensor]:
    dataset = _ObservationDataset(collection.validation, collection.label_names)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    student.eval()
    losses = []
    probability_rows = []
    with torch.no_grad():
        for images, augmented, targets, mask in loader:
            losses.append(float(_objective(student, teacher, augmented, images, targets, mask)))
            probability_rows.append(student.probabilities(images))
    return float(sum(losses) / len(losses)), torch.cat(probability_rows, dim=0)


def _objective(student, teacher, student_images, teacher_images, targets, mask) -> torch.Tensor:
    student_embedding, logits = student(student_images)
    with torch.no_grad():
        teacher_embedding = teacher(teacher_images)
    return _masked_bce(logits, targets, mask) + F.mse_loss(student_embedding, teacher_embedding)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_label = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    valid = mask.sum()
    if float(valid) == 0.0:
        return logits.sum() * 0.0
    return (per_label * mask).sum() / valid


def _ema_update(teacher: Teacher, student: Student) -> None:
    with torch.no_grad():
        pairs = (
            (teacher.encoder, student.encoder),
            (teacher.consistency_head, student.consistency_head),
        )
        for teacher_module, student_module in pairs:
            for teacher_parameter, student_parameter in zip(teacher_module.parameters(), student_module.parameters()):
                teacher_parameter.mul_(TEACHER_MOMENTUM).add_(student_parameter, alpha=1.0 - TEACHER_MOMENTUM)


def _metrics_row(collection, probabilities, epoch, training_loss, validation_loss) -> dict:
    roc_auc = {}
    average_precision = {}
    for index, label_name in enumerate(collection.label_names):
        targets, mask = _label_columns(collection.validation, collection.label_names)
        kept = mask[:, index] > 0
        y_true = targets[kept, index].numpy()
        y_score = probabilities[kept, index].numpy()
        roc_auc[label_name] = float(metrics.roc_auc_score(y_true, y_score))
        average_precision[label_name] = float(metrics.average_precision_score(y_true, y_score))
    headline = float(sum(roc_auc.values()) / len(roc_auc))
    macro_average_precision = float(sum(average_precision.values()) / len(average_precision))
    return {
        "collection": collection.name,
        "epoch": epoch,
        "training_loss": float(training_loss),
        "validation_loss": float(validation_loss),
        "headline_score": headline,
        "roc_auc": roc_auc,
        "macro_average_precision": macro_average_precision,
        "average_precision": average_precision,
    }


def _label_columns(observations: Sequence[Observation], label_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(observations), len(label_names))
    mask = torch.ones(len(observations), len(label_names))
    for row, observation in enumerate(observations):
        for column, label_name in enumerate(label_names):
            target, valid = _target_and_mask(observation.labels, label_name)
            targets[row, column] = target
            mask[row, column] = valid
    return targets, mask


def _target_and_mask(labels: Mapping[str, Optional[float]], label_name: str) -> tuple[float, float]:
    if label_name not in labels:
        return 0.0, 1.0
    value = labels[label_name]
    if value is None:
        return 0.0, 0.0
    if value not in (0, 0.0, 1, 1.0):
        raise ValueError("a label is positive, negative, or uncertain")
    return float(value), 1.0


class _ObservationDataset(Dataset):
    def __init__(self, observations: Sequence[Observation], label_names: Sequence[str]):
        self.observations = list(observations)
        self.label_names = list(label_names)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int):
        observation = self.observations[index]
        image = observation.image.float()
        augmented = image if observation.augmented is None else observation.augmented.float()
        targets, mask = _label_columns([observation], self.label_names)
        return image, augmented, targets[0], mask[0]


def _image_size(collection: Collection) -> int:
    shape = tuple(collection.fit[0].image.shape)
    for observation in tuple(collection.fit) + tuple(collection.validation):
        if tuple(observation.image.shape) != shape:
            raise ValueError("a collection's images must share one shape")
        student_view = observation.image if observation.augmented is None else observation.augmented
        if tuple(student_view.shape) != shape:
            raise ValueError("the augmented image must match the resized original")
    return int(torch.empty(shape).numel())


def _require_collection(collection: Collection) -> None:
    if not collection.name:
        raise ValueError("a collection has a name")
    if not collection.label_names:
        raise ValueError("a collection has its own labels")
    if len(set(collection.label_names)) != len(collection.label_names):
        raise ValueError("a collection's labels are unique")
    if len(collection.fit) == 0 or len(collection.validation) == 0:
        raise ValueError("a collection has a fit list and a validation list")
    _image_size(collection)
    for label_name in collection.label_names:
        values = []
        for observation in collection.validation:
            _target, valid = _target_and_mask(observation.labels, label_name)
            if valid:
                values.append(_target)
        if 0.0 not in values or 1.0 not in values:
            raise ValueError("%s needs a positive and a negative observation on the validation list" % label_name)


def _require_experiment_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("an experiment name is a single directory name")


def _checkpoint_payload(student, teacher, epoch, headline, collection_name) -> dict:
    return {
        "student": _cpu_state(student),
        "teacher": _cpu_state(teacher),
        "epoch": epoch,
        "headline_score": headline,
        "collection": collection_name,
    }


def _cpu_state(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _write_metrics(path: Path, metrics: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
