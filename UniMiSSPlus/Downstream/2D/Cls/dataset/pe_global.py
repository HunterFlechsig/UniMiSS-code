import os

import numpy as np
from PIL import Image
from torch.utils import data

from dataset.my_datasets import build_transform_classification

BIT_NAMES = (
    'Pleural effusion',
    'Lung tumor',
    'Pneumonia',
    'Tuberculosis',
    'Other diseases',
    'No finding',
)
N_BITS = len(BIT_NAMES)
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def label_width(list_path):
    with open(list_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                return len(parts) - 1
    raise ValueError('%s has no labels' % list_path)


def read_label_list(list_path):
    rows = []
    with open(list_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 1 + N_BITS:
                raise ValueError(
                    '%s:%d expected a path and %d label bits' % (list_path, line_number, N_BITS)
                )
            bits = np.array([int(part) for part in parts[1:]], dtype=np.int64)
            if np.any((bits != 0) & (bits != 1)):
                raise ValueError('%s:%d labels are 0 or 1' % (list_path, line_number))
            rows.append((parts[0], bits))
    if not rows:
        raise ValueError('%s is empty' % list_path)
    return rows


def write_label_list(list_path, rows):
    with open(list_path, 'w') as handle:
        for relative, bits in rows:
            handle.write(relative + ' ' + ' '.join(str(int(bit)) for bit in bits) + '\n')


def resolve_radiograph(root, relative):
    candidates = [relative]
    if os.path.splitext(relative)[1] == '':
        candidates.extend(relative + suffix for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        path = os.path.join(root, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(os.path.join(root, relative))


def stratified_validation_indices(labels, val_fraction=0.1, seed=1234):
    labels = np.asarray(labels, dtype=np.int64)
    n_rows, n_bits = labels.shape
    if n_bits != N_BITS:
        raise ValueError('expected %d bits' % N_BITS)
    rng = np.random.RandomState(seed)
    n_val = int(round(n_rows * val_fraction))
    positive = labels.sum(axis=0)
    negative = n_rows - positive
    for bit, name in enumerate(BIT_NAMES):
        if positive[bit] < 2 or negative[bit] < 2:
            raise ValueError(
                '%s has %d positives and %d negatives in the training list; '
                'the fit list and the validation list each need one of both'
                % (name, int(positive[bit]), int(negative[bit]))
            )

    desired = np.rint(positive * val_fraction).astype(np.int64)
    for bit in range(n_bits):
        desired[bit] = int(np.clip(desired[bit], 1, positive[bit] - 1))

    remaining = np.ones(n_rows, dtype=bool)
    in_val = np.zeros(n_rows, dtype=bool)
    got = np.zeros(n_bits, dtype=np.int64)
    fit_positive = positive.copy()
    fit_negative = negative.copy()

    def can_move_to_validation(index):
        return np.all(fit_positive - labels[index] >= 1) and np.all(fit_negative - (1 - labels[index]) >= 1)

    while int(in_val.sum()) < n_val:
        deficits = desired - got
        eligible = [
            bit for bit in range(n_bits)
            if deficits[bit] > 0 and np.any(remaining & (labels[:, bit] == 1))
        ]
        if not eligible:
            break
        bit = min(eligible, key=lambda item: int((remaining & (labels[:, item] == 1)).sum()))
        pool = np.flatnonzero(remaining & (labels[:, bit] == 1))
        movable = np.all(fit_positive - labels[pool] >= 1, axis=1) & np.all(fit_negative - (1 - labels[pool]) >= 1, axis=1)
        safe = pool[movable]
        if safe.size == 0:
            break
        scores = labels[safe] @ np.maximum(deficits, 0)
        choices = safe[scores == scores.max()]
        choice = int(rng.choice(choices))
        in_val[choice] = True
        remaining[choice] = False
        got += labels[choice]
        fit_positive -= labels[choice]
        fit_negative -= 1 - labels[choice]

    order = np.flatnonzero(remaining)
    rng.shuffle(order)
    for index in order:
        if int(in_val.sum()) >= n_val:
            break
        index = int(index)
        if can_move_to_validation(index):
            in_val[index] = True
            remaining[index] = False
            fit_positive -= labels[index]
            fit_negative -= 1 - labels[index]

    for bit, name in enumerate(BIT_NAMES):
        val_positive = int(labels[in_val, bit].sum())
        val_negative = int(in_val.sum()) - val_positive
        fit_positive = int(labels[remaining, bit].sum())
        fit_negative = int(remaining.sum()) - fit_positive
        if min(val_positive, val_negative, fit_positive, fit_negative) < 1:
            raise RuntimeError(
                'validation cut left %s without a positive and a negative in both lists' % name
            )
    return np.flatnonzero(remaining), np.flatnonzero(in_val)


def split_training_rows(rows, val_fraction=0.1, seed=1234):
    labels = np.stack([bits for _, bits in rows], axis=0)
    fit_index, val_index = stratified_validation_indices(labels, val_fraction, seed)
    fit_rows = [rows[int(index)] for index in fit_index]
    val_rows = [rows[int(index)] for index in val_index]
    return fit_rows, val_rows


class PEGlobalDataset(data.Dataset):
    def __init__(self, root, rows, mode):
        self.root = root
        self.rows = rows
        self.augmentation = build_transform_classification(normalize='chestx-ray', mode=mode)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        relative, bits = self.rows[index]
        path = resolve_radiograph(self.root, relative)
        image = Image.open(path).convert('RGB')
        image = self.augmentation(image)
        return image, bits.astype(np.float32)
