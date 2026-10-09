"""Paths and log parsing shared by the report scripts."""

import csv
import math
from pathlib import Path


UNIMISS = Path(__file__).resolve().parents[2]
CLS_DIR = UNIMISS / 'Downstream' / '2D' / 'Cls'
RICORD_DIR = UNIMISS / 'Downstream' / '3D' / 'RICORD'
FIGURES = UNIMISS / 'report' / 'figures'
TABLES = UNIMISS / 'report' / 'tables'

VINDR_LOG = CLS_DIR / 'models' / 'VinDr-CXR' / 'outputxx.txt'
RICORD_LOG = RICORD_DIR / 'models' / 'outputxx.txt'
VINDR_LIST_DIR = CLS_DIR / 'dataset' / 'VinDr-CXR' / 'pe_global'
VINDR_CHECKPOINT = CLS_DIR / 'models' / 'VinDr-CXR' / 'ckp_weights.pth'
VINDR_ROOT = CLS_DIR / 'dataset' / 'VinDr-CXR'
RICORD_CHECKPOINT = RICORD_DIR / 'models' / 'Best.pth'

BIT_NAMES = (
    'Pleural effusion',
    'Lung tumor',
    'Pneumonia',
    'Tuberculosis',
    'Other diseases',
    'No finding',
)
BIT_KEYS = tuple(name.lower().replace(' ', '_') for name in BIT_NAMES)

SCORE_COLUMNS = (
    'collection',
    'split',
    'epoch',
    'headline_auc',
    'macro_ap',
    'pleural_effusion_auc',
    'lung_tumor_auc',
    'pneumonia_auc',
    'tuberculosis_auc',
    'other_diseases_auc',
    'no_finding_auc',
    'sensitivity',
    'specificity',
)

SCORE_ORDER = (
    ('VinDr-CXR', 'validation'),
    ('VinDr-CXR', 'test'),
    ('RICORD', 'validation'),
    ('RICORD', 'test'),
)


def bit_slug(name):
    return name.lower().replace(' ', '_')


def parse_pairs(text):
    pairs = {}
    for part in text.split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        key, value = part.split('=', 1)
        pairs[key.strip()] = float(value)
    return pairs


def last_by_epoch(rows):
    """Keep the last appended line for each epoch, then sort by epoch."""
    by_epoch = {}
    for row in rows:
        by_epoch[row['epoch']] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def parse_vindr_log(path):
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line.startswith('val_epoch'):
                continue
            try:
                head, body = line.split(':', 1)
                epoch = int(head.replace('val_epoch', ''))
                pairs = parse_pairs(body)
                row = {'epoch': epoch, 'macro_auc': pairs['macro_auc'], 'macro_ap': pairs['macro_ap']}
                for key in BIT_KEYS:
                    row[key + '_auc'] = pairs[key + '_auc']
            except (KeyError, ValueError) as error:
                raise ValueError('%s:%d is not a VinDr-CXR validation line (%s)' % (path, line_number, error))
            rows.append(row)
    return rows


def parse_ricord_log(path):
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line.startswith('val') or line.startswith('val_epoch'):
                continue
            try:
                head, body = line.split(':', 1)
                epoch = int(head.replace('val', ''))
                pairs = parse_pairs(body)
                row = {
                    'epoch': epoch,
                    'accuracy': pairs['vacc'],
                    'auc': pairs['vauc'],
                    'average_precision': pairs['vAP'],
                    'sensitivity': pairs['vsens'],
                    'specificity': pairs['vspec'],
                }
            except (KeyError, ValueError) as error:
                raise ValueError('%s:%d is not a RICORD validation line (%s)' % (path, line_number, error))
            rows.append(row)
    return rows


def winning_row(rows, score_key, initial):
    """The log line that training kept, replaying every line in file order."""
    best = initial
    chosen = None
    for row in rows:
        score = row[score_key]
        if score > best:
            best = score
            chosen = row
    return chosen


def checkpoint_epoch(rows, score_key, initial):
    chosen = winning_row(rows, score_key, initial)
    if chosen is None:
        return None
    return chosen['epoch']


def read_pe_global_list(path):
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 1 + len(BIT_NAMES):
                raise ValueError('%s:%d expected a path and %d bits' % (path, line_number, len(BIT_NAMES)))
            bits = []
            for part in parts[1:]:
                if part not in ('0', '1'):
                    raise ValueError('%s:%d labels are 0 or 1' % (path, line_number))
                bits.append(int(part))
            rows.append((parts[0], bits))
    if not rows:
        raise ValueError('%s is empty' % path)
    return rows


def read_ricord_list(path):
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2 or parts[1] not in ('0', '1'):
                raise ValueError('%s:%d expected a volume path and a 0 or 1 label' % (path, line_number))
            rows.append((parts[0], int(parts[1])))
    if not rows:
        raise ValueError('%s is empty' % path)
    return rows


def format_cell(value):
    if value is None or value == '':
        return ''
    if isinstance(value, float):
        if math.isnan(value):
            return ''
        return '%.6f' % value
    return str(value)


def upsert_scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.is_file():
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                existing[(row['collection'], row['split'])] = row
    for row in rows:
        key = (row['collection'], row['split'])
        existing[key] = {column: format_cell(row.get(column, '')) for column in SCORE_COLUMNS}
    ordered = []
    for key in SCORE_ORDER:
        if key in existing:
            ordered.append(existing.pop(key))
    ordered.extend(existing.values())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, '')) for key in fieldnames})
