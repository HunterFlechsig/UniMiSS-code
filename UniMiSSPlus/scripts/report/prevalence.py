#!/usr/bin/env python3
"""Count labels on the fit, validation, and test lists. No images and no GPU.

    python UniMiSSPlus/scripts/report/prevalence.py

Writes report/tables/vindr_prevalence.csv and ricord_counts.csv.
The VinDr-CXR fit and validation lists appear after the first training launch.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BIT_NAMES,
    RICORD_DIR,
    TABLES,
    VINDR_LIST_DIR,
    read_pe_global_list,
    read_ricord_list,
    write_csv,
)


def vindr_rows(split, path):
    listed = read_pe_global_list(path)
    n_rows = len(listed)
    rows = []
    for bit, name in enumerate(BIT_NAMES):
        positives = sum(bits[bit] for _, bits in listed)
        rows.append({
            'split': split,
            'bit': name,
            'positives': positives,
            'radiographs': n_rows,
            'positive_rate': positives / float(n_rows),
        })
    return rows


def ricord_row(split, path):
    listed = read_ricord_list(path)
    category_1 = sum(label for _, label in listed)
    n_rows = len(listed)
    return {
        'split': split,
        'volumes': n_rows,
        'category_1': category_1,
        'category_0': n_rows - category_1,
        'category_1_rate': category_1 / float(n_rows),
    }


def collect(specs, reader):
    rows = []
    missing = []
    for split, path in specs:
        if not path.is_file():
            missing.append(path)
            print('list not found: %s' % path)
            continue
        rows.extend(reader(split, path))
    return rows, missing


def main():
    parser = argparse.ArgumentParser(description='Count PE-global bits and RICORD categories.')
    parser.add_argument('--fit-list', type=Path, default=VINDR_LIST_DIR / 'fit_pe_global_one.txt')
    parser.add_argument('--val-list', type=Path, default=VINDR_LIST_DIR / 'val_pe_global_one.txt')
    parser.add_argument('--test-list', type=Path, default=VINDR_LIST_DIR / 'test_pe_global_one.txt')
    parser.add_argument('--ricord-train', type=Path, default=RICORD_DIR / 'dataset' / 'lists' / 'RICORD_train.txt')
    parser.add_argument('--ricord-val', type=Path, default=RICORD_DIR / 'dataset' / 'lists' / 'RICORD_val.txt')
    parser.add_argument('--ricord-test', type=Path, default=RICORD_DIR / 'dataset' / 'lists' / 'RICORD_test.txt')
    parser.add_argument('--tables', type=Path, default=TABLES)
    args = parser.parse_args()

    vindr, vindr_missing = collect(
        (('fit', args.fit_list), ('validation', args.val_list), ('test', args.test_list)),
        vindr_rows,
    )
    ricord, ricord_missing = collect(
        (('train', args.ricord_train), ('validation', args.ricord_val), ('test', args.ricord_test)),
        lambda split, path: [ricord_row(split, path)],
    )
    if vindr:
        write_csv(
            args.tables / 'vindr_prevalence.csv',
            ['split', 'bit', 'positives', 'radiographs', 'positive_rate'],
            vindr,
        )
    if ricord:
        write_csv(
            args.tables / 'ricord_counts.csv',
            ['split', 'volumes', 'category_1', 'category_0', 'category_1_rate'],
            ricord,
        )
    print('tables in %s' % args.tables)
    if vindr_missing or ricord_missing or not vindr or not ricord:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
