#!/usr/bin/env python3
"""Plot validation curves from the training logs. No GPU.

    python UniMiSSPlus/scripts/report/curves_from_logs.py

Reads models/VinDr-CXR/outputxx.txt and RICORD models/outputxx.txt.
Writes report/figures and the validation rows of report/tables/scores.csv.
A repeated epoch, from a resumed job, keeps the last line for that epoch.
The marked checkpoint is the first epoch that strictly improved the score,
which is the epoch whose weights training saved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BIT_KEYS,
    FIGURES,
    RICORD_LOG,
    TABLES,
    VINDR_LOG,
    checkpoint_epoch,
    last_by_epoch,
    parse_ricord_log,
    parse_vindr_log,
    upsert_scores,
    winning_row,
    write_csv,
)
from plots import plot_ricord_validation, plot_vindr_bits, plot_vindr_macro


def vindr_score_row(rows, checkpoint):
    chosen = next(row for row in rows if row['epoch'] == checkpoint)
    score = {
        'collection': 'VinDr-CXR',
        'split': 'validation',
        'epoch': checkpoint,
        'headline_auc': chosen['macro_auc'],
        'macro_ap': chosen['macro_ap'],
    }
    for key in BIT_KEYS:
        score[key + '_auc'] = chosen[key + '_auc']
    return score


def ricord_score_row(rows, checkpoint):
    chosen = next(row for row in rows if row['epoch'] == checkpoint)
    return {
        'collection': 'RICORD',
        'split': 'validation',
        'epoch': checkpoint,
        'headline_auc': chosen['auc'],
        'macro_ap': chosen['average_precision'],
        'sensitivity': chosen['sensitivity'],
        'specificity': chosen['specificity'],
    }


def write_vindr(log_path, figures, tables):
    recorded = parse_vindr_log(log_path)
    rows = last_by_epoch(recorded)
    if not rows:
        print('no VinDr-CXR validation lines in %s' % log_path)
        return False, None
    checkpoint = checkpoint_epoch(recorded, 'macro_auc', initial=-1.0)
    write_csv(
        tables / 'vindr_validation_epochs.csv',
        ['epoch', 'macro_auc', 'macro_ap'] + [key + '_auc' for key in BIT_KEYS],
        rows,
    )
    plot_vindr_macro(rows, checkpoint, figures / 'vindr_val_macro_auc.png')
    plot_vindr_bits(rows, checkpoint, figures / 'vindr_val_per_bit_auc.png')
    print('VinDr-CXR checkpoint epoch %s from %d validation epochs' % (checkpoint, len(rows)))
    if checkpoint is None:
        return True, None
    return True, vindr_score_row([winning_row(recorded, 'macro_auc', -1.0)], checkpoint)


def write_ricord(log_path, figures, tables):
    recorded = parse_ricord_log(log_path)
    rows = last_by_epoch(recorded)
    if not rows:
        print('no RICORD validation lines in %s' % log_path)
        return False, None
    checkpoint = checkpoint_epoch(recorded, 'auc', initial=0.0)
    write_csv(
        tables / 'ricord_validation_epochs.csv',
        ['epoch', 'auc', 'average_precision', 'sensitivity', 'specificity', 'accuracy'],
        rows,
    )
    plot_ricord_validation(rows, checkpoint, figures / 'ricord_val_auc.png')
    print('RICORD checkpoint epoch %s from %d validation epochs' % (checkpoint, len(rows)))
    if checkpoint is None:
        return True, None
    return True, ricord_score_row([winning_row(recorded, 'auc', 0.0)], checkpoint)


def main():
    parser = argparse.ArgumentParser(description='Plot validation curves from the training logs.')
    parser.add_argument('--vindr-log', type=Path, default=VINDR_LOG)
    parser.add_argument('--ricord-log', type=Path, default=RICORD_LOG)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    parser.add_argument('--tables', type=Path, default=TABLES)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    score_rows = []
    wrote = False
    if args.vindr_log.is_file():
        plotted, row = write_vindr(args.vindr_log, args.figures, args.tables)
        wrote = wrote or plotted
        if row is not None:
            score_rows.append(row)
    else:
        print('VinDr-CXR log not found: %s' % args.vindr_log)
    if args.ricord_log.is_file():
        plotted, row = write_ricord(args.ricord_log, args.figures, args.tables)
        wrote = wrote or plotted
        if row is not None:
            score_rows.append(row)
    else:
        print('RICORD log not found: %s' % args.ricord_log)
    if score_rows:
        upsert_scores(args.tables / 'scores.csv', score_rows)
    if not wrote:
        return 1
    print('figures in %s' % args.figures)
    print('tables in %s' % args.tables)
    return 0


if __name__ == '__main__':
    sys.exit(main())
