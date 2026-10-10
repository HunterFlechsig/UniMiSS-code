#!/usr/bin/env python3
"""Score the saved checkpoints and draw the test curves. Needs one GPU.

    python UniMiSSPlus/scripts/report/score_checkpoints.py -GPU 0

VinDr-CXR loads models/VinDr-CXR/ckp_weights.pth and scores the test list
with the ten-crop view. RICORD loads models/Best.pth, the best validation
epoch, and scores the test list. A missing checkpoint is skipped.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BIT_KEYS,
    BIT_NAMES,
    CLS_DIR,
    RICORD_CHECKPOINT,
    RICORD_DIR,
    RICORD_LOG,
    TABLES,
    FIGURES,
    VINDR_CHECKPOINT,
    VINDR_LIST_DIR,
    VINDR_LOG,
    VINDR_ROOT,
    checkpoint_epoch,
    parse_ricord_log,
    parse_vindr_log,
    upsert_scores,
    write_csv,
)
from plots import plot_ricord_roc, plot_vindr_pr, plot_vindr_roc


def drop_downstream_modules():
    for name in list(sys.modules):
        if name in ('dataset', 'net', 'nets') or name.startswith(('dataset.', 'net.', 'nets.')):
            del sys.modules[name]


def checkpoint_from_log(path, parser, score_key, initial):
    if not path.is_file():
        return ''
    rows = parser(path)
    if not rows:
        return ''
    chosen = checkpoint_epoch(rows, score_key, initial)
    return '' if chosen is None else chosen


def score_vindr(args):
    import numpy as np
    import torch
    from torch.utils import data

    sys.path.insert(0, str(CLS_DIR))
    from dataset.pe_global import PEGlobalDataset, read_label_list
    from net.MiTPlus_encoder import MiTPlus_encoder

    rows = read_label_list(str(args.vindr_list))
    dataset = PEGlobalDataset(str(args.vindr_root), rows, mode='test')
    loader = data.DataLoader(
        dataset, batch_size=3, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = MiTPlus_encoder(num_classes=len(BIT_NAMES))
    state = torch.load(str(args.vindr_checkpoint), map_location='cpu')
    model.load_state_dict(state)
    model.cuda()
    model.eval()
    print('scoring %d VinDr-CXR test radiographs with ten-crop' % len(dataset), flush=True)
    labels, probabilities = collect_tencrop_scores(loader, model)
    del model
    torch.cuda.empty_cache()

    image_ids = [relative for relative, _ in dataset.rows]
    if len(image_ids) != len(labels):
        raise RuntimeError('scored %d radiographs from a list of %d' % (len(labels), len(image_ids)))
    table = []
    for index, image_id in enumerate(image_ids):
        record = {'image_id': image_id}
        for bit, key in enumerate(BIT_KEYS):
            record[key] = int(labels[index, bit])
            record[key + '_probability'] = float(probabilities[index, bit])
        table.append(record)
    fieldnames = ['image_id']
    for key in BIT_KEYS:
        fieldnames.append(key)
    for key in BIT_KEYS:
        fieldnames.append(key + '_probability')
    write_csv(args.tables / 'vindr_test_probabilities.csv', fieldnames, table)

    from sklearn import metrics
    per_auc = metrics.roc_auc_score(labels, probabilities, average=None)
    score = {
        'collection': 'VinDr-CXR',
        'split': 'test',
        'epoch': checkpoint_from_log(args.vindr_log, parse_vindr_log, 'macro_auc', -1.0),
        'headline_auc': float(np.mean(per_auc)),
        'macro_ap': float(metrics.average_precision_score(labels, probabilities, average='macro')),
    }
    for key, auc in zip(BIT_KEYS, per_auc):
        score[key + '_auc'] = float(auc)
    upsert_scores(args.tables / 'scores.csv', [score])
    plot_vindr_roc(labels, probabilities, args.figures / 'vindr_test_roc.png')
    plot_vindr_pr(labels, probabilities, args.figures / 'vindr_test_pr.png')
    print('VinDr-CXR test headline ROC-AUC %.6f' % score['headline_auc'])


def sigmoid_with_flips(model, images):
    import torch
    views = [images, torch.flip(images, [-1]), torch.flip(images, [-2]), torch.flip(images, [-1, -2])]
    probabilities = None
    for view in views:
        logits = model(view)
        probabilities = torch.sigmoid(logits) if probabilities is None else probabilities + torch.sigmoid(logits)
    return probabilities / 4.


def collect_tencrop_scores(dataloader, model):
    import numpy as np
    import torch
    probabilities = []
    labels = []
    total = len(dataloader)
    with torch.no_grad():
        for index, (images, batch_labels) in enumerate(dataloader, start=1):
            batch_size, n_crops, channels, height, width = images.size()
            images = images.view(-1, channels, height, width).cuda()
            averaged = sigmoid_with_flips(model, images)
            averaged = averaged.view(batch_size, n_crops, -1).mean(1)
            probabilities.append(averaged.cpu().numpy())
            labels.append(batch_labels.numpy())
            if index == 1 or index % 50 == 0 or index == total:
                print('VinDr-CXR test %d/%d' % (index, total), flush=True)
    return np.concatenate(labels), np.concatenate(probabilities)


def score_ricord(args):
    import numpy as np
    import torch
    from sklearn import metrics
    from torch.utils import data

    drop_downstream_modules()
    while str(CLS_DIR) in sys.path:
        sys.path.remove(str(CLS_DIR))
    sys.path.insert(0, str(RICORD_DIR))
    from dataset.mydataset3D import ValDataSet3D
    from nets.MiTPlus import model_plus

    previous = os.getcwd()
    os.chdir(str(RICORD_DIR))
    try:
        dataset = ValDataSet3D(args.ricord_root, args.ricord_list, crop_size_3D=(64, 128, 128))
        loader = data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model = model_plus(
            norm_cfg3D='IN3', activation_cfg='LeakyReLU', img_size3D=[64, 128, 128],
            num_classes=2, pretrain=False, pretrain_path=None)
        state = torch.load(str(args.ricord_checkpoint), map_location='cpu')
        model.load_state_dict(state)
        model.cuda()
        model.float()
        model.eval()
        print('scoring %d RICORD test volumes' % len(dataset), flush=True)
        probabilities = []
        labels = []
        total = len(loader)
        with torch.no_grad():
            for index, (images, batch_labels) in enumerate(loader, start=1):
                pred = torch.softmax(model(images.cuda()), dim=-1)
                probabilities.append(pred.cpu().numpy())
                labels.append(batch_labels.numpy())
                print('RICORD test %d/%d' % (index, total), flush=True)
    finally:
        os.chdir(previous)
    probabilities = np.concatenate(probabilities, 0)
    labels = np.concatenate(labels, 0)
    category_1 = probabilities[:, 1]
    names = [item['name'] for item in dataset.files]
    if len(names) != len(labels):
        raise RuntimeError('scored %d volumes from a list of %d' % (len(labels), len(names)))
    write_csv(
        args.tables / 'ricord_test_probabilities.csv',
        ['volume', 'label', 'probability'],
        [{'volume': name, 'label': int(label), 'probability': float(score)}
         for name, label, score in zip(names, labels, category_1)],
    )
    binary = np.argmax(probabilities, axis=-1)
    confusion = metrics.confusion_matrix(labels, binary, labels=[0, 1])

    def safe_rate(hit, miss):
        total = float(hit + miss)
        if total == 0:
            return float('nan')
        return hit / total

    sensitivity = safe_rate(confusion[1, 1], confusion[1, 0])
    specificity = safe_rate(confusion[0, 0], confusion[0, 1])
    score = {
        'collection': 'RICORD',
        'split': 'test',
        'epoch': checkpoint_from_log(args.ricord_log, parse_ricord_log, 'auc', 0.0),
        'headline_auc': float(metrics.roc_auc_score(labels, category_1)),
        'macro_ap': float(metrics.average_precision_score(labels, category_1)),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
    }
    upsert_scores(args.tables / 'scores.csv', [score])
    plot_ricord_roc(labels, category_1, args.figures / 'ricord_test_roc.png')
    print('RICORD test ROC-AUC %.6f' % score['headline_auc'])


def main():
    parser = argparse.ArgumentParser(description='Score VinDr-CXR and RICORD checkpoints for the report.')
    parser.add_argument('-GPU', default='0')
    parser.add_argument('--only', choices=('both', 'vindr', 'ricord'), default='both')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers. Keep 0: the ten-crop transform uses a lambda, which Python 3.14 cannot send to a worker process.')
    parser.add_argument('--vindr-checkpoint', type=Path, default=VINDR_CHECKPOINT)
    parser.add_argument('--vindr-list', type=Path, default=VINDR_LIST_DIR / 'test_pe_global_one.txt')
    parser.add_argument('--vindr-root', type=Path, default=VINDR_ROOT)
    parser.add_argument('--vindr-log', type=Path, default=VINDR_LOG)
    parser.add_argument('--ricord-checkpoint', type=Path, default=RICORD_CHECKPOINT)
    parser.add_argument('--ricord-root', default='dataset/')
    parser.add_argument('--ricord-list', default='lists/RICORD_test.txt')
    parser.add_argument('--ricord-log', type=Path, default=RICORD_LOG)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    parser.add_argument('--tables', type=Path, default=TABLES)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU.split(',')[0]

    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    ran = False
    if args.only in ('both', 'vindr'):
        if args.vindr_checkpoint.is_file():
            score_vindr(args)
            ran = True
        else:
            print('VinDr-CXR checkpoint not found: %s' % args.vindr_checkpoint)
    if args.only in ('both', 'ricord'):
        if args.ricord_checkpoint.is_file():
            score_ricord(args)
            ran = True
        else:
            print('RICORD checkpoint not found: %s' % args.ricord_checkpoint)
    if not ran:
        return 1
    print('figures in %s' % args.figures)
    print('tables in %s' % args.tables)
    return 0


if __name__ == '__main__':
    sys.exit(main())
