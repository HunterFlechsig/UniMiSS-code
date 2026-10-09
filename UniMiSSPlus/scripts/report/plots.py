"""Report figures. Matplotlib is imported when a figure is drawn."""

from common import BIT_KEYS, BIT_NAMES


def _matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis='y', color='#dddddd', linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches='tight')
    figure.clf()
    import matplotlib.pyplot as plt
    plt.close(figure)


def plot_vindr_macro(rows, checkpoint, path):
    plt = _matplotlib()
    epochs = [row['epoch'] for row in rows]
    scores = [row['macro_auc'] for row in rows]
    figure, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(epochs, scores, color='#0072B2', marker='o', linewidth=1.6, label='Headline ROC-AUC')
    if checkpoint is not None:
        chosen = next(row for row in rows if row['epoch'] == checkpoint)
        ax.axvline(checkpoint, color='#333333', linestyle='--', linewidth=1.0, label='Checkpoint (epoch %d)' % checkpoint)
        ax.scatter([checkpoint], [chosen['macro_auc']], color='#D55E00', zorder=3, s=36)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('VinDr-CXR validation, center crop')
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    save_figure(figure, path)


def plot_vindr_bits(rows, checkpoint, path):
    plt = _matplotlib()
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), sharex=True, sharey=True)
    for ax, name, key in zip(axes.ravel(), BIT_NAMES, BIT_KEYS):
        ax.plot(epochs, [row[key + '_auc'] for row in rows], color='#0072B2', marker='o', linewidth=1.4)
        if checkpoint is not None:
            ax.axvline(checkpoint, color='#333333', linestyle='--', linewidth=1.0)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0.0, 1.0)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel('Epoch')
    for ax in axes[:, 0]:
        ax.set_ylabel('ROC-AUC')
    figure.suptitle('VinDr-CXR validation ROC-AUC by bit', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, path)


def plot_ricord_validation(rows, checkpoint, path):
    plt = _matplotlib()
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 7.0), sharex=True)
    axes[0].plot(epochs, [row['auc'] for row in rows], color='#0072B2', marker='o', linewidth=1.6, label='ROC-AUC')
    axes[1].plot(epochs, [row['sensitivity'] for row in rows], color='#0072B2', marker='o', linewidth=1.4, label='Sensitivity')
    axes[1].plot(epochs, [row['specificity'] for row in rows], color='#D55E00', marker='o', linewidth=1.4, label='Specificity')
    if checkpoint is not None:
        for ax in axes:
            ax.axvline(checkpoint, color='#333333', linestyle='--', linewidth=1.0, label='Checkpoint (epoch %d)' % checkpoint)
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        _style(ax)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('ROC-AUC')
    axes[0].set_title('RICORD validation')
    axes[1].set_ylabel('Rate')
    axes[1].set_xlabel('Epoch')
    figure.tight_layout()
    save_figure(figure, path)


def plot_vindr_roc(labels, probabilities, path):
    plt = _matplotlib()
    from sklearn import metrics
    import numpy as np

    colors = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9')
    figure, ax = plt.subplots(figsize=(6.2, 5.4))
    mean_fpr = np.linspace(0, 1, 201)
    mean_tpr = []
    for bit, (name, key, color) in enumerate(zip(BIT_NAMES, BIT_KEYS, colors)):
        y_true = labels[:, bit]
        if y_true.min() == y_true.max():
            continue
        fpr, tpr, _ = metrics.roc_curve(y_true, probabilities[:, bit])
        auc = metrics.roc_auc_score(y_true, probabilities[:, bit])
        ax.plot(fpr, tpr, color=color, linewidth=1.5, label='%s (%.3f)' % (name, auc))
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        mean_tpr.append(interpolated)
    if mean_tpr:
        macro_tpr = np.mean(mean_tpr, axis=0)
        macro_tpr[-1] = 1.0
        headline = float(np.mean([
            metrics.roc_auc_score(labels[:, bit], probabilities[:, bit])
            for bit in range(labels.shape[1])
            if labels[:, bit].min() != labels[:, bit].max()
        ]))
        ax.plot(mean_fpr, macro_tpr, color='#000000', linewidth=2.0, label='Macro-average (%.3f)' % headline)
    ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=1.0, linestyle=':')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title('VinDr-CXR test, ten-crop')
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    figure.tight_layout()
    save_figure(figure, path)


def plot_vindr_pr(labels, probabilities, path):
    plt = _matplotlib()
    from sklearn import metrics

    figure, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), sharex=True, sharey=True)
    for ax, bit, name in zip(axes.ravel(), range(labels.shape[1]), BIT_NAMES):
        y_true = labels[:, bit]
        prevalence = float(y_true.mean())
        ax.axhline(prevalence, color='#888888', linestyle='--', linewidth=1.0, label='Positive rate')
        if y_true.min() != y_true.max():
            precision, recall, _ = metrics.precision_recall_curve(y_true, probabilities[:, bit])
            ap = metrics.average_precision_score(y_true, probabilities[:, bit])
            ax.plot(recall, precision, color='#0072B2', linewidth=1.5, label='Model (AP %.3f)' % ap)
            ax.set_title('%s\nAP %.3f' % (name, ap), fontsize=11)
        else:
            ax.set_title('%s\nundefined' % name, fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('Recall')
    for ax in axes[:, 0]:
        ax.set_ylabel('Precision')
    figure.suptitle('VinDr-CXR test precision-recall, ten-crop', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, path)


def plot_ricord_roc(labels, probabilities, path):
    plt = _matplotlib()
    from sklearn import metrics

    figure, ax = plt.subplots(figsize=(6.2, 5.4))
    if labels.min() != labels.max():
        fpr, tpr, _ = metrics.roc_curve(labels, probabilities)
        auc = metrics.roc_auc_score(labels, probabilities)
        ax.plot(fpr, tpr, color='#0072B2', linewidth=1.8, label='Category 1 (%.3f)' % auc)
    ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=1.0, linestyle=':')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title('RICORD test')
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    figure.tight_layout()
    save_figure(figure, path)
