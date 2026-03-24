import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves files instead of popping windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.decision_tree import DecisionTreeModel
from models.svm import SVMModel
from sklearn.model_selection import train_test_split

# ── Where to save generated images ────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def asset(name):
    """Return full path for an asset file, overwriting any previous version."""
    return os.path.join(ASSETS_DIR, name)


# ══════════════════════════════════════════════════════════════════════════════
# Shared dark-theme style
# ══════════════════════════════════════════════════════════════════════════════
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
ACCENT1 = "#58a6ff"   # blue   → Decision Tree
ACCENT2 = "#3fb950"   # green  → SVM
ACCENT3 = "#f78166"   # coral
ACCENT4 = "#d2a8ff"   # purple → abstract / base
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   CARD,
        'axes.edgecolor':   BORDER,
        'axes.labelcolor':  TEXT,
        'xtick.color':      SUBTEXT,
        'ytick.color':      SUBTEXT,
        'text.color':       TEXT,
        'grid.color':       BORDER,
        'grid.linestyle':   '--',
        'grid.alpha':       0.5,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CLASS HIERARCHY  (architecture.png)
#     Shows the OOP structure: Model ABC → DecisionTreeModel, SVMModel
#     Each box lists its actual attributes and methods
# ══════════════════════════════════════════════════════════════════════════════
def generate_architecture():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis('off')

    # ── helper: draw a UML-style class box ────────────────────────────────────
    def uml_box(cx, cy, class_name, stereotype, attrs, methods, color):
        """
        cx, cy  = centre of the box
        attrs   = list of attribute strings  (shown in middle section)
        methods = list of method strings     (shown in bottom section)
        """
        bw = 4.4
        # row heights
        header_h = 0.55
        attr_h   = max(len(attrs),   1) * 0.36 + 0.15
        meth_h   = max(len(methods), 1) * 0.36 + 0.15
        total_h  = header_h + attr_h + meth_h

        x0 = cx - bw / 2
        y0 = cy - total_h / 2   # bottom-left corner

        # outer border
        outer = mpatches.FancyBboxPatch(
            (x0, y0), bw, total_h,
            boxstyle="round,pad=0,rounding_size=0.18",
            linewidth=2, edgecolor=color, facecolor=CARD, zorder=3)
        ax.add_patch(outer)

        # header fill
        hdr = mpatches.FancyBboxPatch(
            (x0, y0 + attr_h + meth_h), bw, header_h,
            boxstyle="round,pad=0,rounding_size=0.18",
            linewidth=0, facecolor=color, alpha=0.22, zorder=4)
        ax.add_patch(hdr)

        # stereotype  e.g. «abstract»
        ax.text(cx, y0 + attr_h + meth_h + header_h - 0.16,
                stereotype, ha='center', va='center',
                fontsize=7.5, color=color, style='italic', zorder=5)
        # class name
        ax.text(cx, y0 + attr_h + meth_h + 0.2,
                class_name, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color,
                fontfamily='monospace', zorder=5)

        # divider lines
        for dy in [attr_h + meth_h, meth_h]:
            ax.plot([x0, x0 + bw], [y0 + dy, y0 + dy],
                    color=BORDER, lw=1, zorder=4)

        # attributes
        for i, txt in enumerate(attrs):
            ax.text(x0 + 0.18, y0 + meth_h + attr_h - 0.12 - i * 0.36,
                    txt, ha='left', va='center',
                    fontsize=8, color=SUBTEXT,
                    fontfamily='monospace', zorder=5)

        # methods
        for i, txt in enumerate(methods):
            ax.text(x0 + 0.18, y0 + meth_h - 0.12 - i * 0.36,
                    txt, ha='left', va='center',
                    fontsize=8, color=TEXT,
                    fontfamily='monospace', zorder=5)

        # return bottom-centre and top-centre for arrows
        return (cx, y0), (cx, y0 + total_h)

    # ── draw the three classes ─────────────────────────────────────────────────

    # Model (ABC)  — top centre
    _, top_model = uml_box(
        7, 7.0,
        "Model",
        "«abstract»",
        attrs=[
            "+ debug: bool = False",
            "+ model = None",
            "+ is_trained: bool = False",
        ],
        methods=[
            "+ load_data(filepath) → X, y  «abstract»",
            "+ train(X, y)                 «abstract»",
            "+ predict(X) → y_pred         «abstract»",
            "+ evaluate(X, y) → dict       «abstract»",
            "# _debug_print(msg)",
        ],
        color=ACCENT4,
    )
    bot_model = (7, 7.0 - (0.55 + 3*0.36+0.15 + 5*0.36+0.15) / 2)

    # DecisionTreeModel  — bottom left
    bot_dt, top_dt = uml_box(
        3.2, 2.9,
        "DecisionTreeModel",
        "«concrete»",
        attrs=[
            "+ max_depth = None",
            "+ random_state: int = 42",
        ],
        methods=[
            "+ __init__(max_depth, random_state, debug)",
            "+ load_data(filepath) → X, y",
            "+ train(X, y)",
            "+ predict(X) → y_pred",
            "+ evaluate(X, y) → dict",
            "+ plot_confusion_matrix(cm, title)",
        ],
        color=ACCENT1,
    )

    # SVMModel  — bottom right
    bot_svm, top_svm = uml_box(
        10.8, 2.9,
        "SVMModel",
        "«concrete»",
        attrs=[
            "+ kernel: str = 'rbf'",
            "+ C: float = 1.0",
            "+ random_state: int = 42",
        ],
        methods=[
            "+ __init__(kernel, C, random_state, debug)",
            "+ load_data(filepath) → X, y",
            "+ train(X, y)",
            "+ predict(X) → y_pred",
            "+ evaluate(X, y) → dict",
            "+ plot_confusion_matrix(cm, title)",
        ],
        color=ACCENT2,
    )

    # ── inheritance arrows (hollow triangle head = UML generalisation) ─────────
    def inheritance_arrow(child_top, parent_bottom):
        cx, cy = child_top
        px, py = parent_bottom
        # shaft
        ax.annotate(
            '', xy=(px, py), xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle='->', color=BORDER, lw=1.8,
                connectionstyle='arc3,rad=0'),
            zorder=2)
        ax.text((cx + px) / 2 + 0.15, (cy + py) / 2,
                'inherits', fontsize=8, color=SUBTEXT,
                style='italic', ha='left', va='center')

    # approximate top of child boxes
    dt_top_y  = top_dt[1]
    svm_top_y = top_svm[1]
    base_bot_y = bot_model[1]

    inheritance_arrow((3.2,  dt_top_y),  (7, base_bot_y))
    inheritance_arrow((10.8, svm_top_y), (7, base_bot_y))

    # ── sklearn dependencies (dashed notes) ───────────────────────────────────
    def dep_note(cx, cy, text, color):
        rect = mpatches.FancyBboxPatch(
            (cx - 1.3, cy - 0.28), 2.6, 0.56,
            boxstyle="round,pad=0,rounding_size=0.12",
            linewidth=1.2, edgecolor=color, facecolor=CARD,
            linestyle='dashed', zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=8, color=color, fontfamily='monospace', zorder=4)

    dep_note(3.2, 0.5, "DecisionTreeClassifier", ACCENT1)
    ax.annotate('', xy=(3.2, 0.78), xytext=(3.2, bot_dt[1]),
                arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.2,
                                linestyle='dashed'), zorder=2)

    dep_note(10.8, 0.5, "SVC", ACCENT2)
    ax.annotate('', xy=(10.8, 0.78), xytext=(10.8, bot_svm[1]),
                arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.2,
                                linestyle='dashed'), zorder=2)

    ax.text(7, 8.7, 'Class Architecture', ha='center', fontsize=14,
            fontweight='bold', color=TEXT)
    ax.text(7, 8.35, 'models/base.py  ·  models/decision_tree.py  ·  models/svm.py',
            ha='center', fontsize=9, color=SUBTEXT)

    # legend
    handles = [
        mpatches.Patch(color=ACCENT4, label='Abstract base class'),
        mpatches.Patch(color=ACCENT1, label='DecisionTreeModel (concrete)'),
        mpatches.Patch(color=ACCENT2, label='SVMModel (concrete)'),
        mpatches.Patch(color=BORDER,  label='sklearn dependency', linestyle='--'),
    ]
    ax.legend(handles=handles, loc='lower center', ncol=4,
              framealpha=0.15, labelcolor=TEXT, fontsize=8,
              bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout()
    path = asset("architecture.png")
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓ saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ACCURACY COMPARISON  (accuracy_comparison.png)
#     Built from real results collected during main()
# ══════════════════════════════════════════════════════════════════════════════
def generate_accuracy_chart(all_results):
    """
    all_results: list of dicts with keys:
        dataset, model, train_ratio, accuracy, f1
    """
    apply_dark_style()

    datasets = sorted(set(r['dataset'] for r in all_results))
    splits   = sorted(set(r['train_ratio'] for r in all_results))
    model_names = sorted(set(r['model'] for r in all_results))
    colors = {'Decision Tree': ACCENT1, 'SVM': ACCENT2}

    n_splits = len(splits)
    x = np.arange(n_splits)
    w = 0.35

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                             facecolor=BG, sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle('Accuracy by Train/Test Split', fontsize=14,
                 fontweight='bold', color=TEXT, y=1.02)

    for ax, ds in zip(axes, datasets):
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

        offsets = np.linspace(-(len(model_names)-1)*w/2,
                               (len(model_names)-1)*w/2,
                               len(model_names))

        for offset, mname in zip(offsets, model_names):
            vals = []
            for sp in splits:
                matches = [r['accuracy'] for r in all_results
                           if r['dataset'] == ds and r['model'] == mname
                           and r['train_ratio'] == sp]
                vals.append(matches[0] if matches else 0.0)

            bars = ax.bar(x + offset, vals, w * 0.92,
                          label=mname, color=colors.get(mname, ACCENT3),
                          alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.004,
                            f'{v:.3f}', ha='center', va='bottom',
                            fontsize=8, color=SUBTEXT)

        ax.set_xticks(x)
        ax.set_xticklabels([f'Train {int(s*100)}%' for s in splits], fontsize=9)
        ymin = max(0.0, min(r['accuracy'] for r in all_results
                             if r['dataset'] == ds) - 0.08)
        ax.set_ylim(ymin, 1.04)
        ax.set_ylabel('Accuracy', color=SUBTEXT)
        ax.set_title(ds, color=TEXT, fontsize=12, pad=10)
        ax.legend(framealpha=0.2, labelcolor=TEXT, fontsize=9)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    path = asset("accuracy_comparison.png")
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓ saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONFUSION MATRICES  (confusion_matrices.png)
#     One sub-plot per (dataset × model) — uses the last split's matrix
# ══════════════════════════════════════════════════════════════════════════════
def generate_confusion_matrices(all_results):
    """
    all_results: list of dicts; each has 'confusion_matrix' as np.ndarray
    We pick the highest train_ratio result for each (dataset, model) pair.
    """
    apply_dark_style()

    # pick last split per (dataset, model)
    best = {}
    for r in all_results:
        key = (r['dataset'], r['model'])
        if key not in best or r['train_ratio'] > best[key]['train_ratio']:
            best[key] = r

    items = list(best.values())
    n = len(items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4.5 * nrows),
                              facecolor=BG, squeeze=False)
    fig.suptitle(f'Confusion Matrices  (Train {int(max(r["train_ratio"] for r in items)*100)}% split)',
                 fontsize=13, fontweight='bold', color=TEXT, y=1.01)

    model_cmaps = {'Decision Tree': 'Blues', 'SVM': 'Greens'}

    for idx, item in enumerate(items):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        cm = item['confusion_matrix']
        cmap = model_cmaps.get(item['model'], 'Purples')

        ax.set_facecolor(CARD)
        ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, zorder=2)
        ax.set_title(f"{item['model']}  ·  {item['dataset']}",
                     color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel('Predicted', color=SUBTEXT, fontsize=9)
        ax.set_ylabel('Actual',    color=SUBTEXT, fontsize=9)

        n_cls = cm.shape[0]
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels([f'Class {i}' for i in range(n_cls)], fontsize=8)
        ax.set_yticklabels([f'Class {i}' for i in range(n_cls)], fontsize=8)

        thresh = cm.max() / 2
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold', zorder=3,
                        color='white' if cm[i, j] > thresh else TEXT)

        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    # hide any unused axes
    for idx in range(len(items), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = asset("confusion_matrices.png")
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓ saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(model_instance, X, y, train_ratio, dataset_name, model_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print(f"Train: {train_ratio*100:.0f}% | Test: {(1-train_ratio)*100:.0f}%")
    print('='*60)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_ratio, random_state=42, stratify=y
        )
        print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

        model_instance.train(X_train, y_train)
        results = model_instance.evaluate(X_test, y_test)

        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])

        return {
            'dataset':          dataset_name,
            'model':            model_name,
            'train_ratio':      train_ratio,
            'accuracy':         results['accuracy'],
            'f1':               results['f1_score'],
            'confusion_matrix': results['confusion_matrix'],
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    datasets = {
        "Iris (2 classes)": "data/Iris2classes.csv",
        "Breast Cancer":    "data/wdbc.data.csv",
    }

    model_configs = {
        "Decision Tree": lambda: DecisionTreeModel(max_depth=5, debug=False),
        "SVM":           lambda: SVMModel(kernel='rbf', C=1.0, debug=False),
    }

    train_ratios = [0.3, 0.5, 0.7]
    all_results  = []

    for dataset_name, filepath in datasets.items():
        if not os.path.exists(filepath):
            print(f"\n⚠️  File not found: {filepath} — skipping")
            continue

        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*60}")

        for model_name, model_factory in model_configs.items():
            try:
                # fresh instance per (dataset × model)
                instance = model_factory()
                X, y = instance.load_data(filepath)
                print(f"\nLoaded {dataset_name}: {len(X)} samples, {X.shape[1]} features")

                for ratio in train_ratios:
                    # fresh instance per split so weights never bleed over
                    inst = model_factory()
                    inst.load_data(filepath)   # re-init internal state if any
                    result = run_experiment(inst, X, y, ratio, dataset_name, model_name)
                    if result:
                        all_results.append(result)

            except Exception as e:
                print(f"\nERROR with {model_name} on {dataset_name}: {e}")

    # ── Generate / overwrite all visual assets ─────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("Generating visual assets → assets/")
        print('='*60)
        generate_accuracy_chart(all_results)
        generate_confusion_matrices(all_results)

    # Architecture diagram is dataset-independent — always regenerate
    generate_architecture()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ML MODEL EVALUATION — DECISION TREE vs SVM")
    print("="*60)
    main()
    print("\n" + "="*60)
    print("DONE")
    print("="*60)