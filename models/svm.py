# models/svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.base import Model


class SVMModel(Model):
    """SVM classifier wrapping sklearn SVC."""

    def __init__(self, kernel='rbf', C=1.0, random_state=42, debug=False):
        super().__init__(debug)
        self.kernel = kernel
        self.C = C
        self.random_state = random_state

    def train(self, X, y):
        try:
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                random_state=self.random_state,
                probability=True
            )
            self.model.fit(X, y)
            self.is_trained = True
            self._debug_print("Model trained successfully")
        except Exception as e:
            raise Exception(f"Training failed: {e}")

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model not trained yet")
        try:
            return self.model.predict(X)
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")

    def evaluate(self, X, y):
        if not self.is_trained:
            raise Exception("Model not trained yet")
        try:
            preds = self.predict(X)
            return {
                'accuracy': accuracy_score(y, preds),
                'f1_score': f1_score(y, preds, average='weighted'),
                'confusion_matrix': confusion_matrix(y, preds),
            }
        except Exception as e:
            raise Exception(f"Evaluation failed: {e}")

    def plot_confusion_matrix(self, cm, title="Confusion Matrix", save_path=None):
        try:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            im = ax.imshow(cm, interpolation='nearest', cmap='Greens', aspect='auto', vmin=0)
            fig.colorbar(im, ax=ax)
            ax.set_title(title, color="#e6edf3", pad=10)
            ax.set_xlabel('Predicted Label', color="#8b949e")
            ax.set_ylabel('True Label', color="#8b949e")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            n = cm.shape[0]
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            thresh = cm.max() / 2
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                            fontsize=14, fontweight='bold',
                            color="white" if cm[i, j] > thresh else "#e6edf3")

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=140, bbox_inches='tight', facecolor="#0d1117")
                plt.close(fig)
            else:
                plt.show()
        except Exception as e:
            self._debug_print(f"Failed to plot: {e}")