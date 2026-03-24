from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Model(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.model = None
        self.is_trained = False
    
    def load_data(self, filepath, target_column=-1):
        """
        Load CSV and encode ALL non-numeric columns to integers.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        target_column : int or str, default=-1
            - int: column index (0-based) for target
            - str: column name if header exists
            - -1: use last column (default)
        
        Returns:
        --------
        X : numpy array
            Feature matrix
        y : numpy array
            Target labels (integers)
        """
        try:
            # Read everything as string first to avoid mixed type issues
            df = pd.read_csv(filepath, dtype=str)
            self._debug_print(f"Loaded {filepath}, shape: {df.shape}")

            # Drop rows with any missing values
            before = len(df)
            df = df.dropna()
            if before - len(df) > 0:
                self._debug_print(f"Dropped {before - len(df)} rows with missing values")

            # Determine target column
            if isinstance(target_column, int):
                target_idx = target_column
            elif isinstance(target_column, str):
                target_idx = df.columns.get_loc(target_column)
            else:
                target_idx = -1

            # Store target name for later
            target_name = df.columns[target_idx]
            self._debug_print(f"Target column: {target_name} (index {target_idx})")

            # Encode ALL columns
            for col in df.columns:
                try:
                    df[col].astype(float)
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._debug_print(
                        f"Encoded column '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}"
                    )

            # Separate features and target
            X = df.drop(columns=[target_name]).values.astype(float)
            y = df[target_name].values.astype(int)

            self._debug_print(f"X shape: {X.shape}, unique classes: {np.unique(y)}")
            return X, y

        except FileNotFoundError:
            raise Exception(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    @abstractmethod
    def train(self, X, y):
        """Train model on given data"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict labels for given features"""
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """Return accuracy and F1 score"""
        pass
    
    def _debug_print(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")