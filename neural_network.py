from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class TestCasePredictor:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / input_dim),
            size=(input_dim, hidden_dim),
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_dim),
            size=(hidden_dim, output_dim),
        ).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Custom probability step
    # ------------------------------------------------------------------

    @staticmethod
    def step(
        p_t: np.ndarray,
        temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        raw = np.asarray(p_t, dtype=np.float32)

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        safe_raw = np.clip(raw, a_min=None, a_max=85.0)
        condition = np.exp(np.maximum(safe_raw, eps)) >= np.cos(raw)

        logits = np.where(
            condition,
            np.exp(np.maximum(safe_raw, eps)),
            raw / 2.0,
        )

        scaled = logits / temperature

        if raw.ndim == 1:
            shifted = scaled - np.max(scaled)
            exp_s = np.exp(shifted)
            return exp_s / np.sum(exp_s)
        
        if raw.ndim == 2:
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            return exp_s / np.sum(exp_s, axis=1, keepdims=True)

        raise ValueError("p_t must have shape (classes,) or (samples, classes)")

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Forward & Backward Pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        logits = a1 @ self.W2 + self.b2
        probabilities = self.softmax(logits)
        cache = (X, z1, a1, logits, probabilities)
        return probabilities, cache

    @staticmethod
    def cross_entropy(probabilities: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        sample_count = len(y)
        selected = probabilities[np.arange(sample_count), y]
        return float(-np.mean(np.log(np.maximum(selected, eps))))

    def backward(self, cache, y: np.ndarray):
        X, z1, a1, logits, probabilities = cache
        batch_size = X.shape[0]

        d_logits = probabilities.copy()
        d_logits[np.arange(batch_size), y] -= 1.0
        d_logits /= batch_size

        dW2 = a1.T @ d_logits
        db2 = np.sum(d_logits, axis=0)

        d_a1 = d_logits @ self.W2.T
        d_z1 = d_a1 * self.relu_derivative(z1)

        dW1 = X.T @ d_z1
        db1 = np.sum(d_z1, axis=0)

        return dW1, db1, dW2, db2

    def update(self, gradients, learning_rate: float):
        dW1, db1, dW2, db2 = gradients
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def _evaluate_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            probs, _ = self.forward(xb)
            batch_loss = self.cross_entropy(probs, yb)
            total_loss += batch_loss * len(xb)
            correct += np.sum(np.argmax(probs, axis=1) == yb)
        return total_loss / len(X), correct / len(X)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.03,
        batch_size: int = 256,
        validation_fraction: float = 0.2,
        verbose_every: int = 10,
        simulated_missing_rate: float = 0.2, # % of features to randomly drop during training
    ):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # 1. Calculate statistics ignoring existing NaNs
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std = np.where(self.std < 1e-12, 1.0, self.std).astype(np.float32)

        # 2. Impute NaNs with feature mean, then normalize
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std

        indices = self.rng.permutation(len(X))
        split = int(len(X) * (1.0 - validation_fraction))

        X_train, y_train = X[indices[:split]], y[indices[:split]]
        X_valid, y_valid = X[indices[split:]], y[indices[split:]]

        history = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(X_train))

            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]
                xb = X_train[batch_indices]
                yb = y_train[batch_indices]

                # Simulate incomplete data by randomly masking inputs to 0.0 (the mean)
                if simulated_missing_rate > 0.0:
                    mask = self.rng.binomial(1, 1.0 - simulated_missing_rate, size=xb.shape)
                    xb = xb * mask

                _, cache = self.forward(xb)
                gradients = self.backward(cache, yb)
                self.update(gradients, learning_rate)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0):
                train_loss, train_accuracy = self._evaluate_batches(X_train, y_train, batch_size)
                
                if len(X_valid):
                    valid_loss, valid_accuracy = self._evaluate_batches(X_valid, y_valid, batch_size)
                else:
                    valid_loss, valid_accuracy = train_loss, train_accuracy

                history["train_loss"].append(train_loss)
                history["train_accuracy"].append(train_accuracy)
                history["valid_loss"].append(valid_loss)
                history["valid_accuracy"].append(valid_accuracy)

                print(
                    f"epoch {epoch:4d} | "
                    f"train loss {train_loss:.4f} | train acc {train_accuracy:.3f} | "
                    f"valid loss {valid_loss:.4f} | valid acc {valid_accuracy:.3f}"
                )

        return history

    # ------------------------------------------------------------------
    # Prediction (Handles Incomplete Data)
    # ------------------------------------------------------------------

    def raw_probabilities(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Impute missing values (NaN) with the historical mean before normalizing
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std
        
        probabilities, _ = self.forward(X)
        return probabilities

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        raw = self.raw_probabilities(X)
        return self.step(raw, temperature=temperature)

    def predict(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        probabilities = self.predict_proba(X, temperature)
        return np.argmax(probabilities, axis=1)

    def predict_one(self, x: np.ndarray, class_names=None, temperature: float = 1.0):
        probabilities = self.predict_proba(np.asarray(x).reshape(1, -1), temperature=temperature)[0]
        predicted_index = int(np.argmax(probabilities))

        result = {
            "class_index": predicted_index,
            "probabilities": probabilities.tolist(), 
        }

        if class_names is not None:
            result["class"] = class_names[predicted_index]

        return result


# ----------------------------------------------------------------------
# Massive Test case with Missing Data Simulation
# ----------------------------------------------------------------------

def make_massive_demo_dataset(num_classes=2000, samples_per_class=200, num_features=500, seed=42):
    print(f"Generating dataset with {num_classes} classes, {num_features} features...")
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 3.0, size=(num_classes, num_features))

    X, y = [], []
    for class_idx in range(num_classes):
        samples = rng.normal(centroids[class_idx], 1.5, size=(samples_per_class, num_features))
        X.append(samples)
        y.append(np.full(samples_per_class, class_idx, dtype=np.int64))

    X = np.vstack(X)
    y = np.concatenate(y)
    order = rng.permutation(len(X))
    return X[order], y[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=2000)
    args = parser.parse_args()

    num_classes = args.classes
    class_names = [f"case_{i:04d}" for i in range(num_classes)]

    X, y = make_massive_demo_dataset(num_classes=num_classes, num_features=200)
    
    model = TestCasePredictor(
        input_dim=X.shape[1],
        hidden_dim=1024, 
        output_dim=num_classes,
        seed=42,
    )

    # simulated_missing_rate=0.2 forces the model to learn to guess with 20% of data missing
    model.fit(
        X, y,
        epochs=10,
        learning_rate=0.05,
        batch_size=256,
        validation_fraction=0.2,
        verbose_every=10,
        simulated_missing_rate=0.2 
    )

    print("\n--- Testing with Incomplete Data ---")
    
    rng = np.random.default_rng(99)
    test_indices = rng.choice(len(X), size=3, replace=False)
    
    for i, idx in enumerate(test_indices, start=1):
        true_label = y[idx]
        original_features = X[idx].copy()
        
        # DELIBERATELY DESTROY DATA: Set 50% of the features to NaN
        incomplete_features = original_features.copy()
        missing_mask = rng.choice([True, False], size=len(incomplete_features), p=[0.5, 0.5])
        incomplete_features[missing_mask] = np.nan
        
        missing_count = np.sum(np.isnan(incomplete_features))
        
        result = model.predict_one(
            incomplete_features,
            class_names=class_names,
            temperature=1.0,
        )

        probs = np.array(result["probabilities"])
        top_3_idx = np.argsort(probs)[-3:][::-1]
        
        print(f"\nTest Case {i} (Missing {missing_count}/{len(original_features)} features)")
        print(f"  Input array: {np.round(incomplete_features, 2)}")
        print(f"  True label:  {class_names[true_label]}")
        print(f"  Prediction:  {result['class']} {'(CORRECT)' if result['class'] == class_names[true_label] else '(INCORRECT)'}")
        print("  Top 3 probabilities:")
        for top_idx in top_3_idx:
            print(f"    {class_names[top_idx]}: {probs[top_idx]:.4f}")


if __name__ == "__main__":
    main()
