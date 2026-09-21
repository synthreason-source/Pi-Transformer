from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

class TestCasePredictor:
    """
    Small feed-forward classifier implemented from scratch with NumPy.

    Input:
        X: shape (samples, features)

    Output:
        One probability for every test case/class.

    Example classes:
        0 = test case A
        1 = test case B
        2 = test case C
    """

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

        # He/Xavier-style small initialization
        self.W1 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / input_dim),
            size=(input_dim, hidden_dim),
        )
        self.b1 = np.zeros(hidden_dim)

        self.W2 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_dim),
            size=(hidden_dim, output_dim),
        )
        self.b2 = np.zeros(output_dim)

        # Feature normalization statistics
        self.mean = np.zeros(input_dim)
        self.std = np.ones(input_dim)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Your custom probability step
    # ------------------------------------------------------------------

    @staticmethod
    def step(
        p_t: np.ndarray,
        temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Converts raw neural-network scores into probabilities.

        This preserves the structure of your supplied function:

            raw = p_t
            condition = exp(max(raw, eps)) >= cos(p_t)
            logits = where(condition, exp(max(raw, eps)), raw / 2)

            scaled = logits / temperature
            shifted = scaled - max(scaled)
            exp_s = exp(shifted)
            p_next = exp_s / sum(exp_s)

        The implementation supports both one vector and a batch.
        """

        raw = np.asarray(p_t, dtype=np.float64)

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        if raw.ndim == 1:
            condition = (
                np.exp(np.maximum(raw, eps))
                >= np.cos(raw)
            )

            logits = np.where(
                condition,
                np.exp(np.maximum(raw, eps)),
                raw / 2.0,
            )

            scaled = logits / temperature
            shifted = scaled - np.max(scaled)
            exp_s = np.exp(shifted)
            return exp_s / np.sum(exp_s)

        if raw.ndim == 2:
            condition = (
                np.exp(np.maximum(raw, eps))
                >= np.cos(raw)
            )

            logits = np.where(
                condition,
                np.exp(np.maximum(raw, eps)),
                raw / 2.0,
            )

            scaled = logits / temperature
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            return exp_s / np.sum(exp_s, axis=1, keepdims=True)

        raise ValueError("p_t must have shape (classes,) or (samples, classes)")

    # ------------------------------------------------------------------
    # Ordinary training softmax
    # ------------------------------------------------------------------

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
        return (x > 0.0).astype(np.float64)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        logits = a1 @ self.W2 + self.b2

        # Standard softmax is used during training because its gradient
        # combines cleanly with cross-entropy.
        probabilities = self.softmax(logits)

        cache = (X, z1, a1, logits, probabilities)
        return probabilities, cache

    # ------------------------------------------------------------------
    # Loss and backpropagation
    # ------------------------------------------------------------------

    @staticmethod
    def cross_entropy(
        probabilities: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        sample_count = len(y)
        selected = probabilities[np.arange(sample_count), y]
        return float(-np.mean(np.log(np.maximum(selected, eps))))

    def backward(
        self,
        cache,
        y: np.ndarray,
    ):
        X, z1, a1, logits, probabilities = cache
        batch_size = X.shape[0]

        # Derivative of softmax + cross-entropy
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

    def update(
        self,
        gradients,
        learning_rate: float,
    ):
        dW1, db1, dW2, db2 = gradients

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.03,
        batch_size: int = 16,
        validation_fraction: float = 0.2,
        verbose_every: int = 100,
        missing_rate: float = 0.2, # Added parameter to simulate missing data
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features)")

        if y.ndim != 1:
            raise ValueError("y must have shape (samples,)")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        if np.min(y) < 0 or np.max(y) >= self.output_dim:
            raise ValueError("labels are outside the output class range")

        # Ignore NaNs when calculating statistics
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)

        # Impute NaNs with feature means before normalizing
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std

        indices = self.rng.permutation(len(X))
        split = int(len(X) * (1.0 - validation_fraction))

        train_idx = indices[:split]
        valid_idx = indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": [],
        }

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(X_train))

            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]

                xb = X_train[batch_indices]
                yb = y_train[batch_indices]

                # Drop out inputs to teach the model to guess from partial data
                if missing_rate > 0.0:
                    mask = self.rng.binomial(1, 1.0 - missing_rate, size=xb.shape)
                    xb = xb * mask

                _, cache = self.forward(xb)
                gradients = self.backward(cache, yb)
                self.update(gradients, learning_rate)

            train_probs, _ = self.forward(X_train)
            train_loss = self.cross_entropy(train_probs, y_train)
            train_accuracy = np.mean(
                np.argmax(train_probs, axis=1) == y_train
            )

            if len(X_valid):
                valid_probs, _ = self.forward(X_valid)
                valid_loss = self.cross_entropy(valid_probs, y_valid)
                valid_accuracy = np.mean(
                    np.argmax(valid_probs, axis=1) == y_valid
                )
            else:
                valid_loss = train_loss
                valid_accuracy = train_accuracy

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(float(train_accuracy))
            history["valid_loss"].append(valid_loss)
            history["valid_accuracy"].append(float(valid_accuracy))

            if verbose_every and (
                epoch == 1 or epoch % verbose_every == 0
            ):
                print(
                    f"epoch {epoch:5d} | "
                    f"train loss {train_loss:.5f} | "
                    f"train acc {train_accuracy:.3f} | "
                    f"valid loss {valid_loss:.5f} | "
                    f"valid acc {valid_accuracy:.3f}"
                )

        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def raw_probabilities(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Impute missing values with the historical mean before scaling
        X = np.where(np.isnan(X), self.mean, X)
        X = (X - self.mean) / self.std
        
        probabilities, _ = self.forward(X)
        return probabilities

    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Applies your custom step to each output vector.

        Training uses normal softmax/cross-entropy.
        Inference uses the supplied custom transformation.
        """
        raw = self.raw_probabilities(X)

        transformed = np.vstack([
            self.step(row, temperature=temperature)
            for row in raw
        ])

        return transformed

    def predict(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X, temperature)
        return np.argmax(probabilities, axis=1)

    def predict_one(
        self,
        x: np.ndarray,
        class_names=None,
        temperature: float = 1.0,
    ):
        probabilities = self.predict_proba(
            np.asarray(x).reshape(1, -1),
            temperature=temperature,
        )[0]

        predicted_index = int(np.argmax(probabilities))

        result = {
            "class_index": predicted_index,
            "probabilities": probabilities.tolist(),
        }

        if class_names is not None:
            result["class"] = class_names[predicted_index]

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, class_names=None):
        path = Path(path)

        np.savez(
            path.with_suffix(".npz"),
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            mean=self.mean,
            std=self.std,
        )

        metadata = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "class_names": class_names,
        }

        path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str):
        path = Path(path)

        metadata = json.loads(
            path.read_text(encoding="utf-8")
        )

        model = cls(
            input_dim=metadata["input_dim"],
            hidden_dim=metadata["hidden_dim"],
            output_dim=metadata["output_dim"],
        )

        weights = np.load(path.with_suffix(".npz"))

        model.W1 = weights["W1"]
        model.b1 = weights["b1"]
        model.W2 = weights["W2"]
        model.b2 = weights["b2"]
        model.mean = weights["mean"]
        model.std = weights["std"]

        return model, metadata.get("class_names")


# ----------------------------------------------------------------------
# Test case
# ----------------------------------------------------------------------

def make_demo_dataset(seed=7):
    """
    Artificial test-case data.

    Features:
        [input_value, noise_level, signal_strength]

    Classes:
        0 = low-risk test
        1 = medium-risk test
        2 = high-risk test
    """

    rng = np.random.default_rng(seed)

    samples_per_class = 150

    low = np.column_stack([
        rng.normal(0.2, 0.12, samples_per_class),
        rng.normal(0.2, 0.10, samples_per_class),
        rng.normal(0.2, 0.12, samples_per_class),
    ])

    medium = np.column_stack([
        rng.normal(0.55, 0.12, samples_per_class),
        rng.normal(0.55, 0.10, samples_per_class),
        rng.normal(0.55, 0.12, samples_per_class),
    ])

    high = np.column_stack([
        rng.normal(0.85, 0.12, samples_per_class),
        rng.normal(0.80, 0.10, samples_per_class),
        rng.normal(0.85, 0.12, samples_per_class),
    ])

    X = np.vstack([low, medium, high])
    y = np.concatenate([
        np.zeros(samples_per_class, dtype=np.int64),
        np.ones(samples_per_class, dtype=np.int64),
        np.full(samples_per_class, 2, dtype=np.int64),
    ])

    order = rng.permutation(len(X))
    return X[order], y[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="test_case_model.json")
    args = parser.parse_args()

    class_names = [
        "low-risk test",
        "medium-risk test",
        "high-risk test",
    ]

    X, y = make_demo_dataset()

    model = TestCasePredictor(
        input_dim=X.shape[1],
        hidden_dim=16,
        output_dim=len(class_names),
        seed=42,
    )

    model.fit(
        X,
        y,
        epochs=1200,
        learning_rate=0.03,
        batch_size=32,
        validation_fraction=0.2,
        verbose_every=100,
        missing_rate=0.2, # Tell the model to learn to guess
    )

    model.save(args.model, class_names)

    print("\nExample predictions with missing data (np.nan):\n")

    # Added np.nan to simulate missing inputs
    test_cases = np.array([
        [0.10, 0.20, 0.15],      # 100% complete data
        [0.55, np.nan, 0.60],    # Missing the middle feature
        [np.nan, 0.85, np.nan],  # Missing two features
    ])

    for i, test_case in enumerate(test_cases, start=1):
        result = model.predict_one(
            test_case,
            class_names=class_names,
            temperature=1.0,
        )

        print(f"test case {i}")
        print(f"  input:       {test_case}")
        print(f"  prediction:  {result['class']}")
        print(
            "  probabilities:",
            {
                name: round(probability, 4)
                for name, probability in zip(
                    class_names,
                    result["probabilities"],
                )
            },
        )
        print()


if __name__ == "__main__":
    main()
