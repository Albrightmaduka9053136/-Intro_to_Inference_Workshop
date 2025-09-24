import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO
import base64

class ClimateAnomalyAnalyzer:
    def __init__(self, mu, sigma, X):
        self.mu = mu
        self.sigma = sigma
        self.X = X

    def compute_zscore(self):
        return (self.X - self.mu) / self.sigma

    def compute_probabilities(self):
        p_less = norm.cdf(self.X, self.mu, self.sigma)
        p_greater = 1 - p_less
        return {"P(X ≤ x)": p_less, "P(X > x)": p_greater}

    def plot_distribution(self):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 200)
        y = norm.pdf(x, self.mu, self.sigma)

        plt.figure(figsize=(6,4))
        plt.plot(x, y, label="Normal Distribution")
        plt.fill_between(x, 0, y, where=(x >= self.X), color='red', alpha=0.5, label="P(X > x)")
        plt.axvline(self.X, color='red', linestyle="--", label=f"X = {self.X}")
        plt.legend()
        plt.title("Climate Anomaly Probability")
        plt.xlabel("X")
        plt.ylabel("Density")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return encoded
