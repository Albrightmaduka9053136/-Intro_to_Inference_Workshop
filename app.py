from flask import Flask, request, jsonify, render_template_string
from climate_analyzer import ClimateAnomalyAnalyzer

app = Flask(__name__)

# --- Helper to extract parameters safely ---
def _get_params(req):
    mu    = float(req.args.get("mu", 0.5))
    sigma = float(req.args.get("sigma", 0.2))
    X     = float(req.args.get("X", 0.9))
    return mu, sigma, X

# --- Helper to build result dictionary ---
def _build_result(mu, sigma, X):
    a = ClimateAnomalyAnalyzer(mu, sigma, X)
    return {
        "mu": mu,
        "sigma": sigma,
        "X": X,
        "z_score": a.compute_zscore(),
        "probabilities": a.compute_probabilities(),
        "plot": f"data:image/png;base64,{a.plot_distribution()}",
    }

# --- JSON endpoint ---
@app.route("/analyze", methods=["GET"])
def analyze():
    mu, sigma, X = _get_params(request)
    return jsonify(_build_result(mu, sigma, X))

# --- HTML template ---
HTML = """
<!doctype html>
<title>Climate Anomaly Analyzer</title>
<h1>Results</h1>
<ul>
  <li>μ: {{ mu }}</li>
  <li>σ: {{ sigma }}</li>
  <li>X: {{ X }}</li>
  <li>Z-score: {{ z }}</li>
  <li>P(X ≤ x): {{ p_le }}</li>
  <li>P(X > x): {{ p_gt }}</li>
</ul>
<img src="{{ img }}" alt="distribution plot">
"""

# --- HTML endpoint ---
@app.route("/analyze_html", methods=["GET"])
def analyze_html():
    mu, sigma, X = _get_params(request)
    a = ClimateAnomalyAnalyzer(mu, sigma, X)
    probs = a.compute_probabilities()
    return render_template_string(
        HTML,
        mu=mu, sigma=sigma, X=X,
        z=a.compute_zscore(),
        p_le=probs["P(X ≤ x)"],
        p_gt=probs["P(X > x)"],
        img=f"data:image/png;base64,{a.plot_distribution()}",
    )

# --- Entry point ---
if __name__ == "__main__":
    app.run(debug=True)
