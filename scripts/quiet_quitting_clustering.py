import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from math import pi
import textwrap
from scipy.stats import zscore

# --- Load dataset ---
# Default to Excel file in repository root
file_path = "Quiet_Quitting_In_Academia.xlsx"
df = pd.read_excel(file_path)

# --- Inspect, select numeric features ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Drop obvious timestamp-like numeric columns if present
numeric_cols = [c for c in numeric_cols if "timestamp" not in c.lower() and "time" not in c.lower()]
df_numeric = df[numeric_cols].copy()

# If Likert-style columns are strings, map common labels to numbers (example mapping)
likert_map = {
    "Strongly disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly agree": 5,
    "1":1, "2":2, "3":3, "4":4, "5":5
}
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().astype(str).head(20).tolist()
        joined = " ".join(sample).lower()
        if any(x in joined for x in ["agree","disagree","strongly","neutral"]):
            df[col] = df[col].map(likert_map).astype(float)
# refresh numeric selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if "timestamp" not in c.lower()]
df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop constant columns
const_cols = [c for c in df_numeric.columns if df_numeric[c].nunique() <= 1]
df_numeric = df_numeric.drop(columns=const_cols)

# --- Scale features ---
scaler = StandardScaler()
X = scaler.fit_transform(df_numeric)

# --- PCA for visualization ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# --- Choose k using silhouette (k=2..7) ---
inertias, sil_scores = [], []
K_range = range(2,8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labs = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labs))
best_k = K_range[int(np.argmax(sil_scores))]
print("Best k by silhouette:", best_k)

# --- Final KMeans & assign clusters ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50)
labels = kmeans.fit_predict(X)
df['cluster'] = labels

# --- Visualizations ---
# ensure outputs directory exists
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
# PCA scatter
plt.figure(figsize=(8,6))
for c in np.unique(labels):
    mask = labels == c
    plt.scatter(X_pca[mask,0], X_pca[mask,1], label=f"Cluster {c}", s=30)
plt.legend(); plt.title("PCA projection of clusters"); plt.grid(True); plt.savefig(os.path.join(outputs_dir, "pca_clusters.png"), dpi=150, bbox_inches="tight"); plt.show()

# t-SNE scatter (costly but informative)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, init='pca')
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(8,6))
for c in np.unique(labels):
    mask = labels == c
    plt.scatter(X_tsne[mask,0], X_tsne[mask,1], label=f"Cluster {c}", s=30)
plt.legend(); plt.title("t-SNE projection of clusters"); plt.grid(True); plt.savefig(os.path.join(outputs_dir, "tsne_clusters.png"), dpi=150, bbox_inches="tight"); plt.show()

# --- Cluster profiling ---
cluster_profiles = df.groupby('cluster')[df_numeric.columns].mean()
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_profiles.to_csv(os.path.join(outputs_dir, "cluster_profiles.csv"))

# --- Personas: z-score within each cluster to identify salient highs/lows ---
profiles_z_values = zscore(cluster_profiles, axis=1) # Calculate z-scores across features for each cluster
personas = {}
for i, c in enumerate(cluster_profiles.index):
    row_z = profiles_z_values[i] # Get the z-score array for the current cluster
    high_feats = cluster_profiles.columns[row_z > 0.5].tolist()
    low_feats = cluster_profiles.columns[row_z < -0.5].tolist()
    personas[c] = {
        "count": int(cluster_counts.loc[c]),
        "high_features": high_feats,
        "low_features": low_feats,
        "summary": f"High on {', '.join(high_feats)[:120]} | Low on {', '.join(low_feats)[:120]}"
    }
pd.DataFrame.from_dict(personas, orient='index').to_csv(os.path.join(outputs_dir, "personas_summary.csv"))


# --- Radar chart helper (plot up to top 6 variance features) ---
variances = df_numeric.var().sort_values(ascending=False)
top_features = variances.head(6).index.tolist()

# Mapping for compact attribute names; extend as needed
ATTRIBUTE_RENAMES = {
    "On average, how long does it take you to provide feedback on major student assignments?": "Time Taken to provide feedback",
    # Add more exact mappings below as needed:
    # "Full verbose question here": "Short Label",
}

def compact_label(name: str) -> str:
    """Return a compact display label for a verbose attribute name."""
    if name in ATTRIBUTE_RENAMES:
        return ATTRIBUTE_RENAMES[name]
    short = name.strip().replace("?", "")
    # Common phrase simplifications
    replacements = [
        ("On average, ", ""),
        ("How long", "Time"),
        ("how long", "Time"),
        ("provide feedback", "provide feedback"),
        ("major student assignments", "major assignments"),
        ("students", "students"),
        ("student", "student"),
        ("Work-Life Balance", "Work-Life Balance"),
        ("work life", "Work-Life"),
        ("work-life", "Work-Life"),
        ("satisfaction", "satisfaction"),
    ]
    for a, b in replacements:
        short = short.replace(a, b)
    # Collapse multiple spaces and cap length
    short = " ".join(short.split())
    # Wrap to multiple short lines for compact display
    wrapped = "\n".join(textwrap.wrap(short, width=16, break_long_words=False, break_on_hyphens=True))
    return wrapped

def make_radar(values, categories, title):
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    vals = values.tolist()
    vals += vals[:1]
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles[:-1])
    compact_categories = [compact_label(c) for c in categories]
    ax.set_xticklabels(compact_categories)
    ax.tick_params(axis='x', labelsize=7, pad=2)
    ax.tick_params(axis='y', labelsize=7)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    # Two-line compact title: move sample size to new line, smaller font
    compact_title = title.replace(" (n=", "\n(n=")
    ax.set_title(compact_title, fontsize=9, pad=8)
    plt.savefig(os.path.join(outputs_dir, f"radar_cluster_{title.split()[1]}.png"), dpi=150, bbox_inches="tight"); plt.show()

for c in cluster_profiles.index:
    make_radar(cluster_profiles.loc[c, top_features], top_features, f"Cluster {c} (n={cluster_counts.loc[c]})")

# --- Heuristic risk scoring for quiet quitting ---
# Identify columns with names matching engagement/protection vs burnout/risk
score_indicators = {}
for c in cluster_profiles.index:
    score = 0.0
    for col in cluster_profiles.columns:
        col_l = col.lower()
        # Check if the column name contains keywords related to engagement/protection
        if any(k in col_l for k in ['engag','related','commit','support','satisf','connected','valued', 'competence', 'autonomy', 'belonging']):
            score += cluster_profiles.loc[c,col]
        # Check if the column name contains keywords related to burnout/risk
        if any(k in col_l for k in ['burnout','stress','overload','work life','work-life', 'manageable']):
             score -= cluster_profiles.loc[c,col] # Subtract for risk indicators
    score_indicators[c] = score

vals = np.array(list(score_indicators.values()))
if vals.max() != vals.min():
    norm = (vals - vals.min()) / (vals.max() - vals.min())
else:
    norm = np.zeros_like(vals)
risk_scores = {c: 1 - norm[i] for i,c in enumerate(score_indicators.keys())}
print("Risk scores per cluster (0 low -> 1 high):", risk_scores)

# --- Save labeled dataset ---
df.to_csv(os.path.join(outputs_dir, "Quiet_Quitting_Clusters_250_labeled.csv"), index=False)


