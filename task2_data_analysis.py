import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.cluster import KMeans
EXPORT_FOLDER = "exports/task2"

def read_csv_file(path):
    df = pd.read_csv(path)
    return df

def plot_class_distribution(df, label_col='label', out_path='class_distribution.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    if label_col not in df.columns:
        print(f"Column '{label_col}' not found.")
        return
    
    counts = df[label_col].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Class distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)
    plt.close(fig)
    print(f"Saved class distribution to: {out_path}")

def plot_feature_distributions(df, id_col='id', label_col='label', out_path='feature_distributions.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    features = [c for c in df.columns if c not in (id_col, label_col)]
    
    n_features = len(features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df[feature].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature}')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    fig.suptitle('Feature Distributions', fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(EXPORT_FOLDER,out_path), dpi=150)
    plt.close(fig)
    print(f"Feature distributions saved to: {out_path}")


def plot_feature_correlation(df, id_col='id', label_col='label',
                             out_path='feature_correlation.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    features = [c for c in df.columns if c not in (id_col, label_col)]
    
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)
    plt.close(fig)
    print(f"Feature correlation matrix saved to: {out_path}")

def plot_pca_by_label(df, id_col='id', label_col='label',
                      out_path='pca_by_label.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    features = [c for c in df.columns if c not in (id_col, label_col)]
    
    X = df[features].values
    y = df[label_col].values

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in np.unique(y):
        idx = y == label
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   s=8, alpha=0.5, label=f'Class {label}')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA projection colored by class label')
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)
    plt.close(fig)
    print(f"PCA plot saved to: {out_path}")

def plot_kmeans_pca(df, id_col='id', label_col='label',
                    out_path='kmeans_pca.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    
    features = [c for c in df.columns if c not in (id_col, label_col)]
    X = df[features].values

    # Standardisierung (wichtig für k-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k-means im Originalraum
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA nur zur Visualisierung
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in np.unique(clusters):
        idx = clusters == c
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=8,
            alpha=0.5,
            label=f'Cluster {c}'
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('k-means clustering (k=3) visualized in PCA space')
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)
    plt.close(fig)

    print(f"k-means PCA plot saved to: {out_path}")

def task2():
    print(30*"*", "TASK2", 30*"*")
    df = read_csv_file('D.csv')
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_feature_correlation(df)
    plot_pca_by_label(df)
    plot_kmeans_pca(df)

if __name__ == '__main__':
    task2()