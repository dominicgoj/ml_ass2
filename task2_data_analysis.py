import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
import itertools
EXPORT_FOLDER = "exports"

def read_csv_file(path):
    df = pd.read_csv(path)
    return df

def plot_features(df, out_path="feature_plot.png"):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    features = [c for c in df.columns if c not in ('id', 'label')]

    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(72, 72), squeeze=False)
    
    for i, yi in enumerate(features):
        for j, xj in enumerate(features):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[yi].dropna(), bins=30, color='lightgray')
            else:
                ax.scatter(df[xj], df[yi], s=2, alpha=0.5)
            
            if i == n-1:
                ax.set_xlabel(xj, fontsize=60, rotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yi, fontsize=60)
            else:
                ax.set_yticklabels([])
            
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_FOLDER,out_path), dpi=300)
    plt.close()

def plot_class_distribution(df, label_col='label', out_path='class_distribution.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    if label_col not in df.columns:
        print(f"Spalte '{label_col}' nicht gefunden.")
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
    print(counts)

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

def plot_feature_boxplots_by_class(df, id_col='id', label_col='label', out_path='feature_boxplots_by_class.png'):
    """
    Boxplots für jedes Feature, gruppiert nach Klassen.
    x-Achse: Klassen, y-Achse: Feature-Werte
    """
    if label_col not in df.columns:
        print(f"Spalte '{label_col}' nicht gefunden.")
        return
    
    features = [c for c in df.columns if c not in (id_col, label_col)]
    
    n_features = len(features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        # Prepare data for boxplot: [values for class0, values for class1, ...]
        classes = sorted(df[label_col].unique())
        data_by_class = [df[df[label_col] == cls][feature].dropna().values for cls in classes]
        
        ax.boxplot(data_by_class, labels=classes, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='red', linewidth=1.5))
        ax.set_xlabel('Class')
        ax.set_ylabel('Value')
        ax.set_title(f'{feature}')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    fig.suptitle('Feature Distributions by Class (Boxplots)', fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Feature boxplots saved to: {out_path}")

def cluster_feature_data(df, id_col='id', label_col='label', n_clusters=3):
    feature_cols = [col for col in df.columns if col not in [id_col, label_col]]
    X = df[feature_cols].values
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    X_norm = df[feature_cols].copy()
    for c in range(n_clusters):
        mask = (clusters == c)
        X_norm.loc[mask] = (X_norm.loc[mask] - X_norm.loc[mask].mean()) / X_norm.loc[mask].std()

    df_norm = df.copy()
    df_norm[feature_cols] = X_norm

    df_norm['cluster']= clusters
    return df_norm

def plot_pca(df_norm, id_col='id', label_col='label', cluster_col='cluster', out_path='pca.png'):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    # Extract features
    feature_cols = [col for col in df_norm.columns if col not in [id_col, label_col, cluster_col]]
    X = df_norm[feature_cols].values
    y = df_norm[label_col].values
    clusters = df_norm[cluster_col].values

    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot PCA by class
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    for cls in sorted(df_norm[label_col].unique()):
        mask = (y == cls)
        plt.scatter(X_pca[mask,0], X_pca[mask,1], s=10, label=f"Class {cls}", alpha=0.6)
    plt.title("PCA of Normalized Data (colored by class)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    # Plot PCA by cluster (to verify domain removal)
    plt.subplot(1,2,2)
    for c in sorted(df_norm[cluster_col].unique()):
        mask = (clusters == c)
        plt.scatter(X_pca[mask,0], X_pca[mask,1], s=10, label=f"Cluster {c}", alpha=0.6)
    plt.title("PCA of Normalized Data (colored by cluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)

def plot_raw_clusters(df, id_col='id', label_col='label', n_clusters=3, out_path='pca_cluster_before_norm.png'):
    # Extract feature matrix
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values
    y = df[label_col].values

    # Fit KMeans on full data (this is only for visualization!)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    clusters = kmeans.predict(X)

    # PCA on raw data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12,5))

    # Plot clusters
    plt.subplot(1,2,1)
    for c in range(n_clusters):
        mask = (clusters == c)
        plt.scatter(X_pca[mask,0], X_pca[mask,1], s=8, label=f"Cluster {c}", alpha=0.6)
    plt.title("Raw Data PCA — colored by KMeans cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    # Plot classes
    plt.subplot(1,2,2)
    for cls in sorted(set(y)):
        mask = (y == cls)
        plt.scatter(X_pca[mask,0], X_pca[mask,1], s=8, label=f"Class {cls}", alpha=0.6)
    plt.title("Raw Data PCA — colored by class")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_FOLDER,out_path), dpi=150)

def plot_normalized_clusters(X_norm, clusters, out_path='pca_cluster_after_norm.png'):
    # X_norm ist DataFrame mit nur Features
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    X = X_norm.values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6,5))

    for c in sorted(set(clusters)):
        mask = (clusters == c)
        plt.scatter(X_pca[mask,0], X_pca[mask,1], s=8, label=f"Cluster {c}", alpha=0.6)
    plt.title("PCA of normalized data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(os.path.join(EXPORT_FOLDER,out_path), dpi=150)


def plot_boxplots_by_class(df, id_col="id", label_col="label", out_path="boxplots_by_class.png"):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    # Determine feature columns
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    # Layout: 3 rows × 4 columns (12 features)
    n_features = len(feature_cols)
    n_rows = 3
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()

    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(data=df, x=label_col, y=feature, ax=ax)
        ax.set_title(f"{feature} by class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Value")

    # If more subplots are available than features, hide them
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_FOLDER, out_path), dpi=150)
    plt.close(fig)

    print(f"Saved boxplot grid to {os.path.join(EXPORT_FOLDER, out_path)}")

def plot_wcss(df, id_col='id', label_col='label', k_range=range(1, 11)):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values

    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # inertia_ = WCSS

    plt.figure(figsize=(6,4))
    plt.plot(k_range, wcss, marker='o')
    plt.xlabel("Number of clusters K")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method for KMeans")
    plt.grid(True)
    plt.savefig(os.path.join(EXPORT_FOLDER, 'wcss_score.png'))


def plot_silhouette(df, id_col='id', label_col='label', k_range=range(2, 11)):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values

    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    plt.figure(figsize=(6,4))
    plt.plot(k_range, scores, marker='o')
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for KMeans")
    plt.grid(True)
    plt.savefig(os.path.join(EXPORT_FOLDER, 'silhouette_score.png'))


if __name__ == '__main__':
    df = read_csv_file('D.csv')
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_feature_correlation(df)
    plot_pca_by_label(df)
    plot_kmeans_pca(df)
    #df_norm = cluster_feature_data(df)
    #plot_wcss(df)
    #plot_silhouette(df)
    #plot_pca(df_norm)
    #plot_boxplots_by_class(df)