from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
import json
import os
from datetime import datetime
import numpy as np

def evaluate_clustering_models(models, X, X_pca):
    """Evaluate clustering models using various metrics"""
    results = {}
    
    for name, model in models.items():
        try:
            # Get cluster labels
            if hasattr(model, 'labels_'):
                labels = model.labels_
            elif hasattr(model, 'predict'):
                if name == 'kmeans_pca':
                    labels = model.predict(X_pca)
                else:
                    labels = model.predict(X)
            else:
                continue
            
            # Skip if all points are in one cluster or noise
            if len(np.unique(labels)) < 2:
                print(f"{name}: Skipped (insufficient clusters)")
                continue
            
            # Calculate metrics
            data_for_metrics = X_pca if name == 'kmeans_pca' else X
            
            silhouette = silhouette_score(data_for_metrics, labels)
            calinski_harabasz = calinski_harabasz_score(data_for_metrics, labels)
            
            results[name] = {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski_harabasz),
                'n_clusters': len(np.unique(labels[labels >= 0])),  # Exclude noise points (-1)
                'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
            }
            
            print(f"{name.upper()}:")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}")
            print(f"  Number of Clusters: {results[name]['n_clusters']}")
            if results[name]['n_noise_points'] > 0:
                print(f"  Noise Points: {results[name]['n_noise_points']}")
            print()
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    
    # Save results
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_size': len(X),
        'results': results
    }
    
    results_path = os.path.join(models_dir, 'clustering_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_path}")
    return results

def analyze_clusters(X_original, labels, feature_names):
    """Analyze cluster characteristics"""
    import pandas as pd
    
    df_analysis = pd.DataFrame(X_original, columns=feature_names)
    df_analysis['Cluster'] = labels
    
    # Remove noise points for analysis
    df_clean = df_analysis[df_analysis['Cluster'] >= 0]
    
    cluster_summary = df_clean.groupby('Cluster').agg({
        col: ['mean', 'std', 'count'] for col in feature_names
    }).round(2)
    
    print("Cluster Analysis:")
    print("="*50)
    for cluster in sorted(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster]
        print(f"\\nCluster {cluster} (n={len(cluster_data)}):")
        
        # Show top characteristics
        means = cluster_data[feature_names].mean()
        top_features = means.nlargest(5)
        
        print("Top characteristics:")
        for feature, value in top_features.items():
            print(f"  {feature}: {value:.2f}")
    
    return cluster_summary