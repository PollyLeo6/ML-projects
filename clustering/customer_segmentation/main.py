import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_and_preprocess_data
from clustering_models import find_optimal_clusters, train_clustering_models
from model_evaluation import evaluate_clustering_models, analyze_clusters

def main():
    print("=== CUSTOMER SEGMENTATION PROJECT ===")
    
    # Load and preprocess data
    X_scaled, X_pca, X_original, scaler, pca, df = load_and_preprocess_data()
    
    print(f"Processed data shape: {X_scaled.shape}")
    print(f"PCA data shape: {X_pca.shape}")
    
    # Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    optimal_k, inertias, silhouette_scores = find_optimal_clusters(X_scaled)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train clustering models
    print(f"\nTraining clustering models with {optimal_k} clusters...")
    models = train_clustering_models(X_scaled, X_pca, optimal_k)
    
    # Evaluate models
    print("\nEvaluating clustering models:")
    results = evaluate_clustering_models(models, X_scaled, X_pca)
    
    # Find best model based on silhouette score
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['silhouette_score'])
        print(f"Best model: {best_model[0].upper()}")
        print(f"Silhouette Score: {best_model[1]['silhouette_score']:.4f}")
        
        # Analyze best model clusters
        best_model_obj = models[best_model[0]]
        if hasattr(best_model_obj, 'labels_'):
            labels = best_model_obj.labels_
        else:
            data_for_prediction = X_pca if best_model[0] == 'kmeans_pca' else X_scaled
            labels = best_model_obj.predict(data_for_prediction)
        
        print(f"\nAnalyzing {best_model[0]} clusters:")
        feature_names = ['Age', 'Income', 'Total_Spent', 'Total_Purchases', 'Total_Children',
                        'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                        'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases',
                        'NumStorePurchases', 'NumWebVisitsMonth']
        
        if len(feature_names) == X_original.shape[1]:
            analyze_clusters(X_original.values, labels, feature_names)

if __name__ == "__main__":
    main()