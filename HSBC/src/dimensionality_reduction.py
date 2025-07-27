import pandas as pd
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

def perform_PCA(df: pd.DataFrame):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_result,columns=['PC1','PC2'])

    explained_variance_ratio  = pca.explained_variance_ratio_

    #Plotting PCA result
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.scatter(pca_df['PC1'],pca_df['PC2'],alpha=0.5)
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    
    plt.subplot(1,2,2)
    plt.bar(['PC1','PC2'],explained_variance_ratio,alpha=0.8)
    plt.title('Explained Variance Ratio')
    plt.ylabel('Variance Ratio')
    plt.ylim(0,1.2)
    plt.tight_layout()
    plt.show()