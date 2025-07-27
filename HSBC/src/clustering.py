import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

def plot_elbow_method(df_pca: pd.DataFrame):
    wss= []

    for i in range(1,11):
        cls = KMeans(n_clusters=i,random_state=42)
        cls.fit(df_pca)
        wss.append(cls._inertia_)

    plt.plot(range(1,11),wss,marker='o')
    plt.title("Elbow Method for optimal K")
    plt.xlabel("No. of Clusters")
    plt.ylabel("Within-cluster sum of squares")
    plt.show()

def perform_clustering(df: pd.DataFrame, df_pca: pd.DataFrame):
    model = KMeans(n_clusters=3,random_state=42)
    model.fit_predict(df)
    clusters = model.labels_ 

    return model


def add_clusters(df: pd.DataFrame,df_pca: pd.DataFrame, model):
    clusters = model.labels_

    df_pca['Cluster'] = clusters
    df['Cluster'] = clusters
    return df

def save_clustered_data(df: pd.DataFrame, path='clustered_data.csv'):
    df.to_csv(path,index=False)

def save_model(model, path='kmeans_model.pkl'):
    joblib.dump(model, path) 


def load_model(path='kmeans_model.pkl'):
    model = joblib.load(path)
    return model 
