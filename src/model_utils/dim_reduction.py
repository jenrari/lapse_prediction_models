from matplotlib.pyplot import title
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def pca_2d(X_scaled, y):

    pca = PCA(n_components=2)
    # Calculating first two principal components
    model = pca.fit(X_scaled)
    pca_df = pd.DataFrame(
        data=model.components_,
        columns=X_scaled.columns,
        index=['PCA1', 'PCA2']
    )
    print("Components analysis", pca_df)
    print("Explained variance ratio for first two components: ", model.explained_variance_ratio_)
    print("Sum of explained variance ratio for first two components: ", sum(model.explained_variance_ratio_))
    x_pca = model.transform(X_scaled)
    # Convert x_pca to dataframe and adding target column
    pca_df = pd.DataFrame(
        data=x_pca,
        columns=['PC1', 'PC2'])

    pca_df['target'] = y

    # Creating the plot
    plt.figure()
    labels = {0: "0: no impago", 1: "1: impago"}
    for i in [0, 1]:
        plt.scatter(pca_df[pca_df['target'] == i]['PC1'],
                    pca_df[pca_df['target'] == i]['PC2'],
                    alpha=.8,
                    label = labels[i])

    plt.title('PCA')
    plt.legend(title="Clase")
    plt.show()


def tsne_3d(X_scaled,y):

    tsne = TSNE(n_components=3, perplexity=65, learning_rate='auto',
                init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Convert X_tsne to dataframe and adding target column
    tsne_df = pd.DataFrame(
        data=X_tsne,
        columns=['TSNE1', 'TSNE2', 'TSNE3']
    )

    print("TSNE", tsne_df)
    tsne_df['target'] = y

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in [0, 1]:
        subset = tsne_df[tsne_df['target'] == i].copy()
        ax.scatter(
            subset['TSNE1'],
            subset['TSNE2'],
            subset['TSNE3'],
            alpha=0.8,
            label=f"Clase {i}"
        )

    ax.set_title('t-SNE (3D)')
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.set_zlabel('TSNE3')
    ax.legend()
    plt.tight_layout()
    plt.show()


def tsne_2d(X_scaled,y):

    tsne = TSNE(n_components=2, perplexity=65, learning_rate='auto',
                init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    # Convert X_tsne to dataframe and adding target column
    tsne_df = pd.DataFrame(
        data=X_tsne,
        columns=['TSNE1', 'TSNE2']
    )
    print("TSNE", tsne_df)
    tsne_df['target'] = y

    plt.figure()
    labels = {0: "0: no impago", 1: "1: impago"}
    for i in [0, 1]:
        plt.scatter(tsne_df[tsne_df['target'] == i]['TSNE1'],
                    tsne_df[tsne_df['target'] == i]['TSNE2'],
                    label=labels[i],
                    alpha=.8)

    plt.title('T-distributed Stochastic Neighbor Embedding')
    plt.legend(title="Clase")
    plt.show()

