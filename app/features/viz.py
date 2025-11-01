import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@st.cache_data(show_spinner=False)
def run_pca(X_scaled: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


@st.cache_data(show_spinner=False)
def run_tsne(
    X_scaled: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne
