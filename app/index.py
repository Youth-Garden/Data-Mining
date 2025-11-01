import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN

# Local modules
from features.data import load_data, scale_continuous
from features.viz import run_pca, run_tsne
from features.clustering import evaluate_clustering
from features.modeling import run_classification
from features.utils import auto_table_height
from features.notebook_runner import run_notebook, notebook_to_html

st.set_page_config(page_title="Glass Data Mining Demo", layout="wide")
sns.set_style("whitegrid")
# Smooth scrolling for in-page anchor links
st.markdown(
    """
    <style>
    html { scroll-behavior: smooth; }
    .anchor { scroll-margin-top: 80px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Sidebar ===
st.sidebar.title("Glass Data Mining Demo")
st.sidebar.markdown("### Điều hướng")
st.sidebar.markdown(
    """
    - [Giới thiệu](#gioi-thieu)
    - [1) Khảo sát dữ liệu](#khao-sat-du-lieu)
    - [2) Trực quan hóa](#truc-quan-hoa)
    - [3) Phân loại](#phan-loai)
    - [4) Gom cụm](#gom-cum)
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Source: glass-data.csv")

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Không thể đọc dữ liệu: {e}")
    st.stop()

# Common derived data
X_cont, X_scaled = scale_continuous(df)
y = df["Type"].values

st.markdown('<div id="gioi-thieu" class="anchor"></div>', unsafe_allow_html=True)
st.title("Glass Data Mining Demo")
st.markdown(
    """
    Ứng dụng web trình diễn các bước khai phá dữ liệu trên bộ dữ liệu kính:
    - Khảo sát dữ liệu cơ bản (shape, dtype, thống kê, phân bố nhãn)
    - Trực quan hóa với PCA và t-SNE
    - Huấn luyện và so sánh các mô hình phân loại (KNN, Random Forest, SVM)
    - Gom cụm với K-Means và DBSCAN, kèm các độ đo F1, ARI, NMI
    """
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Số mẫu", df.shape[0])
with c2:
    st.metric("Số thuộc tính", df.shape[1])
with c3:
    st.metric("Số lớp (Type)", int(df["Type"].nunique()))

st.subheader("Đề bài")
st.markdown(
    """
    1) Khảo sát và xử lý dữ liệu: kích thước, kiểu dữ liệu, phân bố nhãn, thống kê.
    2) Trực quan hóa: giảm chiều (PCA, t-SNE) và biểu diễn phân bố.
    3) Phân loại: so sánh KNN, Random Forest, SVM bằng 10-fold CV (F1 Macro).
    4) Gom cụm: K-Means và DBSCAN, đánh giá bằng F1 Macro, ARI, NMI.
    """
)

st.subheader("Xuất/ tải dữ liệu")
col_a, col_b = st.columns(2)
with col_a:
    st.write("Tải bản CSV gốc (tab-separated)")
    try:
        with open("glass-data.csv", "rb") as f:
            st.download_button("Tải glass-data.csv", f, file_name="glass-data.csv", mime="text/tab-separated-values")
    except Exception:
        st.caption("Không tìm thấy glass-data.csv trong thư mục hiện tại.")
with col_b:
    st.write("Tải bản CSV đã chuẩn hóa tên cột")
    cleaned_csv = df.to_csv(index=False)
    st.download_button("Tải cleaned_glass.csv", cleaned_csv, file_name="cleaned_glass.csv", mime="text/csv")

st.subheader("Chạy Notebook (không sửa code)")
kernel = "python3"  # kernel mặc định thường là "python3"; có thể đổi nếu khác
if st.button("Chạy main.ipynb"):
    try:
        with st.spinner("Đang chạy notebook... (lần đầu có thể hơi lâu)"):
            out_nb = run_notebook("main.ipynb", "outputs/main_executed.ipynb", kernel_name=kernel)
            try:
                html_preview = notebook_to_html(out_nb)
            except Exception:
                html_preview = None
        st.success("Notebook đã chạy xong.")
        # Nút tải notebook đã chạy
        with open(out_nb, "rb") as fnb:
            st.download_button("Tải notebook đã chạy (ipynb)", fnb, file_name="main_executed.ipynb", mime="application/x-ipynb+json")
        # Xem trước HTML trong expander (nếu convert thành công)
        if html_preview:
            with st.expander("Xem trước kết quả notebook (HTML)", expanded=False):
                st.components.v1.html(html_preview, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Không thể chạy notebook: {e}\nVui lòng cài đặt thêm nbformat, nbclient, nbconvert (pip install nbformat nbclient nbconvert) rồi thử lại.")

st.markdown('<div id="khao-sat-du-lieu" class="anchor"></div>', unsafe_allow_html=True)
st.header("1) Khảo sát dữ liệu")

st.subheader("Bảng dữ liệu đầy đủ")
st.dataframe(df, width="stretch", height=auto_table_height(len(df)))
st.caption(f"Hiển thị toàn bộ {len(df)} dòng.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Thông tin chung")
    st.write("Shape:", df.shape)
    st.write("Kiểu dữ liệu:")
    st.write(df.dtypes)
st.write("Phân bố nhãn (Type):")
type_counts = df["Type"].value_counts().sort_index()
st.bar_chart(type_counts)
with c2:
    st.subheader("Thống kê mô tả (các cột số)")
    st.dataframe(df.describe().T, width="stretch")

st.markdown('<div id="truc-quan-hoa" class="anchor"></div>', unsafe_allow_html=True)
st.header("2) Trực quan hóa")

show_pca = st.checkbox("Hiển thị PCA 2D", value=True)
show_tsne = st.checkbox("Hiển thị t-SNE 2D (chậm hơn)", value=False)
label_palette = sns.color_palette("tab10", n_colors=int(df["Type"].nunique()))

if show_pca:
    X_pca, pca = run_pca(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, t in enumerate(sorted(df["Type"].unique())):
        mask = df["Type"] == t
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=60,
            label=f"Type {t}",
            c=[label_palette[idx]],
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA Visualization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

if show_tsne:
    with st.spinner("Đang chạy t-SNE, vui lòng đợi..."):
        X_tsne = run_tsne(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, t in enumerate(sorted(df["Type"].unique())):
        mask = df["Type"] == t
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=60,
            label=f"Type {t}",
            c=[label_palette[idx]],
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE Visualization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown('<div id="phan-loai" class="anchor"></div>', unsafe_allow_html=True)
st.header("3) Phân loại (Classification)")

st.info("Theo đề bài: dùng 10-fold Cross Validation (có thể giảm để chạy nhanh hơn).")
cv_k = st.slider("Số folds (KFold)", min_value=3, max_value=10, value=10, step=1)

if st.button("Chạy huấn luyện & so sánh mô hình"):
    with st.spinner("Đang huấn luyện mô hình..."):
        progress = st.progress(0)
        def _cb(i, total):
            progress.progress(int(i / total * 100))
        results = run_classification(df, cv_k=cv_k, progress_callback=_cb)
    st.subheader("Kết quả")
    st.dataframe(results, width="stretch")
    st.download_button("Tải kết quả (CSV)", results.to_csv(index=False), file_name="classification_results.csv", mime="text/csv")

st.markdown('<div id="gom-cum" class="anchor"></div>', unsafe_allow_html=True)
st.header("4) Gom cụm (Clustering)")

tab1, tab2 = st.tabs(["K-Means", "DBSCAN"])

with tab1:
    k_default = int(df["Type"].nunique())
    n_clusters = st.number_input("Số cụm (k)", min_value=2, max_value=10, value=k_default, step=1)
    if st.button("Chạy K-Means"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        km_labels = kmeans.fit_predict(X_scaled)
        km_eval = evaluate_clustering(y, km_labels)
        st.write("Đánh giá:")
        st.json(km_eval)

        X_pca, _ = run_pca(X_scaled)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="viridis", s=60)
        ax.set_title("KMeans Clusters (PCA 2D)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(sc, label="Cluster")
        st.pyplot(fig)

with tab2:
    eps = st.slider("eps", min_value=0.3, max_value=2.0, value=1.0, step=0.1)
    min_samples = st.slider("min_samples", min_value=3, max_value=20, value=5, step=1)
    if st.button("Chạy DBSCAN"):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = db.fit_predict(X_scaled)
        db_eval = evaluate_clustering(y, db_labels)
        st.write("Đánh giá:")
        st.json(db_eval)

        X_pca, _ = run_pca(X_scaled)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="viridis", s=60)
        ax.set_title("DBSCAN Clusters (PCA 2D)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(sc, label="Cluster")
        st.pyplot(fig)

    # So sánh nhanh cả hai thuật toán
    if st.button("Chạy cả hai & so sánh"):
        kmeans = KMeans(n_clusters=int(df["Type"].nunique()), random_state=42, n_init=20)
        km_labels = kmeans.fit_predict(X_scaled)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = db.fit_predict(X_scaled)
        km_eval = evaluate_clustering(y, km_labels)
        db_eval = evaluate_clustering(y, db_labels)
        compare_df = pd.DataFrame([
            ["K-Means", km_eval["F1_macro"], km_eval["ARI"], km_eval["NMI"]],
            ["DBSCAN", db_eval["F1_macro"], db_eval["ARI"], db_eval["NMI"]],
        ], columns=["Algorithm", "F1_macro", "ARI", "NMI"])
        st.subheader("So sánh K-Means vs DBSCAN")
        st.dataframe(compare_df, width="stretch")
        st.download_button("Tải so sánh (CSV)", compare_df.to_csv(index=False), file_name="clustering_compare.csv", mime="text/csv")
