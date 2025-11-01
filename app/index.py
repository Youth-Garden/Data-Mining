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

# Professional styling - minimal and clean
st.markdown(
    """
    <style>
    html { scroll-behavior: smooth; }
    .anchor { scroll-margin-top: 80px; }
    /* Giao diện tối giản, không card */
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .stMetric { background-color: transparent; padding: 0; border: none; }
    .stExpander { border: none; }
    .stPlotlyChart { border: none; }
    .stDataFrame { border: none; }
    h1, h2, h3 { color: #1f1f1f; margin-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Sidebar ===
st.sidebar.title("Glass Data Mining Demo")
st.sidebar.markdown("### Mục lục")
st.sidebar.markdown(
    """
    - [Giới thiệu](#gioi-thieu)
    - [1. Khảo sát dữ liệu](#khao-sat-du-lieu)
    - [2. Trực quan hóa](#truc-quan-hoa)
    - [3. Phân loại](#phan-loai)
    - [4. Gom cụm](#gom-cum)
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
    1. Khảo sát và xử lý dữ liệu: kích thước, kiểu dữ liệu, phân bố nhãn, thống kê.
    2. Trực quan hóa: giảm chiều (PCA, t-SNE) và biểu diễn phân bố.
    3. Phân loại: so sánh KNN, Random Forest, SVM bằng 10-fold CV (F1 Macro).
    4. Gom cụm: K-Means và DBSCAN, đánh giá bằng F1 Macro, ARI, NMI.
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

st.markdown("---")

with st.expander("Kết quả từ Notebook", expanded=False):
    if st.button("Chạy Notebook", key="run_notebook"):
        try:
            with st.spinner("Đang chạy notebook..."):
                out_nb = run_notebook("main.ipynb", "outputs/main_executed.ipynb", kernel_name=None)
                try:
                    html_preview = notebook_to_html(out_nb)
                except Exception:
                    html_preview = None
            st.success("Notebook đã chạy xong")
        
            if html_preview:
                with st.expander("Xem trước chi tiết notebook (HTML)", expanded=False):
                    st.components.v1.html(html_preview, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Lỗi khi chạy notebook: {e}\nCài đặt: pip install nbformat nbclient nbconvert")
    else:
        st.info("Nhấn nút 'Chạy Notebook' để thực thi")

st.markdown('<div id="khao-sat-du-lieu" class="anchor"></div>', unsafe_allow_html=True)
st.header("1. Khảo sát dữ liệu")

# Bảng dữ liệu
st.subheader("Bảng dữ liệu")
st.dataframe(df, use_container_width=True, height=400)

# Thông tin và thống kê - 2 cột bằng nhau
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin chung")
    info_data = pd.DataFrame({
        'Thuộc tính': ['Số mẫu', 'Số thuộc tính', 'Số lớp (Type)'],
        'Giá trị': [int(df.shape[0]), int(df.shape[1]), int(df["Type"].nunique())]
    })
    st.dataframe(info_data, use_container_width=True, hide_index=True, height=150)
    
    st.subheader("**Kiểu dữ liệu**")
    dtype_data = pd.DataFrame({
        'Cột': df.dtypes.index.tolist(),
        'Kiểu': [str(dt) for dt in df.dtypes.values]
    })
    st.dataframe(dtype_data, use_container_width=True, hide_index=True, height=300)

with col2:
    st.subheader("Phân bố nhãn")
    type_counts = df["Type"].value_counts().sort_index()
    st.bar_chart(type_counts, use_container_width=True, height=150)
    
    st.subheader("Thống kê mô tả")
    st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True, height=300)

st.markdown('<div id="truc-quan-hoa" class="anchor"></div>', unsafe_allow_html=True)
st.header("2. Trực quan hóa")

label_palette = sns.color_palette("Set2", n_colors=int(df["Type"].nunique()))

# PCA - tự động chạy
with st.spinner("Tính toán PCA..."):
    X_pca, pca = run_pca(X_scaled)

col1, col2 = st.columns(2)

with col1:
    st.subheader("PCA 2D")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, t in enumerate(sorted(df["Type"].unique())):
        mask = df["Type"] == t
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=70,
            label=f"Type {t}",
            alpha=0.75,
            edgecolors="#555555",
            linewidths=0.4,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=11)
    ax.set_title("Biểu đồ PCA", fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("t-SNE 2D")
    with st.spinner("Tính toán t-SNE..."):
        X_tsne = run_tsne(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, t in enumerate(sorted(df["Type"].unique())):
        mask = df["Type"] == t
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=70,
            label=f"Type {t}",
            alpha=0.75,
            edgecolors="#555555",
            linewidths=0.4,
        )
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title("Biểu đồ t-SNE", fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('<div id="phan-loai" class="anchor"></div>', unsafe_allow_html=True)
st.header("3. Phân loại (Classification)")

col_cf1, col_cf2 = st.columns([4, 1])
with col_cf1:
    cv_k = st.slider("KFold splits", min_value=3, max_value=10, value=10, step=1, key="cv_slider")
with col_cf2:
    st.write("")  # Spacer
    run_classification_btn = st.button("Chạy", key="run_classification")

# Chỉ chạy khi nhấn nút
if run_classification_btn:
    with st.spinner("Đang huấn luyện các mô hình..."):
        progress = st.progress(0)
        def _cb(i, total):
            progress.progress(min(int(i / total * 100), 99))
        results = run_classification(df, cv_k=cv_k, progress_callback=_cb)
        progress.progress(100)
        st.session_state['classification_results'] = results

# Hiển thị kết quả nếu có
if 'classification_results' in st.session_state:
    st.success("Phân loại hoàn thành")
    st.subheader("Kết quả")
    st.dataframe(st.session_state['classification_results'], width="stretch")

st.markdown('<div id="gom-cum" class="anchor"></div>', unsafe_allow_html=True)
st.header("4. Gom cụm (Clustering)")

col_cluster1, col_cluster2 = st.columns(2)

# K-MEANS - Cột trái
with col_cluster1:
    st.subheader("K-Means")
    
    n_clusters_kmeans = st.slider("Số cụm", min_value=2, max_value=10, value=6, step=1, key="kmeans_clusters")
    run_kmeans = st.button("Chạy", key="btn_kmeans")
    
    # Container cố định cho metrics và plot
    result_container_km = st.container()
    
    # Chỉ chạy khi nhấn nút
    if run_kmeans:
        with st.spinner("Đang chạy K-Means..."):
            kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=20)
            km_labels = kmeans.fit_predict(X_scaled)
            km_eval = evaluate_clustering(y, km_labels)
            
            # Visualization
            X_pca, _ = run_pca(X_scaled)
            
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="tab10", 
                                s=60, alpha=0.7, edgecolors="#555555", linewidths=0.5)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.set_title("K-Means Clustering", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            cbar = plt.colorbar(scatter, ax=ax, label="Cụm", fraction=0.046, pad=0.04)
            fig.tight_layout()
            
            # Lưu vào session state
            st.session_state['kmeans_eval'] = km_eval
            st.session_state['kmeans_fig'] = fig
    
    # Hiển thị kết quả trong container
    with result_container_km:
        if 'kmeans_eval' in st.session_state:
            st.success("K-Means hoàn thành")
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                st.metric("F1", f"{st.session_state['kmeans_eval']['F1_macro']:.3f}")
            with col_k2:
                st.metric("ARI", f"{st.session_state['kmeans_eval']['ARI']:.3f}")
            with col_k3:
                st.metric("NMI", f"{st.session_state['kmeans_eval']['NMI']:.3f}")
            
            st.pyplot(st.session_state['kmeans_fig'], use_container_width=False)

# DBSCAN - Cột phải
with col_cluster2:
    st.subheader("DBSCAN")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        eps_dbscan = st.slider("eps", min_value=0.3, max_value=2.0, value=1.0, step=0.1, key="dbscan_eps")
    with col_d2:
        min_samples_db = st.slider("min_samples", min_value=3, max_value=20, value=5, step=1, key="dbscan_min")
    
    run_dbscan = st.button("Chạy", key="btn_dbscan")
    
    # Container cố định cho metrics và plot
    result_container_db = st.container()
    
    # Chỉ chạy khi nhấn nút
    if run_dbscan:
        with st.spinner("Đang chạy DBSCAN..."):
            db = DBSCAN(eps=eps_dbscan, min_samples=min_samples_db)
            db_labels = db.fit_predict(X_scaled)
            db_eval = evaluate_clustering(y, db_labels)
            n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise = list(db_labels).count(-1)
            
            # Visualization - CÙNG KÍCH THƯỚC với K-Means
            X_pca, _ = run_pca(X_scaled)
            
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="tab10",
                                s=60, alpha=0.7, edgecolors="#555555", linewidths=0.5)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.set_title("DBSCAN Clustering", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            cbar = plt.colorbar(scatter, ax=ax, label="Cụm", fraction=0.046, pad=0.04)
            fig.tight_layout()
            
            # Lưu vào session state
            st.session_state['dbscan_eval'] = db_eval
            st.session_state['dbscan_clusters'] = n_clusters_db
            st.session_state['dbscan_noise'] = n_noise
            st.session_state['dbscan_fig'] = fig
    
    # Hiển thị kết quả trong container
    with result_container_db:
        if 'dbscan_eval' in st.session_state:
            st.success("DBSCAN hoàn thành")
            col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
            with col_d1:
                st.metric("F1", f"{st.session_state['dbscan_eval']['F1_macro']:.3f}")
            with col_d2:
                st.metric("ARI", f"{st.session_state['dbscan_eval']['ARI']:.3f}")
            with col_d3:
                st.metric("NMI", f"{st.session_state['dbscan_eval']['NMI']:.3f}")
            with col_d4:
                st.metric("Cụm", st.session_state['dbscan_clusters'])
            with col_d5:
                st.metric("Nhiễu", st.session_state['dbscan_noise'])
            
            st.pyplot(st.session_state['dbscan_fig'], use_container_width=False)
