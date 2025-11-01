import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import subprocess

# Local modules
from features.data import load_data, scale_continuous
from features.viz import run_pca, run_tsne
from features.clustering import evaluate_clustering
from features.modeling import run_classification
from features.utils import auto_table_height
from features.notebook_runner import run_notebook, notebook_to_html

# === Cấu hình trang và CSS ===

st.set_page_config(
    page_title="Glass Data Mining Demo", 
    layout="wide",
    page_icon="📊" # Thêm icon chuyên nghiệp
)
sns.set_style("whitegrid")

def load_css(file_name: str) -> None:
    """
    Tải file CSS tùy chỉnh và chèn vào <head> của ứng dụng Streamlit.
    ...
    """
    try:
        css_path = Path(__file__).parent / file_name
        # Thêm encoding="utf-8" để đọc file
        with open(css_path, encoding="utf-8") as f:  # <--- ĐÃ SỬA
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file {file_name} tại {css_path}")

load_css("style.css")

def ensure_chromium_installed():
    """Ensure Playwright's Chromium is installed before using nbconvert[webpdf]."""
    try:
        from playwright._impl._driver import compute_driver_executable
        compute_driver_executable()
    except Exception:
        print("Installing Chromium for Playwright...")
        subprocess.run(["python", "-m", "playwright", "install", "chromium", "--with-deps"], check=True)

ensure_chromium_installed()


# === Caching (Lưu đệm) cho các hàm tốn tài nguyên ===

@st.cache_data
def get_data() -> pd.DataFrame:
    """
    Tải và cache dữ liệu thô từ file.

    Returns:
        pd.DataFrame: DataFrame đã được tải và (có thể) đã làm sạch.
    
    Raises:
        st.stop: Nếu không thể đọc được file dữ liệu.
    """
    try:
        df = load_data()
        return df
    except Exception as e:
        st.error(f"Không thể đọc dữ liệu: {e}")
        st.stop()

@st.cache_data
def get_scaled_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tách và chuẩn hóa dữ liệu từ DataFrame.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu thô.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - X_cont (np.ndarray): Các đặc trưng liên tục (chưa scale).
            - X_scaled (np.ndarray): Các đặc trưng liên tục (đã scale).
            - y (np.ndarray): Mảng chứa nhãn (Type).
    """
    X_cont, X_scaled = scale_continuous(df)
    y = df["Type"].values
    return X_cont, X_scaled, y

@st.cache_data
def get_pca(X_scaled: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """
    Chạy thuật toán PCA trên dữ liệu đã scale và cache kết quả.

    Args:
        X_scaled (np.ndarray): Dữ liệu đã được chuẩn hóa.

    Returns:
        Tuple[np.ndarray, PCA]:
            - np.ndarray: Dữ liệu đã giảm chiều (2D).
            - PCA: Đối tượng PCA đã fit.
    """
    return run_pca(X_scaled)

@st.cache_data
def get_tsne(X_scaled: np.ndarray) -> np.ndarray:
    """
    Chạy thuật toán t-SNE trên dữ liệu đã scale và cache kết quả.

    Args:
        X_scaled (np.ndarray): Dữ liệu đã được chuẩn hóa.

    Returns:
        np.ndarray: Dữ liệu đã giảm chiều (2D).
    """
    return run_tsne(X_scaled)


# === Các hàm hiển thị cho từng mục ===

def display_sidebar() -> None:
    """
    Hiển thị thanh sidebar điều hướng (mục lục) của ứng dụng.
    """
    st.sidebar.title("Mục lục")
    # st.sidebar.markdown("### Mục lục")
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

def display_introduction(df: pd.DataFrame) -> None:
    """
    Hiển thị phần Giới thiệu, Đề bài, các chỉ số tổng quan 
    và các nút tải dữ liệu.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu chính để lấy thông tin 
                           tổng quan (shape, nunique).
    """
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
        # Khởi tạo biến trong session_state nếu chưa có
        if "notebook_ran" not in st.session_state:
            st.session_state.notebook_ran = False
        if "executed_notebook" not in st.session_state:
            st.session_state.executed_notebook = None
        if "pdf_path" not in st.session_state:
            st.session_state.pdf_path = None
        if "html_preview" not in st.session_state:
            st.session_state.html_preview = None

        # Nút chạy notebook
        if st.button("Chạy Notebook", key="run_notebook"):
            try:
                with st.spinner("Đang chạy notebook..."):
                    out_nb = run_notebook(
                        "main.ipynb",
                        "outputs/main_executed.ipynb",
                        kernel_name=None
                    )

                    # Tạo HTML preview
                    try:
                        html_preview = notebook_to_html(out_nb)
                    except Exception:
                        html_preview = None

                    # Tạo PDF
                    pdf_path = None
                    try:
                        from features.notebook_runner import notebook_to_pdf
                        pdf_path = notebook_to_pdf(out_nb, "outputs/main_executed.pdf")
                    except Exception as e:
                        st.warning(f"Không thể tạo PDF: {e}")

                    # Lưu vào session_state
                    st.session_state.executed_notebook = out_nb
                    st.session_state.pdf_path = pdf_path
                    st.session_state.html_preview = html_preview
                    st.session_state.notebook_ran = True


            except Exception as e:
                st.error(f"Lỗi khi chạy notebook: {e}\nCài đặt: pip install nbformat nbclient nbconvert")

        # --- Hiển thị kết quả nếu đã chạy ---
        if st.session_state.notebook_ran:
            out_nb = st.session_state.executed_notebook
            pdf_path = st.session_state.pdf_path
            html_preview = st.session_state.html_preview

            st.success("Notebook đã chạy xong")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if out_nb and os.path.exists(out_nb):
                    with open(out_nb, "rb") as f:
                        st.download_button(
                            "Tải Notebook",
                            f,
                            file_name="main_executed.ipynb",
                            mime="application/x-ipynb+json",
                            use_container_width=True,
                        )
            with col_dl2:
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Tải PDF",
                            f,
                            file_name="main_executed.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.info("PDF không khả dụng")

            if html_preview:
                with st.expander("Xem trước chi tiết notebook (HTML)", expanded=False):
                    st.components.v1.html(html_preview, height=700, scrolling=True)
        else:
            st.info("Nhấn nút **'Chạy Notebook'** để thực thi và xem kết quả.")


def display_eda(df: pd.DataFrame) -> None:
    """
    Hiển thị phần Khảo sát dữ liệu (EDA - Exploratory Data Analysis).

    Bao gồm: Bảng dữ liệu, thông tin chung (shape, dtypes),
    phân bố nhãn và thống kê mô tả.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu chính để hiển thị.
    """
    st.markdown('<div id="khao-sat-du-lieu" class="anchor"></div>', unsafe_allow_html=True)
    st.header("1. Khảo sát dữ liệu")

    st.subheader("Bảng dữ liệu")
    st.dataframe(df, use_container_width=True, height=400)

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

def display_visualization(df: pd.DataFrame, X_scaled: np.ndarray) -> None:
    """
    Hiển thị phần Trực quan hóa (PCA và t-SNE).
    
    Sử dụng dữ liệu đã cache từ `get_pca` và `get_tsne`.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu (để lấy nhãn 'Type').
        X_scaled (np.ndarray): Dữ liệu đã chuẩn hóa (để truyền vào 
                               hàm cache).
    """
    st.markdown('<div id="truc-quan-hoa" class="anchor"></div>', unsafe_allow_html=True)
    st.header("2. Trực quan hóa")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA 2D")
        with st.spinner("Tính toán PCA..."):
            X_pca, pca = get_pca(X_scaled) # Dùng hàm đã cache
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for t in sorted(df["Type"].unique()):
            mask = df["Type"] == t
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=70, label=f"Type {t}", alpha=0.75,
                edgecolors="#555555", linewidths=0.4,
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
            X_tsne = get_tsne(X_scaled) # Dùng hàm đã cache
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for t in sorted(df["Type"].unique()):
            mask = df["Type"] == t
            ax.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                s=70, label=f"Type {t}", alpha=0.75,
                edgecolors="#555555", linewidths=0.4,
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

def display_classification(df: pd.DataFrame) -> None:
    """
    Hiển thị giao diện và xử lý logic cho phần Phân loại (Classification).

    Cho phép người dùng chọn số K-Fold và chạy so sánh các mô hình.
    Kết quả được lưu vào `st.session_state`.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu (để truyền vào 
                           hàm `run_classification`).
    """
    st.markdown('<div id="phan-loai" class="anchor"></div>', unsafe_allow_html=True)
    st.header("3. Phân loại (Classification)")

    col_cf1, col_cf2 = st.columns([4, 1])
    with col_cf1:
        cv_k = st.slider("KFold splits", min_value=3, max_value=10, value=10, step=1, key="cv_slider")
    with col_cf2:
        st.write("")  # Spacer
        run_classification_btn = st.button("Chạy", key="run_classification")

    if run_classification_btn:
        with st.spinner("Đang huấn luyện các mô hình..."):
            progress = st.progress(0)
            def _cb(i: int, total: int):
                progress.progress(min(int(i / total * 100), 99))
            
            results = run_classification(df, cv_k=cv_k, progress_callback=_cb)
            progress.progress(100)
            st.session_state['classification_results'] = results

    if 'classification_results' in st.session_state:
        st.success("Phân loại hoàn thành")
        st.subheader("Kết quả")
        st.dataframe(st.session_state['classification_results'], use_container_width=True)

def display_clustering(X_scaled: np.ndarray, y: np.ndarray) -> None:
    """
    Hiển thị giao diện và xử lý logic cho phần Gom cụm (Clustering).

    Bao gồm K-Means và DBSCAN. Người dùng có thể tương tác với
    tham số và xem kết quả (chỉ số, biểu đồ).

    Args:
        X_scaled (np.ndarray): Dữ liệu đã được chuẩn hóa.
        y (np.ndarray): Nhãn (ground-truth) của dữ liệu.
    """
    st.markdown('<div id="gom-cum" class="anchor"></div>', unsafe_allow_html=True)
    st.header("4. Gom cụm (Clustering)")

    col_cluster1, col_cluster2 = st.columns(2)

    # --- K-MEANS ---
    with col_cluster1:
        st.subheader("K-Means")
        n_clusters_kmeans = st.slider("Số cụm K-Means", min_value=2, max_value=10, value=6, step=1, key="kmeans_clusters")
        run_kmeans = st.button("Chạy", key="btn_kmeans")
        
        result_container_km = st.container(border=False) # container cho kết quả
        
        if run_kmeans:
            with st.spinner("Đang chạy K-Means..."):
                kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=20)
                km_labels = kmeans.fit_predict(X_scaled)
                km_eval = evaluate_clustering(y, km_labels)
                
                X_pca, _ = get_pca(X_scaled) # Dùng lại PCA đã cache
                
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
                plt.colorbar(scatter, ax=ax, label="Cụm", fraction=0.046, pad=0.04)
                fig.tight_layout()
                
                st.session_state['kmeans_eval'] = km_eval
                st.session_state['kmeans_fig'] = fig
        
        with result_container_km:
            if 'kmeans_eval' in st.session_state:
                st.success("K-Means hoàn thành")
                col_k1, col_k2, col_k3 = st.columns(3)
                col_k1.metric("F1", f"{st.session_state['kmeans_eval']['F1_macro']:.3f}")
                col_k2.metric("ARI", f"{st.session_state['kmeans_eval']['ARI']:.3f}")
                col_k3.metric("NMI", f"{st.session_state['kmeans_eval']['NMI']:.3f}")
                st.pyplot(st.session_state['kmeans_fig'], use_container_width=False)

    # --- DBSCAN ---
    with col_cluster2:
        st.subheader("DBSCAN")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            eps_dbscan = st.slider("Bán kính epsilon (ε)", 0.3, 2.0, 1.0, 0.1, key="dbscan_eps")
        with col_d2:
            min_samples_db = st.slider("Số mẫu tối thiểu (min_samples)", 3, 20, 5, 1, key="dbscan_min")

        run_dbscan = st.button("Chạy", key="btn_dbscan")
        result_container_db = st.container(border=False) # container cho kết quả
        
        if run_dbscan:
            with st.spinner("Đang chạy DBSCAN..."):
                db = DBSCAN(eps=eps_dbscan, min_samples=min_samples_db)
                db_labels = db.fit_predict(X_scaled)
                db_eval = evaluate_clustering(y, db_labels)
                n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                n_noise = list(db_labels).count(-1)
                
                X_pca, _ = get_pca(X_scaled) # Dùng lại PCA đã cache
                
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
                plt.colorbar(scatter, ax=ax, label="Cụm", fraction=0.046, pad=0.04)
                fig.tight_layout()
                
                st.session_state['dbscan_eval'] = db_eval
                st.session_state['dbscan_clusters'] = n_clusters_db
                st.session_state['dbscan_noise'] = n_noise
                st.session_state['dbscan_fig'] = fig
        
        with result_container_db:
            if 'dbscan_eval' in st.session_state:
                st.success("DBSCAN hoàn thành")
                cols = st.columns(5)
                cols[0].metric("F1", f"{st.session_state['dbscan_eval']['F1_macro']:.3f}")
                cols[1].metric("ARI", f"{st.session_state['dbscan_eval']['ARI']:.3f}")
                cols[2].metric("NMI", f"{st.session_state['dbscan_eval']['NMI']:.3f}")
                cols[3].metric("Cụm", st.session_state['dbscan_clusters'])
                cols[4].metric("Nhiễu", st.session_state['dbscan_noise'])
                
                st.pyplot(st.session_state['dbscan_fig'], use_container_width=False)


# === Hàm main để chạy ứng dụng ===

def main() -> None:
    """
    Hàm chính điều phối toàn bộ ứng dụng Streamlit.
    
    Tải dữ liệu, sau đó gọi các hàm `display_` để
    vẽ lên từng phần của giao diện.
    """
    
    # Tải dữ liệu (đã cache)
    df = get_data()
    X_cont, X_scaled, y = get_scaled_data(df)

    # Hiển thị các thành phần
    display_sidebar()
    display_introduction(df)
    display_eda(df)
    display_visualization(df, X_scaled)
    display_classification(df)
    display_clustering(X_scaled, y)

if __name__ == "__main__":
    main()