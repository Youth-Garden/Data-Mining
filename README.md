# Data Mining - Glass Classification & Clustering

![Python](https://img.shields.io/badge/Python-3+-3776AB?style=flat-square&logo=python&logoColor=white)
![Libraries](https://img.shields.io/badge/Libraries-scikit--learn_|_pandas_|_numpy-F7931E?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)

Dự án phân tích và phân loại dữ liệu kính (Glass Identification) sử dụng các thuật toán Machine Learning.

## Mô tả

Dự án này thực hiện phân tích dữ liệu kính bao gồm:

- Khảo sát và xử lý dữ liệu
- Trực quan hóa dữ liệu với PCA và t-SNE
- Huấn luyện và đánh giá các mô hình phân loại
- Gom cụm dữ liệu với K-Means và DBSCAN

## Cấu trúc thư mục

```
Data-Mining/
├── main.ipynb              # Notebook chính chứa toàn bộ code
├── glass-data.csv          # Dữ liệu kính
├── requirements.txt        # Các thư viện cần thiết
├── images/                 # Thư mục chứa các biểu đồ đã tạo
│   ├── visualization_2d.png
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   ├── kmeans_clusters_pca.png
│   └── dbscan_clusters_pca.png
└── README.md
```

## Dataset

Dữ liệu kính gồm 214 mẫu với 10 thuộc tính:

- **Id**: Mã định danh
- **RI**: Chỉ số khúc xạ (Refractive Index)
- **Na**: Sodium
- **Mg**: Magnesium
- **Al**: Aluminum
- **Si**: Silicon
- **K**: Potassium
- **Ca**: Calcium
- **Ba**: Barium
- **Fe**: Iron
- **Type**: Loại kính (1, 2, 3, 5, 6, 7)

## 🌐 Web demo (Streamlit)

### Live Demo (Streamlit Cloud)

Truy cập trực tiếp: **[https://data-mining-fvbrceyfcj8zqjdygdkpmt.streamlit.app/](https://data-mining-fvbrceyfcj8zqjdygdkpmt.streamlit.app/)**

### Chạy locally

Để chạy ứng dụng web trên máy của bạn:

```bash
streamlit run app/index.py
```

Sau khi chạy, mở đường dẫn mà Streamlit in ra (thường là http://localhost:8501).

### Tính năng Web

- 📊 Khảo sát dữ liệu: Bảng dữ liệu, thống kê, phân bố nhãn
- 📈 Trực quan hóa: PCA, t-SNE, phân bố các thuộc tính
- 🤖 Phân loại: GridSearchCV với KNN, Random Forest, SVM
- 🎯 Gom cụm: K-Means, DBSCAN với đánh giá chi tiết
- 📓 Notebook viewer: Xem kết quả từ main.ipynb (chạy tự động)

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/Youth-Garden-School/Data-Mining.git
cd Data-Mining
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Chạy notebook

```bash
jupyter notebook main.ipynb
```

hoặc mở trực tiếp trong VS Code.

## Nội dung chính

### 1. Khảo sát và xử lý dữ liệu

- Đọc dữ liệu và hiển thị kích thước
- Kiểm tra kiểu dữ liệu
- Phân tích phân bố các nhãn
- Thống kê giá trị min, max, mean

### 2. Trực quan hóa dữ liệu

- **PCA (Principal Component Analysis)**: Giảm chiều dữ liệu xuống 2D
- **t-SNE**: Trực quan hóa phi tuyến
- **Ma trận tương quan**: Phân tích mối quan hệ giữa các thuộc tính
- **Phân bố thuộc tính**: Histogram theo từng loại kính

**Output**:

- `visualization_2d.png`: Biểu đồ phân bố nhãn, PCA và t-SNE
- `correlation_matrix.png`: Ma trận tương quan
- `feature_distributions.png`: Phân bố các thuộc tính

### 3. Huấn luyện mô hình phân loại

Huấn luyện và so sánh 3 mô hình với **10-fold Cross Validation**:

#### K-Nearest Neighbors (KNN)

- Grid search: n_neighbors, weights
- Scaling: StandardScaler

#### Random Forest

- Grid search: n_estimators, max_depth, min_samples_split

#### Support Vector Machine (SVM)

- Grid search: C, gamma, kernel
- Scaling: StandardScaler

**Độ đo đánh giá**: F1-Score (Macro)

### 4. Gom cụm (Clustering)

#### K-Means

- Số cụm: Tự động xác định từ số nhãn thực tế
- Đánh giá: F1-Score, ARI, NMI

#### DBSCAN

- Tham số: eps=1.0, min_samples=5
- Đánh giá: F1-Score, ARI, NMI

**Output**:

- `kmeans_clusters_pca.png`: Biểu đồ cụm K-Means trên không gian PCA 2D
- `dbscan_clusters_pca.png`: Biểu đồ cụm DBSCAN trên không gian PCA 2D

## Kết quả

### Mô hình phân loại

| Model         | F1 Score (Macro) | Best Parameters                 |
| ------------- | ---------------- | ------------------------------- |
| KNN           | ~0.68            | n_neighbors=9, weights=distance |
| Random Forest | ~0.75            | n_estimators=300, max_depth=20  |
| SVM (RBF)     | ~0.70            | C=10, gamma=scale               |

### Gom cụm

| Algorithm | F1-Score | ARI   | NMI   |
| --------- | -------- | ----- | ----- |
| K-Means   | 0.391    | 0.170 | 0.313 |
| DBSCAN    | 0.244    | 0.205 | 0.341 |

## Công nghệ sử dụng

- **Python 3.x**
- **pandas**: Xử lý dữ liệu
- **numpy**: Tính toán số học
- **matplotlib**: Vẽ biểu đồ
- **seaborn**: Trực quan hóa nâng cao
- **scikit-learn**: Machine Learning
  - Models: KNN, Random Forest, SVM, K-Means, DBSCAN
  - Preprocessing: StandardScaler
  - Dimensionality Reduction: PCA, t-SNE
  - Model Selection: GridSearchCV, KFold
  - Metrics: F1-Score, ARI, NMI

---

**Lưu ý**: Trước khi chạy notebook, đảm bảo file `glass-data.csv` và thư mục `images/` tồn tại trong cùng thư mục.
