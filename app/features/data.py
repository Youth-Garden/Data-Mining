import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner=False)
def load_data(path: str = "glass-data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df.columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    # Enforce stable dtypes to avoid Arrow conversion issues
    try:
        df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("Int64").astype("int64")
    except Exception:
        pass
    for col in ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    try:
        df["Type"] = pd.to_numeric(df["Type"], errors="coerce").astype("Int64").astype("int64")
    except Exception:
        pass
    return df


@st.cache_data(show_spinner=False)
def scale_continuous(df: pd.DataFrame):
    """Return (X_df, X_scaled) where X_df excludes Id and Type."""
    X = df.drop(columns=["Id", "Type"])  # continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled
