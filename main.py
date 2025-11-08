# ============================================================
# main.py — Sentiment Analysis Training & Inference Pipeline
# ============================================================

import os

# Pastikan folder output ada
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

print("\n============================================")
print(" Sentiment Analysis Pipeline (Dockerized)")
print("============================================\n")

import os
import io
import re
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports (used across multiple funcs)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import StratifiedKFold

# For saving models
import joblib

# IPython.display fallback (works in notebooks; safe fallback to print)
try:
    from IPython.display import display, clear_output  # type: ignore
except Exception:
    def display(x):
        print(x)
    def clear_output(wait=False):
        pass

# ---------------------------
# CONFIG (ubah sesuai kebutuhan)
# ---------------------------
FILEPATH = os.path.join(os.getcwd(), "data_gabungan.csv")
MODELS_DIR = "models"
IMAGES_DIR = "images"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------
# I/O & UTIL
# ---------------------------
def read_csv_robust_from_path(path: str) -> pd.DataFrame:
    """Baca CSV dengan fallback encoding dan on_bad_lines skip."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    for enc in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            pass
    # fallback skip bad lines
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

def save_joblib(obj: Any, path: str):
    joblib.dump(obj, path)
    print(f"[SAVED] {path}")

def save_fig(filename: str):
    path = os.path.join(IMAGES_DIR, filename)
    try:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[IMG SAVED] {path}")
    except Exception as e:
        print(f"[IMG SAVE FAILED] {path} -> {e}")

def _sanitize_filename(s: str) -> str:
    # make a safe filename from title-like strings
    s = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in s)
    s = s.strip().replace(" ", "_")
    return s.lower()

# ---------------------------
# EXPLORATION
# ---------------------------
def explore_data(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    print("=== INFO DATA ===")
    print(f"Jumlah baris: {n_rows:,}")
    print(f"Jumlah kolom: {n_cols}")
    print("Nama kolom  :", list(df.columns))
    print()

    print("=== dtypes ===")
    print(df.dtypes)
    print()

    missing = df.isna().sum().to_frame("missing")
    missing["missing_%"] = (missing["missing"] / max(len(df), 1) * 100).round(2)
    print("=== Missing values ===")
    print(missing)
    print()

    schema_rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        nunique = int(df[col].nunique(dropna=True))
        ex = df[col].dropna().iloc[0] if non_null > 0 else None
        schema_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": non_null,
            "missing": int(len(df) - non_null),
            "missing_%": round(100 * (len(df) - non_null) / max(len(df), 1), 2),
            "unique": nunique,
            "example": ex if isinstance(ex, (int, float, str)) else str(ex)
        })
    schema_df = pd.DataFrame(schema_rows)
    print("=== Schema ringkas ===")
    print(schema_df.to_string(index=False))
    print()

    print("=== 10 baris pertama ===")
    display(df.head(10))

def candidate_text_label_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    text_keywords = ["review", "text", "content", "body", "comment", "message", "description"]
    label_keywords = ["rating", "stars", "label", "sentiment", "polarity", "score", "target", "class"]

    cand_text = [c for c in df.columns if any(k in str(c).lower() for k in text_keywords)]
    cand_label = [c for c in df.columns if any(k in str(c).lower() for k in label_keywords)]

    if not cand_text:
        avg_lens = {}
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].dropna().astype(str)
                if len(s) > 0:
                    avg_lens[c] = s.str.len().mean()
        if avg_lens:
            cand_text = sorted(avg_lens, key=avg_lens.get, reverse=True)[:1]

    return cand_text, cand_label

# ---------------------------
# CLEANING
# ---------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' ', s)
    s = re.sub(r'[@#]\w+', ' ', s)
    s = re.sub(r'\S+@\S+\.\S+', ' ', s)
    s = re.sub(r"[^0-9a-zA-Záàâäãåçéèêëíìîïñóòôöõúùûü’'\.\,\!\?\s]", " ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def run_cleaning_pipeline(df: pd.DataFrame,
                          text_col_candidates: List[str] = None,
                          required_cols: List[str] = None,
                          min_chars: int = 5,
                          min_words: int = 2) -> pd.DataFrame:
    """
    Cleaning utama:
    - Tentukan kolom teks (default heuristik)
    - Buang NA review_text
    - Bersihkan teks
    - Buang baris kosong setelah cleaning
    - Tambah char_len & word_len
    - Buang teks terlalu pendek
    - Dedup (teks + rating)
    - Buat label binary & 3-class
    """
    df = df.copy()
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Kolom '{col}' tidak ditemukan di data.")

    # Pilih kolom teks
    if text_col_candidates:
        text_col = text_col_candidates[0]
    else:
        cand_text, _ = candidate_text_label_columns(df)
        text_col = cand_text[0] if cand_text else None

    if text_col is None:
        raise KeyError("Tidak ditemukan kandidat kolom teks. Beri tahu nama kolom teks.")

    # Buang NA pada teks asli
    before_na = len(df)
    df = df[~df[text_col].isna()].copy()
    after_na = len(df)

    # Bersihkan
    df["review_text_clean"] = df[text_col].astype(str).map(clean_text)

    # Buang kosong
    before_empty = len(df)
    df = df[df["review_text_clean"].str.len() > 0].copy()
    after_empty = len(df)

    # Panjang
    df["char_len"] = df["review_text_clean"].str.len()
    df["word_len"] = df["review_text_clean"].str.split().str.len()

    # Buang terlalu pendek
    before_short = len(df)
    df = df[(df["char_len"] >= min_chars) & (df["word_len"] >= min_words)].copy()
    after_short = len(df)

    # Dedup berdasarkan (review_text_clean, rating) bila ada rating
    if "rating" in df.columns:
        dup_tl_before = df.duplicated(subset=["review_text_clean", "rating"]).sum()
        df = df.drop_duplicates(subset=["review_text_clean", "rating"]).copy()
        dup_tl_after = df.duplicated(subset=["review_text_clean", "rating"]).sum()
    else:
        dup_tl_before = dup_tl_after = 0

    # Label mapping
    def map_3class(r):
        if r in [1, 2]:
            return "neg"
        elif r == 3:
            return "neu"
        elif r in [4, 5]:
            return "pos"
        return None

    def map_binary(r):
        if r in [1, 2]:
            return "neg"
        elif r in [4, 5]:
            return "pos"
        return None

    if "rating" in df.columns:
        df["sentiment_3class"] = df["rating"].map(map_3class)
        df["sentiment_binary"] = df["rating"].map(map_binary)
    else:
        # Jika rating tidak ada, mungkin dataset already labeled differently — jaga aman
        df["sentiment_3class"] = df.get("sentiment_3class", None)
        df["sentiment_binary"] = df.get("sentiment_binary", None)

    # Ringkasan singkat
    print("\n=== RINGKAS BERSIH ===")
    print(f"Baris awal (raw): {before_na:,}")
    print(f"Setelah buang NA teks      : {after_na:,} (hapus {before_na - after_na:,})")
    print(f"Setelah buang kosong pasca-clean  : {after_empty:,} (hapus {before_empty - after_empty:,})")
    print(f"Setelah buang teks terlalu pendek : {after_short:,} (hapus {before_short - after_short:,})")
    print(f"Dedup (teks+rating), sebelum/ sesudah: {dup_tl_before:,} -> {dup_tl_after:,}")
    print(f"Total baris bersih (all rows)     : {len(df):,}")

    return df

# ---------------------------
# SVM (Linear) BINARY (blok SVM asli)
# ---------------------------
def train_svm_binary(df: pd.DataFrame,
                     text_col: str = "review_text_clean",
                     label_col: str = "sentiment_binary",
                     test_size: float = 0.10) -> Dict[str, Any]:
    """
    Train SVM pipeline with TF-IDF and GridSearch tuning on train only.
    Kembalikan model terbaik, hasil evaluasi, dan saved predictions path.
    """
    assert text_col in df.columns, f"Kolom teks '{text_col}' tidak ditemukan."
    assert label_col in df.columns, f"Kolom label '{label_col}' tidak ditemukan."

    data_bin = df[df[label_col].notna()].copy()
    X = data_bin[text_col].astype(str).values
    y = data_bin[label_col].astype(str).values

    print("\n=== INFO DATASET BINER (SVM) ===")
    print(f"Teks kolom   : {text_col}")
    print(f"Sampel total : {len(y):,}")
    print(pd.Series(y).value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    print("\nSplit selesai.")
    print("Train:", len(y_train), "| Test:", len(y_test))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                                  min_df=2, max_df=0.98, sublinear_tf=True, norm="l2")),
        ("clf", LinearSVC())
    ])

    param_grid = {
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__tol": [1e-4, 1e-3, 1e-2, 0.5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1, refit=True)
    grid.fit(X_train, y_train)

    print("\n=== HASIL GRID SEARCH (SVM) ===")
    print("Best params:", grid.best_params_)
    print(f"CV best mean accuracy: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=["neg", "pos"], average="macro")

    print("\n=== EVALUASI TEST (SVM) ===")
    print(f"Akurasi       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=["neg", "pos"])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix – SVM (Linear)")
    plt.tight_layout()
    fname = "cm_svm_binary.png"
    save_fig(fname)
    plt.show()

    # Calibrate for probabilities
    try:
        clf_step = best_model.named_steps["clf"]
        tfidf = best_model.named_steps["tfidf"]
        calib = CalibratedClassifierCV(clf_step, cv=5, method="sigmoid")
        Xtr_tfidf = tfidf.transform(X_train)
        Xte_tfidf = tfidf.transform(X_test)
        calib.fit(Xtr_tfidf, y_train)
        proba_pos = calib.predict_proba(Xte_tfidf)[:, 1]
        y_test_bin = pd.Series(y_test).map({"neg": 0, "pos": 1}).values
        auc = roc_auc_score(y_test_bin, proba_pos)
        print(f"\nROC AUC (pos vs neg): {auc:.4f}")
        RocCurveDisplay.from_predictions(y_test_bin, proba_pos)
        plt.title("ROC Curve – SVM (Linear, Calibrated)")
        fname = "roc_svm_binary.png"
        save_fig(fname)
        plt.show()
    except Exception as e:
        print("AUC/calibration gagal:", e)
        proba_pos = None
        auc = None

    out_pred = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "pred_label": y_pred,
        **({"proba_pos": proba_pos} if proba_pos is not None else {})
    })
    out_pred_path = "svm_linear_predictions_test.csv"
    out_pred.to_csv(out_pred_path, index=False)
    print(f"File prediksi test disimpan: {out_pred_path}")

    return {
        "model": best_model,
        "calibrated": calib if 'calib' in locals() else None,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc},
        "pred_csv": out_pred_path
    }

# ---------------------------
# LOGISTIC REGRESSION (binary & multiclass)
# ---------------------------
def build_features(texts, kind="count", use_svd=False, n_components=300, fit=None):
    if kind == "count":
        vec = CountVectorizer(analyzer="word", ngram_range=(1,1), lowercase=True, token_pattern=r"\b\w+\b")
    elif kind == "tfidf":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1,1), lowercase=True, token_pattern=r"\b\w+\b")
    else:
        raise ValueError("kind harus 'count' atau 'tfidf'")

    if fit is None:
        X = vec.fit_transform(texts)
    else:
        vec = fit["vec"]
        X = vec.transform(texts)

    svd = None
    if use_svd:
        if fit is None:
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
            X = svd.fit_transform(X)
        else:
            svd = fit["svd"]
            X = svd.transform(X)

    return X, {"vec": vec, "svd": svd}

def evaluate_clf(y_true, y_pred, labels_order=None, title="confusion"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"Akurasi       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1       : {f1:.4f}\n")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    if labels_order is None:
        labels_order = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"CM – {title}")
    plt.tight_layout()
    # save + show
    fname = f"cm_{_sanitize_filename(str(title))}.png"
    save_fig(fname)
    plt.show()
    return acc, prec, rec, f1

def run_logistic_regression_variants(df: pd.DataFrame,
                                     text_col: str = "review_text_clean",
                                     binary_label_col: str = "sentiment_binary"):
    # Binary dataset
    if binary_label_col not in df.columns:
        def map_binary(r):
            if r in [1,2]: return "neg"
            if r in [4,5]: return "pos"
            return None
        df["sentiment_binary"] = df["rating"].map(map_binary)

    data_bin = df[df[binary_label_col].notna()].copy()
    X_text = data_bin[text_col].astype(str).values
    y_bin = data_bin[binary_label_col].astype(str).values

    print("\n=== INFO DATASET BINER (LR) ===")
    print(f"Teks kolom: {text_col}")
    print(f"Sampel total : {len(y_bin):,}")
    print(pd.Series(y_bin).value_counts().to_string())

    Xtr_text, Xte_text, ytr_bin, yte_bin = train_test_split(X_text, y_bin, test_size=0.20, stratify=y_bin, random_state=RANDOM_STATE)
    print("\nSplit 80/20 selesai.")

    def run_lr_binary(kind="count", use_svd=False, n_components=300, name=""):
        print(f"\n--- BINER | {name or (kind.upper() + ('+SVD' if use_svd else ''))} ---")
        Xtr, fitobjs = build_features(Xtr_text, kind=kind, use_svd=use_svd, n_components=n_components, fit=None)
        Xte, _ = build_features(Xte_text, kind=kind, use_svd=use_svd, n_components=n_components, fit=fitobjs)
        clf = LogisticRegression(solver="lbfgs", max_iter=10000, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr, ytr_bin)
        ypred = clf.predict(Xte)
        evaluate_clf(yte_bin, ypred, labels_order=["neg", "pos"], title=f"LR_Binary_{name or kind}")
        # ROC AUC if possible
        try:
            proba_pos = clf.predict_proba(Xte)[:, 1]
            y_true_bin = (pd.Series(yte_bin).map({"neg":0, "pos":1}).values)
            auc = roc_auc_score(y_true_bin, proba_pos)
            print(f"ROC AUC (pos vs neg): {auc:.4f}")
            RocCurveDisplay.from_predictions(y_true_bin, proba_pos)
            plt.title(f"ROC – LR Binary ({name or kind})")
            fname = f"roc_lr_binary_{_sanitize_filename(name or kind)}.png"
            save_fig(fname)
            plt.show()
        except Exception:
            pass

    # Run variations
    run_lr_binary(kind="count", use_svd=False, name="Count (tanpa DR)")
    run_lr_binary(kind="count", use_svd=True, n_components=750, name="Count + SVD (≈paper)")
    run_lr_binary(kind="tfidf", use_svd=False, name="TF-IDF (tanpa DR)")
    run_lr_binary(kind="tfidf", use_svd=True, n_components=400, name="TF-IDF + SVD")

    # Multi-class (5-class rating)
    data_mc = df.dropna(subset=["rating", text_col]).copy()
    X_text_mc = data_mc[text_col].astype(str).values
    y_mc = data_mc["rating"].astype(int).values
    print("\n=== INFO DATASET 5-KELAS (LR) ===")
    print(pd.Series(y_mc).value_counts().sort_index().to_string())

    Xtr_text_mc, Xte_text_mc, ytr_mc, yte_mc = train_test_split(X_text_mc, y_mc, test_size=0.20, stratify=y_mc, random_state=RANDOM_STATE)
    print("\nSplit 80/20 (5-class) selesai.")

    def run_lr_multiclass(kind="count", use_svd=False, n_components=300, name=""):
        print(f"\n--- 5-KELAS | {name or (kind.upper() + ('+SVD' if use_svd else ''))} ---")
        Xtr, fitobjs = build_features(Xtr_text_mc, kind=kind, use_svd=use_svd, n_components=n_components, fit=None)
        Xte, _ = build_features(Xte_text_mc, kind=kind, use_svd=use_svd, n_components=n_components, fit=fitobjs)
        clf = LogisticRegression(solver="lbfgs", max_iter=10000, class_weight="balanced", n_jobs=-1, multi_class="auto")
        clf.fit(Xtr, ytr_mc)
        ypred = clf.predict(Xte)
        labels = [1,2,3,4,5]
        evaluate_clf(yte_mc, ypred, labels_order=labels, title=f"LR_5class_{name or kind}")

    run_lr_multiclass(kind="count", use_svd=False, name="Count (tanpa DR)")
    run_lr_multiclass(kind="count", use_svd=True, n_components=300, name="Count + SVD")
    run_lr_multiclass(kind="tfidf", use_svd=False, name="TF-IDF (tanpa DR)")
    run_lr_multiclass(kind="tfidf", use_svd=True, n_components=400, name="TF-IDF + SVD")

# ---------------------------
# NAIVE BAYES
# ---------------------------
def run_naive_bayes(df: pd.DataFrame, text_col: str = "review_text_clean"):
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,1), token_pattern=r"\b\w+\b", lowercase=True, sublinear_tf=True, norm="l2")

    def eval_and_show(y_true, y_pred, labels_order=None, title="nb_confusion"):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"Akurasi       : {acc:.4f}")
        print(f"Macro Precision: {prec:.4f}")
        print(f"Macro Recall   : {rec:.4f}")
        print(f"Macro F1       : {f1:.4f}\n")
        print("Classification report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        if labels_order is None:
            labels_order = sorted(pd.unique(pd.Series(list(y_true)) + pd.Series(list(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"CM – {title}")
        plt.tight_layout()
        fname = f"cm_{_sanitize_filename(str(title))}.png"
        save_fig(fname)
        plt.show()
        return acc, prec, rec, f1

    # Binary 90/10
    def ensure_binary_label():
        if "sentiment_binary" not in df.columns and "rating" in df.columns:
            def mb(r):
                if r in [1,2]: return "neg"
                if r in [4,5]: return "pos"
                return None
            df["sentiment_binary"] = df["rating"].map(mb)

    ensure_binary_label()
    data_bin = df[df["sentiment_binary"].notna()].copy()
    X_bin = data_bin[text_col].astype(str).values
    y_bin = data_bin["sentiment_binary"].astype(str).values

    print("=== INFO DATASET BINER (NB) ===")
    print(f"Teks kolom   : {text_col}")
    print(f"Sampel total : {len(y_bin):,}")
    print(pd.Series(y_bin).value_counts().to_string())

    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_bin, y_bin, test_size=0.10, stratify=y_bin, random_state=RANDOM_STATE)
    print("\nSplit 90/10 (biner) selesai.")

    def run_nb_binary(clf, name):
        print(f"\n--- NB BINER | {name} ---")
        pipe = Pipeline([("tfidf", vec), ("nb", clf)])
        pipe.fit(Xtr_b, ytr_b)
        ypred = pipe.predict(Xte_b)
        acc, prec, rec, f1 = eval_and_show(yte_b, ypred, labels_order=["neg", "pos"], title=f"NB_binary_{name}")
        # ROC AUC if possible
        if hasattr(pipe.named_steps["nb"], "predict_proba"):
            proba_pos = pipe.predict_proba(Xte_b)[:, 1]
            y_true_bin = pd.Series(yte_b).map({"neg":0, "pos":1}).values
            auc = roc_auc_score(y_true_bin, proba_pos)
            print(f"ROC AUC (pos vs neg): {auc:.4f}")
            RocCurveDisplay.from_predictions(y_true_bin, proba_pos)
            plt.title(f"ROC – NB Binary ({name})")
            fname = f"roc_nb_binary_{_sanitize_filename(name)}.png"
            save_fig(fname)
            plt.show()

            out_path = f"nb_binary_tfidf_pred_test_{name.replace(' ', '_').lower()}.csv"
            pd.DataFrame({"text": Xte_b, "true_label": yte_b, "pred_label": ypred, "proba_pos": proba_pos}).to_csv(out_path, index=False)
        else:
            out_path = f"nb_binary_tfidf_pred_test_{name.replace(' ', '_').lower()}.csv"
            pd.DataFrame({"text": Xte_b, "true_label": yte_b, "pred_label": ypred}).to_csv(out_path, index=False)
        print("Saved:", out_path)

    run_nb_binary(MultinomialNB(alpha=1.0), "MultinomialNB")
    run_nb_binary(ComplementNB(alpha=1.0), "ComplementNB")

    # 3-class 70/30
    def to_3class(r):
        if r in [1,2]: return "neg"
        if r == 3:    return "neu"
        if r in [4,5]:return "pos"
        return None

    if "rating" in df.columns:
        df["sentiment_3class"] = df["rating"].map(to_3class)
    data_3c = df[df["sentiment_3class"].notna()].copy()
    X_3c = data_3c[text_col].astype(str).values
    y_3c = data_3c["sentiment_3class"].astype(str).values

    print("\n=== INFO DATASET 3-KELAS (NB) ===")
    print(pd.Series(y_3c).value_counts().to_string())

    Xtr_3c, Xte_3c, ytr_3c, yte_3c = train_test_split(X_3c, y_3c, test_size=0.30, stratify=y_3c, random_state=RANDOM_STATE)
    print("\nSplit 70/30 (3-class) selesai.")

    def run_nb_3class(clf, name):
        print(f"\n--- NB 3-KELAS | {name} ---")
        pipe = Pipeline([("tfidf", vec), ("nb", clf)])
        pipe.fit(Xtr_3c, ytr_3c)
        ypred = pipe.predict(Xte_3c)
        eval_and_show(yte_3c, ypred, labels_order=["neg","neu","pos"], title=f"NB_3class_{name}")
        out_path = f"nb_3class_tfidf_pred_test_{name.replace(' ', '_').lower()}.csv"
        pd.DataFrame({"text": Xte_3c, "true_label": yte_3c, "pred_label": ypred}).to_csv(out_path, index=False)
        print("Saved:", out_path)

    run_nb_3class(MultinomialNB(alpha=1.0), "MultinomialNB")
    run_nb_3class(ComplementNB(alpha=1.0), "ComplementNB")

# ---------------------------
# Visual comparison across final models
# ---------------------------
def compare_models_visual(df: pd.DataFrame, text_col: str = "review_text_clean", label_col: str = "sentiment_binary"):
    assert text_col in df.columns, f"Kolom teks '{text_col}' tidak ditemukan."
    if label_col not in df.columns and "rating" in df.columns:
        def map_binary(r):
            if r in [1,2]: return "neg"
            if r in [4,5]: return "pos"
            return None
        df["sentiment_binary"] = df["rating"].map(map_binary)

    data_bin = df[df["sentiment_binary"].notna()].copy()
    X = data_bin[text_col].astype(str).values
    y = data_bin["sentiment_binary"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE)

    pipelines = {
        "SVM (Linear)": Pipeline([("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.98, sublinear_tf=True, norm="l2")), ("clf", LinearSVC(C=0.5, tol=1e-4))]),
        "Logistic Regression": Pipeline([("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,1), token_pattern=r"\b\w+\b", lowercase=True, sublinear_tf=True, norm="l2")), ("clf", LogisticRegression(solver="lbfgs", max_iter=10000, n_jobs=-1))]),
        "Naive Bayes (Complement)": Pipeline([("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,1), token_pattern=r"\b\w+\b", lowercase=True, sublinear_tf=True, norm="l2")), ("clf", ComplementNB(alpha=1.0))])
    }

    rows = []
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred_tr = pipe.predict(X_train)
        acc_tr = accuracy_score(y_train, y_pred_tr)
        f1_tr = f1_score(y_train, y_pred_tr, average="macro")
        y_pred_te = pipe.predict(X_test)
        acc_te = accuracy_score(y_test, y_pred_te)
        f1_te = f1_score(y_test, y_pred_te, average="macro")
        rows.append({"model": name, "train_accuracy": acc_tr, "train_macroF1": f1_tr, "test_accuracy": acc_te, "test_macroF1": f1_te})

    comp_df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    display(comp_df.style.format({
        "train_accuracy": "{:.4f}",
        "train_macroF1": "{:.4f}",
        "test_accuracy": "{:.4f}",
        "test_macroF1": "{:.4f}"
    }))

    # Save only comparison CSV (as requested)
    comp_df.to_csv("model_comparison_binary.csv", index=False)
    print("Saved: model_comparison_binary.csv")

    # Visuals
    plt.figure(figsize=(7,4))
    plt.bar(comp_df["model"], comp_df["train_accuracy"])
    plt.title("Training Accuracy per Model (Binary, 90/10)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    fname = "comparison_train_accuracy.png"
    save_fig(fname)
    plt.show()

    plt.figure(figsize=(7,4))
    plt.bar(comp_df["model"], comp_df["test_accuracy"])
    plt.title("Test Accuracy per Model (Binary, 90/10)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    fname = "comparison_test_accuracy.png"
    save_fig(fname)
    plt.show()

    best = comp_df.sort_values(["test_accuracy", "test_macroF1"], ascending=False).iloc[0]
    print("\n=== PERBANDINGAN AKHIR (Binary 90/10) ===")
    print(f"Model terbaik (test accuracy): {best['model']}")
    print(f"- Test Accuracy : {best['test_accuracy']:.4f}")
    print(f"- Test Macro-F1 : {best['test_macroF1']:.4f}")

# ---------------------------
# Live inference UI-like (terminal-friendly) integrated into final training flow
# ---------------------------
def train_final_pipelines_and_live_inference(df: pd.DataFrame, text_col: str = "review_text_clean"):
    # Train three final pipelines on entire clean binary dataset
    if "rating" in df.columns and "sentiment_binary" not in df.columns:
        def map_binary(r):
            if r in [1,2]: return "neg"
            if r in [4,5]: return "pos"
            return None
        df["sentiment_binary"] = df["rating"].map(map_binary)

    _df_bin = df.dropna(subset=[text_col, "sentiment_binary"]).copy()
    _df_bin = _df_bin[_df_bin["sentiment_binary"].notna()].copy()
    print(f"Melatih model biner di SELURUH data bersih: {len(_df_bin):,} sampel (neg/pos)")

    def make_vec_svm():
        return TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.98, sublinear_tf=True, norm="l2")

    def make_vec_unigram():
        return TfidfVectorizer(analyzer="word", ngram_range=(1,1), token_pattern=r"\b\w+\b", lowercase=True, sublinear_tf=True, norm="l2")

    _svm_base = LinearSVC(C=0.5, tol=1e-4)
    svm_bin = make_pipeline(make_vec_svm(), CalibratedClassifierCV(_svm_base, method="sigmoid", cv=3))
    svm_bin.fit(_df_bin[text_col].astype(str).values, _df_bin["sentiment_binary"].astype(str).values)

    lr_bin = make_pipeline(make_vec_unigram(), LogisticRegression(solver="lbfgs", max_iter=10000, n_jobs=-1))
    lr_bin.fit(_df_bin[text_col].astype(str).values, _df_bin["sentiment_binary"].astype(str).values)

    nb_bin = make_pipeline(make_vec_unigram(), ComplementNB(alpha=1.0))
    nb_bin.fit(_df_bin[text_col].astype(str).values, _df_bin["sentiment_binary"].astype(str).values)

    # ---------------------------
    # BLOK 8: Terminal-Based Live Inference
    # (inserted here: after training pipelines, before saving models)
    # ---------------------------
    print("\n==========================================")
    print("     AUTO INFERENCE (SVM / LR / NB)")
    print("==========================================")
    print("✦ Mode otomatis diaktifkan karena Docker tidak mendukung input().")
    print("✦ Menggunakan sampel teks otomatis.\n")

    # Kamu bisa edit daftar teks ini bebas
    sample_texts = [
        "Ayamnya enak banget, tapi aplikasinya error pas mau bayar"
    ]

    print("Teks yang diuji:")
    for i, txt in enumerate(sample_texts):
        print(f"{i+1}. {txt}")

    print("\nSedang memproses...\n")

    def predict_all_models_local(texts_list):
        rows = []
        for t in texts_list:
            # SVM
            try:
                pred = svm_bin.predict([t])[0]
                proba = svm_bin.predict_proba([t])[0]
            except Exception:
                pred = svm_bin.predict([t])[0]
                proba = [np.nan, np.nan]

            rows.append({
                "Model": "SVM (Linear)",
                "Input": t,
                "Prediksi": pred,
                "p(neg)": float(proba[0]) if not np.isnan(proba[0]) else None,
                "p(pos)": float(proba[1]) if not np.isnan(proba[1]) else None
            })

            # LR
            pred = lr_bin.predict([t])[0]
            proba = lr_bin.predict_proba([t])[0]
            rows.append({
                "Model": "Logistic Regression",
                "Input": t,
                "Prediksi": pred,
                "p(neg)": float(proba[0]),
                "p(pos)": float(proba[1])
            })

            # NB
            pred = nb_bin.predict([t])[0]
            proba = nb_bin.predict_proba([t])[0]
            rows.append({
                "Model": "Naive Bayes (Complement)",
                "Input": t,
                "Prediksi": pred,
                "p(neg)": float(proba[0]),
                "p(pos)": float(proba[1])
            })

        return pd.DataFrame(rows)

    df_live = predict_all_models_local(sample_texts)

    print("\n======= HASIL PREDIKSI (AUTO) =======\n")
    print(df_live.to_string(index=False))

    out_path = os.path.join(MODELS_DIR, "auto_inference_results.csv")
    df_live.to_csv(out_path, index=False)
    print(f"\n💾 Saved: {out_path}")

    # Simpen model
    save_joblib(svm_bin, os.path.join(MODELS_DIR, "svm_bin_final.pkl"))
    save_joblib(lr_bin, os.path.join(MODELS_DIR, "logreg_bin_final.pkl"))
    save_joblib(nb_bin, os.path.join(MODELS_DIR, "nb_bin_final.pkl"))

    print("\n✅ Auto inference complete. Models saved.")
    return {"svm": svm_bin, "lr": lr_bin, "nb": nb_bin}

# ---------------------------
# Save bundle of models (all_models_bin.pkl)
# ---------------------------
def bundle_and_save_models(models: Dict[str, Any], out_path: str = None):
    if out_path is None:
        out_path = os.path.join(MODELS_DIR, "all_models_bin.pkl")
    bundle_meta = {
        "task": "binary",
        "models": list(models.keys())
    }
    models_bundle = {**models, "meta": bundle_meta}
    joblib.dump(models_bundle, out_path)
    print(f"[SAVED] {out_path}")

# ---------------------------
# MAIN flow
# ---------------------------
def main():
    # 1) Load
    print("Loading data...")
    df = read_csv_robust_from_path(FILEPATH)

    # 2) Explore (prints)
    explore_data(df)

    # 3) Candidate text/label detection
    cand_text, cand_label = candidate_text_label_columns(df)
    print("Kandidat kolom teks :", cand_text if cand_text else "—")
    print("Kandidat kolom label:", cand_label if cand_label else "—")
    print()

    # 4) Cleaning (memerlukan nama kolom review)
    text_col = cand_text[0] if cand_text else "review_text"
    required_cols = ["rating"] if "rating" in df.columns else []
    df_clean = run_cleaning_pipeline(df, text_col_candidates=[text_col], required_cols=required_cols)

    # 5) Jika ada kandidat label, tampil distribusi
    if "sentiment_binary" in df_clean.columns:
        print("\n=== Distribusi label (binary) ===")
        display(pd.Series(df_clean["sentiment_binary"]).value_counts().to_frame("count"))
    if "sentiment_3class" in df_clean.columns:
        print("\n=== Distribusi label (3class) ===")
        display(pd.Series(df_clean["sentiment_3class"]).value_counts().to_frame("count"))

    # 6) SVM train + tune (binary) — heavy, optional
    try:
        svm_res = train_svm_binary(df_clean)
    except Exception as e:
        print("Training SVM gagal (skip). Error:", e)
        svm_res = None

    # 7) Logistic Regression experiments
    try:
        run_logistic_regression_variants(df_clean)
    except Exception as e:
        print("Logistic Regression runs gagal (skip). Error:", e)

    # 8) Naive Bayes runs
    try:
        run_naive_bayes(df_clean)
    except Exception as e:
        print("Naive Bayes runs gagal (skip). Error:", e)

    # 9) Compare summary visuals
    try:
        compare_models_visual(df_clean)
    except Exception as e:
        print("Compare visual gagal (skip). Error:", e)

    # 10) Train final pipelines on all clean data and save
    try:
        final_models = train_final_pipelines_and_live_inference(df_clean)
        # bundle & save
        bundle_and_save_models({k: final_models[k] for k in ["svm", "lr", "nb"]})
    except Exception as e:
        print("Training final models gagal (skip). Error:", e)

    print("\nSelesai. Cek file hasil model keseluruhan sudah disimpan!.")

if __name__ == "__main__":
    main()

print("\n============================================")
print(" Pipeline selesai. Models & images telah dibuat.")
print("============================================\n")