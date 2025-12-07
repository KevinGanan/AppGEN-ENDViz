import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
# NO import cv2 - eliminado

# -----------------------------
# Imagen: métricas objetivas
# -----------------------------
def wcag_like_contrast_ratio(img: Image.Image, k=2):
    """
    Aproxima contraste tipo WCAG usando K-Means (k=2) sobre luminancia.
    Retorna ratio (>=1). Nota: es aproximación, no segmentación semántica.
    """
    im = img.convert("RGB").resize((512, 512))
    arr = np.asarray(im).astype(np.float32) / 255.0
    # luminancia relativa sRGB
    lum = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    X = lum.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    centers = np.sort(km.cluster_centers_.flatten())
    L1, L2 = centers[0], centers[-1]
    ratio = (L2 + 0.05) / (L1 + 0.05)
    return float(ratio)

def edge_density(img: Image.Image):
    """
    Densidad de bordes usando filtros Pillow (sin OpenCV).
    """
    # Convertir a escala de grises y redimensionar
    im = img.convert("L").resize((512, 512))
    
    # Aplicar filtro de detección de bordes
    edges = im.filter(ImageFilter.FIND_EDGES)
    
    # Convertir a array numpy
    edges_arr = np.array(edges, dtype=np.float32)
    
    # Normalizar y calcular densidad (0-1)
    if edges_arr.max() > 0:
        edges_arr = edges_arr / edges_arr.max()
    
    return float(edges_arr.mean())

# -----------------------------
# Acuerdo entre evaluadores
# -----------------------------
def fleiss_kappa_from_long(df_long, rater_col, item_col, cat_col):
    """
    Fleiss' kappa para datos categóricos (1..4) en formato largo.
    df_long: columnas [item_col, rater_col, cat_col]
    """
    cats = sorted(df_long[cat_col].dropna().unique())
    items = df_long[item_col].dropna().unique()
    n_cats = len(cats)
    N = len(items)

    # Matriz n_ij: cuenta de raters que asignan categoría j al ítem i
    counts = np.zeros((N, n_cats), dtype=int)
    item_index = {it:i for i,it in enumerate(items)}
    cat_index = {c:j for j,c in enumerate(cats)}

    for _, row in df_long.dropna(subset=[item_col, cat_col]).iterrows():
        i = item_index[row[item_col]]
        j = cat_index[row[cat_col]]
        counts[i, j] += 1

    # Total de evaluadores por ítem
    n_i = counts.sum(axis=1)
    if np.any(n_i == 0):
        return np.nan

    # Proporciones por categoría
    p_j = counts.sum(axis=0) / counts.sum()
    # Acuerdo por ítem
    P_i = ( (counts*(counts-1)).sum(axis=1) ) / ( n_i*(n_i-1 + 1e-8) )
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    kappa = (P_bar - P_e) / (1 - P_e + 1e-8)
    return float(kappa)

def avg_cohen_kappa_pairwise(df_wide):
    """
    Promedia Cohen's kappa en pares de evaluadores para una columna (1..4).
    df_wide: columnas = evaluadores, filas = ítems (una métrica a la vez).
    """
    from sklearn.metrics import cohen_kappa_score
    cols = df_wide.columns
    if len(cols) < 2:
        return np.nan
    kappas = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a = df_wide[cols[i]].dropna()
            b = df_wide[cols[j]].dropna()
            idx = a.index.intersection(b.index)
            if len(idx) >= 3:
                kappas.append(cohen_kappa_score(a.loc[idx], b.loc[idx]))
    return float(np.mean(kappas)) if kappas else np.nan