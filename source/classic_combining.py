#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classic_combining.py, ALL IDB2 classic-only pipeline, sem CNN,
com modo SWEEP para executar TODAS as combinações de modelos base,
sempre usando stacking com meta=SVM, salvando resultados em pastas separadas.

Exemplos

1) Rodar um experimento único (stacking, meta=SVM, modelos em --models)
python3 classic_combining.py --data_root /home/ubuntu/all-idb/dataset --cv 5 --out_dir ./ens_one \
  --models extratrees xgboost --tune --calibrate --calib_method sigmoid \
  --ring1 10 --ring2 24 --save_debug_masks --save_failures

2) Rodar sweep com todas as combinações possíveis, pool automático
python3 classic_combining.py --data_root /home/ubuntu/all-idb/dataset --cv 5 --out_dir ./sweep_svm \
  --tune --calibrate --calib_method sigmoid --ring1 10 --ring2 24 --save_failures \
  --sweep_all_combos

3) Rodar sweep com pool e tamanhos controlados
python3 classic_combining.py --data_root /home/ubuntu/all-idb/dataset --cv 5 --out_dir ./sweep_svm_trees \
  --tune --calibrate --calib_method sigmoid --ring1 10 --ring2 24 --save_failures \
  --sweep_all_combos --sweep_models_pool extratrees rf xgboost lightgbm catboost \
  --sweep_min_k 2 --sweep_max_k 4
"""

import os
import json
import argparse
import random
import itertools
import csv
from datetime import datetime

import numpy as np
import cv2
from glob import glob
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_AUC = True
except Exception:
    HAS_AUC = False

try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

CLASS_DIRS = {"leukemia": 1, "healthy": 0}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ============= DETERMINISM =============
def seed_everything(seed: int = 42, deterministic_opencv: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic_opencv:
        try:
            cv2.setRNGSeed(seed)
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

@dataclass
class Sample:
    path: str
    y: int

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def list_samples(data_root: str):
    samples = []
    for cname, y in CLASS_DIRS.items():
        cdir = os.path.join(data_root, cname)
        if not os.path.isdir(cdir):
            continue
        for p in sorted(glob(os.path.join(cdir, "**"), recursive=True)):
            if os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
                samples.append(Sample(path=p, y=y))
    if not samples:
        raise RuntimeError(f"No images found under {data_root}, expected folders: {list(CLASS_DIRS.keys())}")
    return samples

def read_image_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def safe_div(a, b, eps=1e-9):
    return float(a) / float(b + eps)

def fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    m = (binary_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape
    ff = m.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(m, holes)


def crop_center(arr_u8: np.ndarray, frac: float = 0.75) -> np.ndarray:
    h, w = arr_u8.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return arr_u8[y0:y0 + ch, x0:x0 + cw]


# ============= SEGMENTATION =============
def nucleus_pseudomask(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    img2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    _, _, Bn = cv2.split(lab2)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)

    score = (255 - Bn).astype(np.float32) + 0.5 * S.astype(np.float32)
    score = np.clip(score, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (5, 5), 0)

    center = crop_center(score, 0.75)
    t, _ = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th = cv2.threshold(score, t, 255, cv2.THRESH_BINARY)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    m = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=1)
    m = fill_holes(m)

    frac_area = (m > 0).mean()
    if frac_area < 0.01 or frac_area > 0.65:
        labf = lab2.reshape(-1, 3).astype(np.float32)
        sf = S.reshape(-1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 1.0)
        _, labels, centers = cv2.kmeans(labf, 3, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)

        best = None
        for ci in range(centers.shape[0]):
            meanL = float(centers[ci, 0])
            mask_ci = (labels.reshape(-1) == ci)
            meanS = float(sf[mask_ci].mean()) if np.any(mask_ci) else 0.0
            key = (meanL + (0.0 if meanS > 25.0 else 25.0))
            if best is None or key < best[0]:
                best = (key, ci)

        dark_idx = int(best[1])
        km = (labels.reshape(256, 256) == dark_idx).astype(np.uint8) * 255
        km = cv2.morphologyEx(km, cv2.MORPH_OPEN, k1, iterations=1)
        km = cv2.morphologyEx(km, cv2.MORPH_CLOSE, k2, iterations=1)
        m = fill_holes(km)

    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        nucleus = (m > 0).astype(np.uint8) * 255
        nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, k1, iterations=1)
        nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, k1, iterations=1)
        return nucleus

    candidates = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 120:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if x <= 1 or y <= 1 or (x + w) >= 255 or (y + h) >= 255:
            continue

        comp = (labels_cc == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = float(cv2.contourArea(c))
        p = float(cv2.arcLength(c, True))
        if a < 10:
            continue
        hull = cv2.convexHull(c)
        hull_a = float(cv2.contourArea(hull)) + 1e-6
        solidity = a / hull_a
        circularity = (4.0 * np.pi * a) / (p * p + 1e-6)

        comp_m = comp.astype(bool)
        meanS = float(S[comp_m].mean()) if comp_m.sum() > 0 else 0.0
        s = 0.55 * solidity + 0.35 * circularity + 0.10 * (meanS / 255.0)
        candidates.append((s, i, int(area)))

    if candidates:
        candidates.sort(reverse=True, key=lambda t: t[0])
        best_score, best_idx, best_area = candidates[0]
        keep = [best_idx]
        if len(candidates) > 1:
            s2, i2, a2 = candidates[1]
            if (s2 >= best_score - 0.08) and (a2 >= 0.35 * best_area):
                keep.append(i2)

        nucleus = np.zeros((256, 256), dtype=np.uint8)
        for ki in keep:
            nucleus = cv2.bitwise_or(nucleus, (labels_cc == ki).astype(np.uint8) * 255)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = 1 + int(np.argmax(areas))
        nucleus = (labels_cc == best_idx).astype(np.uint8) * 255

    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, k1, iterations=1)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, k1, iterations=1)
    return nucleus


def cell_pseudomask(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)

    score = (S.astype(np.float32) + (255 - L).astype(np.float32) * 0.7)
    score = np.clip(score, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (5, 5), 0)

    center = crop_center(score, 0.80)
    t, _ = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th = cv2.threshold(score, t, 255, cv2.THRESH_BINARY)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    m = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=1)
    m = fill_holes(m)

    frac_area = (m > 0).mean()
    if frac_area < 0.05 or frac_area > 0.75:
        labf = lab.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1.0)
        _, labels, centers = cv2.kmeans(labf, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)
        order = np.argsort(centers[:, 0])
        mid_idx = int(order[1])
        km = (labels.reshape(256, 256) == mid_idx).astype(np.uint8) * 255
        km = cv2.morphologyEx(km, cv2.MORPH_OPEN, k1, iterations=1)
        km = cv2.morphologyEx(km, cv2.MORPH_CLOSE, k2, iterations=1)
        m = fill_holes(km)

    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        cell = (m > 0).astype(np.uint8) * 255
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = 1 + int(np.argmax(areas))
        cell = (labels_cc == idx).astype(np.uint8) * 255

    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, k1, iterations=1)
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, k1, iterations=1)

    if (cell > 0).mean() < 0.10:
        cell = np.ones((256, 256), dtype=np.uint8) * 255
    return cell

def ring_mask_from_nucleus(nucleus_u8: np.ndarray, cell_u8: np.ndarray, dilate_px: int) -> np.ndarray:
    nucleus = (nucleus_u8 > 0).astype(np.uint8) * 255
    cell = (cell_u8 > 0).astype(np.uint8) * 255

    area = float((nucleus > 0).sum())
    rad = float(np.sqrt(max(1.0, area) / np.pi))
    cap = int(min(28, max(10, 2.6 * rad)))
    dilate_px = int(min(dilate_px, cap))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    dil = cv2.dilate(nucleus, k, iterations=1)
    ring = cv2.subtract(dil, nucleus)
    ring = cv2.bitwise_and(ring, ring, mask=(cell > 0).astype(np.uint8))
    return (ring > 0).astype(np.uint8) * 255


# ============= TEXTURE =============
def lbp_hist_u8(gray_u8: np.ndarray, mask_u8: np.ndarray, bins: int = 256) -> np.ndarray:
    g = gray_u8.astype(np.uint8)
    m = (mask_u8 > 0).astype(np.uint8)

    m2 = m.copy()
    m2[0, :] = 0
    m2[-1, :] = 0
    m2[:, 0] = 0
    m2[:, -1] = 0
    if m2.sum() < 50:
        return np.zeros((bins,), dtype=np.float32)

    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)

    n0 = g[0:-2, 0:-2]
    n1 = g[0:-2, 1:-1]
    n2 = g[0:-2, 2:]
    n3 = g[1:-1, 2:]
    n4 = g[2:, 2:]
    n5 = g[2:, 1:-1]
    n6 = g[2:, 0:-2]
    n7 = g[1:-1, 0:-2]

    code |= ((n0 >= c) << 7).astype(np.uint8)
    code |= ((n1 >= c) << 6).astype(np.uint8)
    code |= ((n2 >= c) << 5).astype(np.uint8)
    code |= ((n3 >= c) << 4).astype(np.uint8)
    code |= ((n4 >= c) << 3).astype(np.uint8)
    code |= ((n5 >= c) << 2).astype(np.uint8)
    code |= ((n6 >= c) << 1).astype(np.uint8)
    code |= ((n7 >= c) << 0).astype(np.uint8)

    valid = (m2[1:-1, 1:-1] > 0)
    vals = code[valid].ravel()
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.float32)

    hist = np.bincount(vals, minlength=bins).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    return hist

def glcm_feats(gray_u8: np.ndarray, mask_u8: np.ndarray):
    if not HAS_SKIMAGE:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}
    m = (mask_u8 > 0)
    if m.sum() < 60:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}

    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = gray_u8[y0:y1 + 1, x0:x1 + 1]
    cmask = m[y0:y1 + 1, x0:x1 + 1]

    q = (crop.astype(np.float32) / 16.0).astype(np.uint8)
    q[~cmask] = 0

    glcm = graycomatrix(
        q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )

    def prop(name):
        return float(np.mean(graycoprops(glcm, name)))

    return {
        "glcm_contrast": prop("contrast"),
        "glcm_homogeneity": prop("homogeneity"),
        "glcm_energy": prop("energy"),
        "glcm_correlation": prop("correlation"),
    }

# ============= TABULAR FEATURES =============
def region_stats(img_bgr_256: np.ndarray, gray_256: np.ndarray, mask_u8: np.ndarray, prefix: str):
    m = (mask_u8 > 0)
    if m.sum() < 30:
        return {
            f"{prefix}_area_frac": 0.0,
            f"{prefix}_mean_b": 0.0, f"{prefix}_mean_g": 0.0, f"{prefix}_mean_r": 0.0,
            f"{prefix}_std_b": 0.0, f"{prefix}_std_g": 0.0, f"{prefix}_std_r": 0.0,
            f"{prefix}_mean_gray": 0.0, f"{prefix}_std_gray": 0.0,
        }

    b, g, r = cv2.split(img_bgr_256)
    bvals = b[m].astype(np.float32)
    gvals = g[m].astype(np.float32)
    rvals = r[m].astype(np.float32)
    grvals = gray_256[m].astype(np.float32)

    return {
        f"{prefix}_area_frac": float(m.mean()),
        f"{prefix}_mean_b": float(bvals.mean()),
        f"{prefix}_mean_g": float(gvals.mean()),
        f"{prefix}_mean_r": float(rvals.mean()),
        f"{prefix}_std_b": float(bvals.std()),
        f"{prefix}_std_g": float(gvals.std()),
        f"{prefix}_std_r": float(rvals.std()),
        f"{prefix}_mean_gray": float(grvals.mean()),
        f"{prefix}_std_gray": float(grvals.std()),
    }

def morph_stats(mask_u8: np.ndarray, prefix: str):
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() < 30:
        return {
            f"{prefix}_area": 0.0,
            f"{prefix}_perimeter": 0.0,
            f"{prefix}_circularity": 0.0,
            f"{prefix}_eccentricity": 0.0,
            f"{prefix}_solidity": 0.0,
            f"{prefix}_extent": 0.0,
            f"{prefix}_major_minor_ratio": 0.0,
            f"{prefix}_roughness": 0.0,
            f"{prefix}_convexity": 0.0,
        }

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    area_px = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    circularity = safe_div(4.0 * np.pi * area_px, perim * perim)

    x, y, w, h = cv2.boundingRect(c)
    extent = safe_div(area_px, float(w * h))

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = safe_div(area_px, hull_area)

    hull_perim = float(cv2.arcLength(hull, True)) if hull is not None else 0.0
    convexity = safe_div(hull_perim, perim)
    roughness = safe_div(perim, np.sqrt(area_px))

    eccentricity = 0.0
    major_minor_ratio = 0.0
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        major = max(MA, ma)
        minor = max(1e-6, min(MA, ma))
        major_minor_ratio = safe_div(major, minor)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))))

    return {
        f"{prefix}_area": float(area_px),
        f"{prefix}_perimeter": float(perim),
        f"{prefix}_circularity": float(circularity),
        f"{prefix}_eccentricity": float(eccentricity),
        f"{prefix}_solidity": float(solidity),
        f"{prefix}_extent": float(extent),
        f"{prefix}_major_minor_ratio": float(major_minor_ratio),
        f"{prefix}_roughness": float(roughness),
        f"{prefix}_convexity": float(convexity),
    }

def downsample32(hist256: np.ndarray) -> np.ndarray:
    h = hist256.reshape(32, 8).sum(axis=1)
    h /= (h.sum() + 1e-9)
    return h.astype(np.float32)

def extract_features_all(img_bgr: np.ndarray, ring1_px: int, ring2_px: int, want_glcm: bool):
    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    bg = ((V > 235) & (S < 35)).astype(np.uint8) * 255
    bg_inv = cv2.bitwise_not(bg)

    nuc = nucleus_pseudomask(img)
    cell = cell_pseudomask(img)
    nuc = cv2.bitwise_and(nuc, nuc, mask=(cell > 0).astype(np.uint8))

    nuc = cv2.bitwise_and(nuc, nuc, mask=bg_inv)
    cell = cv2.bitwise_and(cell, cell, mask=bg_inv)

    ring1 = ring_mask_from_nucleus(nuc, cell, dilate_px=ring1_px)
    ring2 = ring_mask_from_nucleus(nuc, cell, dilate_px=ring2_px)
    ring2_only = cv2.subtract(ring2, ring1)
    ring2_only = (ring2_only > 0).astype(np.uint8) * 255

    ring1 = cv2.bitwise_and(ring1, ring1, mask=bg_inv)
    ring2_only = cv2.bitwise_and(ring2_only, ring2_only, mask=bg_inv)

    nuc_area = float((nuc > 0).sum())
    cell_area = float((cell > 0).sum())
    ring1_area = float((ring1 > 0).sum())
    ring2_area = float((ring2_only > 0).sum())

    feats = {}
    feats.update(morph_stats(nuc, "nuc"))
    feats.update(morph_stats(cell, "cell"))
    feats.update(morph_stats(ring1, "ring1"))
    feats.update(morph_stats(ring2_only, "ring2"))

    feats.update(region_stats(img, gray, nuc, "nuc"))
    feats.update(region_stats(img, gray, cell, "cell"))
    feats.update(region_stats(img, gray, ring1, "ring1"))
    feats.update(region_stats(img, gray, ring2_only, "ring2"))

    for ch in ["b", "g", "r", "gray"]:
        feats[f"delta1_mean_{ch}"] = feats[f"nuc_mean_{ch}"] - feats[f"ring1_mean_{ch}"]
        feats[f"delta1_std_{ch}"] = feats[f"nuc_std_{ch}"] - feats[f"ring1_std_{ch}"]
        feats[f"delta2_mean_{ch}"] = feats[f"nuc_mean_{ch}"] - feats[f"ring2_mean_{ch}"]
        feats[f"delta2_std_{ch}"] = feats[f"nuc_std_{ch}"] - feats[f"ring2_std_{ch}"]

    feats["ratio_nuc_cell"] = float(safe_div(nuc_area, cell_area))
    feats["ratio_ring1_cell"] = float(safe_div(ring1_area, cell_area))
    feats["ratio_ring2_cell"] = float(safe_div(ring2_area, cell_area))
    feats["ratio_nuc_ring1"] = float(safe_div(nuc_area, ring1_area))
    feats["ratio_nuc_ring2"] = float(safe_div(nuc_area, ring2_area))

    lbp_n = downsample32(lbp_hist_u8(gray, nuc, bins=256))
    lbp_r1 = downsample32(lbp_hist_u8(gray, ring1, bins=256))
    lbp_r2 = downsample32(lbp_hist_u8(gray, ring2_only, bins=256))

    for i, v in enumerate(lbp_n):
        feats[f"lbp_nuc_{i:02d}"] = float(v)
    for i, v in enumerate(lbp_r1):
        feats[f"lbp_ring1_{i:02d}"] = float(v)
    for i, v in enumerate(lbp_r2):
        feats[f"lbp_ring2_{i:02d}"] = float(v)

    if want_glcm:
        gn = glcm_feats(gray, nuc)
        g1 = glcm_feats(gray, ring1)
        g2 = glcm_feats(gray, ring2_only)
        for k, v in gn.items():
            feats["nuc_" + k] = float(v)
        for k, v in g1.items():
            feats["ring1_" + k] = float(v)
        for k, v in g2.items():
            feats["ring2_" + k] = float(v)

    return feats, nuc, ring1, ring2_only, cell, img

def build_feature_matrix(samples, ring1_px: int, ring2_px: int, want_glcm: bool, out_debug_dir=None, max_debug=0):
    X, y, paths = [], [], []
    feat_names = None
    tab_keys = None

    dbg = 0
    if out_debug_dir is not None:
        for sub in ["overlay", "nucleus", "cell", "ring1", "ring2"]:
            ensure_dir(os.path.join(out_debug_dir, sub))

    for s in samples:
        img0 = read_image_bgr(s.path)
        feats, nuc, ring1, ring2, cell, img256 = extract_features_all(
            img0,
            ring1_px=ring1_px,
            ring2_px=ring2_px,
            want_glcm=want_glcm,
        )

        if feat_names is None:
            tab_keys = list(feats.keys())
            feat_names = tab_keys.copy()

        row = [feats[k] for k in tab_keys]
        X.append(row)
        y.append(s.y)
        paths.append(s.path)

        if out_debug_dir is not None and dbg < max_debug:
            fname = os.path.basename(s.path)
            cv2.imwrite(os.path.join(out_debug_dir, "nucleus", fname), nuc)
            cv2.imwrite(os.path.join(out_debug_dir, "cell", fname), cell)
            cv2.imwrite(os.path.join(out_debug_dir, "ring1", fname), ring1)
            cv2.imwrite(os.path.join(out_debug_dir, "ring2", fname), ring2)
            out = img256.copy()
            out[nuc > 0] = (0.55 * out[nuc > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
            out[ring1 > 0] = (0.60 * out[ring1 > 0] + 0.40 * np.array([255, 0, 0])).astype(np.uint8)
            out[ring2 > 0] = (0.60 * out[ring2 > 0] + 0.40 * np.array([0, 255, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(out_debug_dir, "overlay", fname), out)
            dbg += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), paths, feat_names

# ============= MODELS =============
def build_model(name: str, tune: bool):
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=3200 if tune else 2200,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=0.65 if tune else "sqrt",
            bootstrap=False,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=1800 if tune else 1200,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=6.0 if tune else 3.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            )),
        ])

    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=9 if tune else 7, weights="distance")),
        ])

    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=10000, class_weight="balanced", solver="lbfgs")),
        ])

    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed, pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=3200 if tune else 2200,
            max_depth=7 if tune else 6,
            learning_rate=0.018 if tune else 0.035,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=0.5,
            reg_alpha=0.1,
            min_child_weight=1,
            gamma=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )

    raise ValueError("Unknown model name")

def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise RuntimeError("Model has no predict_proba")

def maybe_calibrate(base_model, X_tr, y_tr, do_calibrate: bool, method="sigmoid"):
    if not do_calibrate:
        base_model.fit(X_tr, y_tr)
        return base_model
    cal = CalibratedClassifierCV(base_model, method=method, cv=5)
    cal.fit(X_tr, y_tr)
    return cal

def metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return float(sens), float(spec)

# ============= META MODELS FOR STACKING =============
def fit_stacking_meta(P_tr_oof, y_tr, meta_name: str):
    if meta_name == "svm":
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=4.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42)),
        ])
        meta.fit(P_tr_oof, y_tr)
        return meta

    if meta_name == "logreg":
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=10000, class_weight="balanced", solver="lbfgs")),
        ])
        meta.fit(P_tr_oof, y_tr)
        return meta

    if meta_name == "ridge":
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=0.1, class_weight="balanced", random_state=42)),
        ])
        meta.fit(P_tr_oof, y_tr)
        return meta

    if meta_name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed, pip install xgboost")
        meta = xgb.XGBClassifier(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_lambda=0.5,
            min_child_weight=1,
            gamma=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        meta.fit(P_tr_oof, y_tr)
        return meta

    raise ValueError("Unknown meta")

def meta_score(meta, X):
    if hasattr(meta, "predict_proba"):
        return meta.predict_proba(X)[:, 1]
    if hasattr(meta, "decision_function"):
        s = meta.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    p = meta.predict(X).astype(np.float32)
    return np.clip(p, 0.0, 1.0)

# ============= DEBUG FAILURES =============
def save_failures(out_dir, fold, paths_te, y_true, y_pred, probs, debug_masks_dir=None, max_copy=50):
    ensure_dir(out_dir)
    fail_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
    rows = []
    for i in fail_idx:
        rows.append({
            "fold": int(fold),
            "path": paths_te[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "prob1": float(probs[i]),
        })
    csv_path = os.path.join(out_dir, f"failures_fold{fold}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("fold,path,y_true,y_pred,prob1\n")
        for r in rows:
            f.write(f"{r['fold']},{r['path']},{r['y_true']},{r['y_pred']},{r['prob1']:.8f}\n")

    if debug_masks_dir and len(fail_idx) > 0:
        dst_dir = os.path.join(out_dir, f"failures_masks_fold{fold}")
        ensure_dir(dst_dir)
        copied = 0
        for i in fail_idx:
            if copied >= max_copy:
                break
            bn = os.path.basename(paths_te[i])
            found = False
            for sub in ["overlay", "nucleus", "cell", "ring1", "ring2"]:
                src = os.path.join(debug_masks_dir, sub, bn)
                if os.path.isfile(src):
                    dst = os.path.join(dst_dir, f"{sub}_{bn}")
                    try:
                        with open(src, "rb") as fr:
                            data = fr.read()
                        with open(dst, "wb") as fw:
                            fw.write(data)
                        copied += 1
                        found = True
                        break
                    except Exception as e:
                        print(f"[warning] Failed to copy {src}: {e}")
            if not found:
                print(f"[warning] No mask found for {bn}")


# ============= SWEEP HELPERS =============
def _available_model_names():
    names = ["extratrees", "rf", "svm", "knn", "logreg"]
    if HAS_XGB:
        names.append("xgboost")
    return names

def _combo_dir_name(models_subset):
    return "stack_svm__" + "__".join(models_subset)

def _write_features_csv(out_dir_exp, paths, y, X, feat_names):
    csv_path = os.path.join(out_dir_exp, "features.csv")
    if os.path.isfile(csv_path):
        return
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,label," + ",".join(feat_names) + "\n")
        for p, yi, row in zip(paths, y, X):
            f.write(f"{p},{yi}," + ",".join([f"{v:.6f}" for v in row]) + "\n")

def run_one_experiment(*, X, y, paths, feat_names, args, models_subset, out_dir_exp):
    ensure_dir(out_dir_exp)

    debug_masks_dir = os.path.join(out_dir_exp, "debug_masks") if args.save_debug_masks else None
    max_debug = 1000000 if args.save_debug_masks else 0

    # Para ter debug_masks por experimento, precisamos reconstruir as features com out_debug_dir.
    # Para economizar, fazemos isso só se save_debug_masks estiver ativo.
    if args.save_debug_masks:
        # Reconstruir X somente para gerar debug_masks, mantendo consistência
        # Isso custa tempo, mas você pediu para salvar por pasta
        samples = list_samples(args.data_root)
        X2, y2, paths2, feat_names2 = build_feature_matrix(
            samples,
            ring1_px=args.ring1,
            ring2_px=args.ring2,
            want_glcm=not args.no_glcm,
            out_debug_dir=debug_masks_dir,
            max_debug=max_debug,
        )
        # Validação de consistência
        if len(y2) != len(y) or not np.all(y2 == y):
            raise RuntimeError("Inconsistência ao reconstruir features para debug_masks")
        X_use = X2
        paths_use = paths2
        feat_names_use = feat_names2
    else:
        X_use = X
        paths_use = paths
        feat_names_use = feat_names

    _write_features_csv(out_dir_exp, paths_use, y, X_use, feat_names_use)

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    accs, baccs, mccs, senss, specs = [], [], [], [], []
    aucs, praucs = [], []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(skf.split(X_use, y), 1):
        base_specs = [(name, build_model(name, tune=args.tune)) for name in models_subset]

        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed + 1000 + fold)
        P_tr_oof = np.zeros((len(tr), len(base_specs)), dtype=np.float32)

        for i_m, (mname, _) in enumerate(base_specs):
            for itr, iva in inner.split(X_use[tr], y[tr]):
                base = build_model(mname, tune=args.tune)
                fitted = maybe_calibrate(
                    base_model=base,
                    X_tr=X_use[tr][itr],
                    y_tr=y[tr][itr],
                    do_calibrate=args.calibrate,
                    method=args.calib_method,
                )
                P_tr_oof[iva, i_m] = get_proba(fitted, X_use[tr][iva]).astype(np.float32)

        meta = fit_stacking_meta(P_tr_oof, y[tr], meta_name="svm")

        P_te = np.zeros((len(te), len(base_specs)), dtype=np.float32)
        for i_m, (mname, base) in enumerate(base_specs):
            fitted = maybe_calibrate(
                base_model=base,
                X_tr=X_use[tr],
                y_tr=y[tr],
                do_calibrate=args.calibrate,
                method=args.calib_method,
            )
            P_te[:, i_m] = get_proba(fitted, X_use[te]).astype(np.float32)

        p_final = meta_score(meta, P_te)
        pred = (p_final >= 0.5).astype(int)

        acc = accuracy_score(y[te], pred)
        bacc = balanced_accuracy_score(y[te], pred)
        mcc = matthews_corrcoef(y[te], pred)
        cm = confusion_matrix(y[te], pred, labels=[0, 1])
        sens, spec = metrics_from_cm(cm)

        accs.append(acc)
        baccs.append(bacc)
        mccs.append(mcc)
        senss.append(sens)
        specs.append(spec)

        all_true.extend(list(y[te]))
        all_pred.extend(list(pred))

        if HAS_AUC:
            try:
                aucs.append(float(roc_auc_score(y[te], p_final)))
                praucs.append(float(average_precision_score(y[te], p_final)))
            except Exception:
                pass

        if args.save_failures:
            save_failures(
                out_dir=os.path.join(out_dir_exp, "failures"),
                fold=fold,
                paths_te=[paths_use[i] for i in te],
                y_true=y[te],
                y_pred=pred,
                probs=p_final,
                debug_masks_dir=debug_masks_dir,
                max_copy=80,
            )

    def mean_std(v):
        return float(np.mean(v)), float(np.std(v))

    mean_acc, std_acc = mean_std(accs)
    mean_bacc, std_bacc = mean_std(baccs)
    mean_mcc, std_mcc = mean_std(mccs)
    mean_sens, std_sens = mean_std(senss)
    mean_spec, std_spec = mean_std(specs)

    mean_auc = 0.0
    mean_prauc = 0.0
    if aucs:
        mean_auc, _ = mean_std(aucs)
        mean_prauc, _ = mean_std(praucs)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_images": int(len(y)),
        "models": list(models_subset),
        "mode": "stacking",
        "meta": "svm",
        "tune": bool(args.tune),
        "calibrate": bool(args.calibrate),
        "calib_method": args.calib_method,
        "cv": int(args.cv),
        "rings": {"ring1": int(args.ring1), "ring2": int(args.ring2)},
        "glcm": bool(not args.no_glcm),
        "results": {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_bacc": mean_bacc,
            "std_bacc": std_bacc,
            "mean_mcc": mean_mcc,
            "std_mcc": std_mcc,
            "mean_sens": mean_sens,
            "std_sens": std_sens,
            "mean_spec": mean_spec,
            "std_spec": std_spec,
            "mean_auc": float(mean_auc),
            "mean_prauc": float(mean_prauc),
        },
    }

    with open(os.path.join(out_dir_exp, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="./out_classic")
    ap.add_argument("--cv", type=int, default=5)

    ap.add_argument(
        "--models",
        nargs="+",
        default=["extratrees", "xgboost"],
        choices=["extratrees", "rf", "svm", "knn", "logreg", "xgboost"],
    )

    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib_method", default="sigmoid", choices=["sigmoid", "isotonic"])

    ap.add_argument("--ring1", type=int, default=10)
    ap.add_argument("--ring2", type=int, default=24)
    ap.add_argument("--no_glcm", action="store_true")

    ap.add_argument("--save_debug_masks", action="store_true")
    ap.add_argument("--save_failures", action="store_true")

    ap.add_argument("--seed", type=int, default=42)

    # Sweep
    ap.add_argument("--sweep_all_combos", action="store_true",
                    help="Executa todas as combinacoes nao vazias de modelos base, usando stacking e meta=svm, salvando cada uma em uma pasta.")
    ap.add_argument("--sweep_min_k", type=int, default=1)
    ap.add_argument("--sweep_max_k", type=int, default=0, help="0 significa sem limite, usa ate o total de modelos disponiveis")
    ap.add_argument("--sweep_models_pool", nargs="+", default=None,
                    help="Lista de modelos a considerar no sweep. Se vazio, usa todos disponiveis no ambiente.")

    args = ap.parse_args()
    seed_everything(args.seed)

    if args.ring2 <= args.ring1 + 2:
        raise RuntimeError("ring2 must be larger than ring1 by at least 3 pixels")

    ensure_dir(args.out_dir)

    samples = list_samples(args.data_root)
    print(f"[info] Found {len(samples)} images")
    print(f"[info] rings, ring1={args.ring1}, ring2={args.ring2}, glcm={not args.no_glcm}")
    print(f"[info] tune={args.tune}, calibrate={args.calibrate} ({args.calib_method})")
    print(f"[info] save_debug_masks={args.save_debug_masks}, save_failures={args.save_failures}")

    # Features base, calculadas uma vez para o sweep inteiro
    X, y, paths, feat_names = build_feature_matrix(
        samples,
        ring1_px=args.ring1,
        ring2_px=args.ring2,
        want_glcm=not args.no_glcm,
        out_debug_dir=None,
        max_debug=0,
    )

    if args.sweep_all_combos:
        pool = args.sweep_models_pool if args.sweep_models_pool else _available_model_names()
        avail = set(_available_model_names())
        pool = [m for m in pool if m in avail]
        if not pool:
            raise RuntimeError("sweep_models_pool vazio, ou nenhum modelo do pool esta disponivel no ambiente")

        max_k = args.sweep_max_k if args.sweep_max_k and args.sweep_max_k > 0 else len(pool)
        min_k = max(1, int(args.sweep_min_k))
        max_k = min(len(pool), int(max_k))
        if min_k > max_k:
            raise RuntimeError("sweep_min_k maior que sweep_max_k")

        leaderboard_path = os.path.join(args.out_dir, "leaderboard.csv")
        rows = []

        total = 0
        for k in range(min_k, max_k + 1):
            total += len(list(itertools.combinations(pool, k)))
        print(f"[info] Sweep pool={pool}, combos={total}, stacking=True, meta=svm")

        for k in range(min_k, max_k + 1):
            for subset in itertools.combinations(pool, k):
                subset = list(subset)
                exp_dir = os.path.join(args.out_dir, _combo_dir_name(subset))
                print(f"\n[info] Running models={subset}, out={exp_dir}")

                summary = run_one_experiment(
                    X=X, y=y, paths=paths, feat_names=feat_names,
                    args=args,
                    models_subset=subset,
                    out_dir_exp=exp_dir,
                )

                r = summary["results"]
                rows.append({
                    "models": "+".join(subset),
                    "mean_acc": r["mean_acc"],
                    "mean_bacc": r["mean_bacc"],
                    "mean_mcc": r["mean_mcc"],
                    "mean_auc": r["mean_auc"],
                    "mean_prauc": r["mean_prauc"],
                    "out_dir": exp_dir,
                })

                # salva incrementalmente
                with open(leaderboard_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    for rr in sorted(rows, key=lambda z: (z["mean_mcc"], z["mean_acc"]), reverse=True):
                        w.writerow(rr)

        print(f"\n[info] Sweep finished, leaderboard saved at {leaderboard_path}")
        return

    # Run único
    exp_dir = args.out_dir
    print(f"[info] Single run, models={list(args.models)}, stacking=True, meta=svm, out={exp_dir}")
    summary = run_one_experiment(
        X=X, y=y, paths=paths, feat_names=feat_names,
        args=args,
        models_subset=list(args.models),
        out_dir_exp=exp_dir,
    )
    print(f"[info] Done. Summary saved to {os.path.join(exp_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
