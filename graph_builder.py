# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:39:20 2025

@author: nina c
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

严格文件名匹配（主名=文件名去扩展名，大小写不敏感），三分类（NC/EMCI/LMCI），
结构监督功能网络 + 被试级稀疏图。

输入目录（可多层）：
  DTI/  NC|EMCI|LMCI/...  (R×R, .txt|.npy)
  fMRI/ NC|EMCI|LMCI/...  (T×R, .txt|.npy)
表型：CSV/Excel（主ID列与可选的Alt ID列按exact匹配；诊断列映射为NC/EMCI/LMCI）

输出（out_dir）：
- A_subject.npy (N,N)                 被试级稀疏图（行归一）
- W_fused.npy   (N,R,R)               个体融合网络（功能+结构监督）
- X_nodes.npy   (N,pca_dim 或更小)    被试特征（上三角 + 可选拼DTI，上PCA）
- labels.npy     (N,)                 三分类标签（0/2/3）
- subject_ids.npy (N,)                主名（文件主名，大小写不敏感）
- label_mapping.json                  {"NC":0,"EMCI":2,"LMCI":3}
- build_graph_config.json             运行配置
- audit_summary.json / audit_*.csv/json 审计结果
"""

import os
import re
import json
import math
import argparse
import csv
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ----------------------------------------------------
def norm_exact_name(s: str) -> str:
    return str(s).strip().upper()

def file_stem(path: str) -> str:
    return norm_exact_name(os.path.splitext(os.path.basename(path))[0])

def normalize_class_name(raw: str) -> str:
    raw = (raw or "").strip().upper()
    if raw in ("HC","CN","CONTROL","NORMAL","NC"): return "NC"
    if raw.startswith("EMCI"): return "EMCI"
    if raw.startswith("LMCI") or raw=="MCI" or raw.startswith("MCI_"): return "LMCI"
    return raw  # 其它（如 SMC/SCD/SCI）不在三分类，后续会被过滤

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5*(M + M.T)

def zscore_ts(Y: np.ndarray, axis=0, eps=1e-8) -> np.ndarray:
    mu = Y.mean(axis=axis, keepdims=True)
    sd = Y.std(axis=axis, keepdims=True) + eps
    return (Y - mu) / sd

def corrcoef_mat(ts: np.ndarray) -> np.ndarray:
    ts = zscore_ts(ts, axis=0)
    R = np.nan_to_num(ts.T @ ts / max(ts.shape[0]-1, 1), nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R

def sliding_windows(T: int, win: int, step: int):
    out, s = [], 0
    while s + win <= T:
        out.append((s, s+win)); s += step
    return out or [(0, T)]

def minmax01(A: np.ndarray, eps=1e-12) -> np.ndarray:
    a, b = float(np.min(A)), float(np.max(A))
    if b - a < eps: return np.zeros_like(A)
    return (A - a) / (b - a)

def upper_tri_vector(M: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=k)
    return M[iu]

def row_topk_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    N = A.shape[0]
    A = A.copy()
    np.fill_diagonal(A, -np.inf)
    kk = max(1, min(k, N-1))
    for i in range(N):
        keep = np.argpartition(A[i], -kk)[-kk:]
        mask = np.ones(N, dtype=bool); mask[keep] = False
        A[i, mask] = 0.0
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T)  # 并集对称

def row_normalize(A: np.ndarray, eps=1e-12) -> np.ndarray:
    d = A.sum(axis=1, keepdims=True) + eps
    return A / d

# -------------------------- 结构监督的个体功能网络 --------------------------
def fused_fcn_from_timeseries(ts: np.ndarray, sc: Optional[np.ndarray],
                              win=35, step=5, lam=0.05, n_iters=200, lr=0.5, eps=1e-6) -> np.ndarray:
    T, R = ts.shape
    sFC = corrcoef_mat(ts)
    dFC = np.mean([corrcoef_mat(ts[a:b]) for a,b in sliding_windows(T, win, step)], axis=0) if T > win else sFC
    W = 0.5 * (sFC + dFC); np.fill_diagonal(W, 0.0)

    if sc is None:
        C = np.ones((R, R), np.float32)
    else:
        S = sc / (np.max(sc) + eps)
        C = 1.0 / (S + eps)            # 结构强→惩罚小；结构弱→惩罚大
        C = minmax01(C); C = 0.75*C + 0.25
    np.fill_diagonal(C, np.inf)

    Y = zscore_ts(ts, 0)
    G = Y.T @ Y
    L = np.linalg.norm(G, 2) + eps
    step_size = lr / L

    for _ in range(n_iters):
        grad = 2.0 * (G @ W - G)
        W   -= step_size * grad
        thr  = step_size * lam * C
        W    = np.sign(W) * np.maximum(np.abs(W) - thr, 0.0)
        np.fill_diagonal(W, 0.0)

    W = symmetrize(W)
    return np.clip(np.nan_to_num(W), -1.0, 1.0)

# -------------------------- 读文件 & 扫描（exact） --------------------------
def load_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path) if path.endswith(".txt") else np.load(path)

def scan_modality_exact(root: str):
    """
    返回：
      name2mat: {MAIN_NAME -> ndarray}
      name2class: {MAIN_NAME -> class_name}
      duplicates: {MAIN_NAME: [paths...]} // 仅记录重复项（择优保留一份）
    类别来自路径中任意包含 NC/EMCI/LMCI（或 HC/CN→NC）；否则保留原片段首层名。
    同名重复择优规则：.npy > .txt；若相同扩展名，保留文件更大者。
    """
    name2mat, name2class, duplicates = {}, {}, {}
    if not root or not os.path.isdir(root): return name2mat, name2class, duplicates

    def path_class(p: str) -> str:
        parts = [normalize_class_name(x) for x in p.split(os.sep)]
        for x in parts:
            if x in ("NC","EMCI","LMCI"): return x
        return parts[0] if parts else ""

    def better_choice(old_path: str, new_path: str) -> str:
        # 选择更优文件：优先 .npy，其次更大文件
        def score(p):
            ext = os.path.splitext(p)[1].lower()
            ext_score = 1 if ext==".npy" else 0
            try:
                size = os.path.getsize(p)
            except:
                size = 0
            return (ext_score, size)
        return new_path if score(new_path) > score(old_path) else old_path

    chosen_path = {}  # main_name -> best file path

    for subdir, _, files in os.walk(root):
        rel = os.path.relpath(subdir, root)
        if rel == ".": 
            continue
        cls = path_class(rel)
        for fname in files:
            if not (fname.endswith(".txt") or fname.endswith(".npy")):
                continue
            main = file_stem(fname)  # exact 主名
            path = os.path.join(subdir, fname)
            if main in chosen_path:
                # 记录重复，同时择优
                duplicates.setdefault(main, []).append(path)
                best = better_choice(chosen_path[main], path)
                chosen_path[main] = best
            else:
                chosen_path[main] = path
                duplicates.setdefault(main, [])

            # 同步类别：若已有，优先保留已识别到的三类
            if main not in name2class:
                name2class[main] = cls
            else:
                if name2class[main] not in ("NC","EMCI","LMCI") and cls in ("NC","EMCI","LMCI"):
                    name2class[main] = cls

    # 读取择优后的矩阵
    for main, path in chosen_path.items():
        try:
            arr = load_matrix(path).astype(np.float32)
        except Exception:
            continue
        name2mat[main] = arr

    return name2mat, name2class, {k:v for k,v in duplicates.items() if len(v)>0}

# -------------------------- 表型（exact 主名/备选名，都必须 exact 命中） --------------------------
def load_phenotype_exact(pheno_path: Optional[str]) -> Optional[Tuple[pd.DataFrame, Dict[str,int]]]:
    if not pheno_path or not os.path.isfile(pheno_path): return None
    ext = os.path.splitext(pheno_path)[-1].lower()
    try:
        if ext in [".xlsx",".xls"]:
            df = pd.read_excel(pheno_path, engine="openpyxl")
        else:
            df = pd.read_csv(pheno_path)
    except Exception:
        return None

    df = df.copy(); df.columns = [c.strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}
    def pick(*keys):
        for k in keys:
            if k.lower() in cmap: return cmap[k.lower()]
        return None

    main_id = pick("Subject ID","Subject_ID","ID","subject_id","sid","participant","SubjectID","Name","Code","组ID","受试者")
    if main_id is None: main_id = df.columns[0]
    alt_id  = pick("Alt ID","AltID","Code","code","ShortID","Short ID","SecondID","第二ID","次要ID")

    col_sex = pick("sex","Sex","gender")
    col_age = pick("age","Age")
    col_ctr = pick("center","site","SITE_ID","Site","CENTER")
    col_site= pick("batch","scanner","site2","Site2","SITE","S")
    col_dx  = pick("DX Group","diagnosis","DX","Diagnosis","Group","Label","Class","组别")

    out = pd.DataFrame()
    out["MAIN_NAME"] = df[main_id].astype(str).map(norm_exact_name)
    out["ALT_NAME"]  = df[alt_id].astype(str).map(norm_exact_name) if alt_id is not None else ""

    def norm_sex(x):
        s = str(x).strip().upper()
        if s in ("M","1","MALE"): return 1
        if s in ("F","0","FEMALE"): return 0
        return None
    def norm_dx(x):
        s = str(x).strip().upper()
        if s in ("HC","CN","CONTROL","NORMAL","NC"): return "NC"
        if s.startswith("EMCI"): return "EMCI"
        if s.startswith("LMCI") or s=="MCI" or s.startswith("MCI_"): return "LMCI"
        return None

    out["sex"]    = df[col_sex].apply(norm_sex) if col_sex is not None else None
    out["age"]    = pd.to_numeric(df[col_age], errors="coerce") if col_age is not None else None
    out["center"] = df[col_ctr].astype(str).str.strip() if col_ctr is not None else None
    out["site"]   = df[col_site].astype(str).str.strip() if col_site is not None else None
    out["dx"]     = df[col_dx].apply(norm_dx) if col_dx is not None else None

    # exact 索引（主名/备选名都要求完全相等）
    idx = {}
    for i, r in out.iterrows():
        m = r["MAIN_NAME"]; a = r["ALT_NAME"]
        if m: idx[m] = i
        if a: idx[a] = i
    return out, idx

# -------------------------- 审计记录器 --------------------------
class Auditor:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.skips = []  # {sid_or_name, reason, detail}
        os.makedirs(out_dir, exist_ok=True)
    def add(self, sid: str, reason: str, detail: str = ""):
        self.skips.append({"sid_or_name": sid, "reason": reason, "detail": detail})
    def write_sets(self, dti_names, fmri_names, pheno_names, inter_names):
        with open(os.path.join(self.out_dir, "audit_id_sets.json"), "w", encoding="utf-8") as f:
            json.dump({
                "DTI_names_count": len(dti_names),
                "fMRI_names_count": len(fmri_names),
                "pheno_names_count": len(pheno_names),
                "intersection_count": len(inter_names),
                "note": "exact 主名（大小写不敏感）"
            }, f, ensure_ascii=False, indent=2)
    def write_list_csv(self, fname: str, header: List[str], rows: List[Tuple]):
        with open(os.path.join(self.out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows: w.writerow(list(r))
    def write_missing(self):
        with open(os.path.join(self.out_dir, "audit_missing_ids.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sid_or_name","reason","detail"])
            w.writeheader(); w.writerows(self.skips)
    def write_summary(self, dti_count, fmri_count, pheno_count, inter_count, paired, used):
        by_reason = {}
        for r in self.skips:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        with open(os.path.join(self.out_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "counts": {
                    "DTI_names": dti_count,
                    "fMRI_names": fmri_count,
                    "pheno_names": pheno_count,
                    "intersection": inter_count,
                    "paired": paired,
                    "used": used
                },
                "skip_reasons": by_reason
            }, f, ensure_ascii=False, indent=2)

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser(description="Exact 文件名配对 + 三分类(NC/EMCI/LMCI) + 审计")
    ap.add_argument("--dti_dir", required=True)
    ap.add_argument("--fmri_dir", required=True)
    ap.add_argument("--pheno_path", required=True)
    ap.add_argument("--out_dir", required=True)

    # 个体网络
    ap.add_argument("--win", type=int, default=35)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--n_iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)

    # 被试图
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)

    # 特征
    ap.add_argument("--feature", choices=["wfunc","wfunc_plus_sc"], default="wfunc")
    ap.add_argument("--pca_dim", type=int, default=128)

    # 三分类（默认，无 SMC）
    ap.add_argument("--label_map_json", type=str, default='{"NC":0,"EMCI":2,"LMCI":3}')

    ap.add_argument("--export_per_subject", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    auditor = Auditor(args.out_dir)

    # 1) exact 扫描 + 重名择优
    dti_name2mat, dti_name2cls, dti_dups = scan_modality_exact(args.dti_dir)
    fmri_name2mat, fmri_name2cls, fmri_dups = scan_modality_exact(args.fmri_dir)
    for m, paths in dti_dups.items(): auditor.add(m, "duplicate_in_DTI", f"extra_files={len(paths)}")
    for m, paths in fmri_dups.items(): auditor.add(m, "duplicate_in_fMRI", f"extra_files={len(paths)}")

    dti_names  = set(dti_name2mat.keys())
    fmri_names = set(fmri_name2mat.keys())
    paired_names = sorted(dti_names & fmri_names)

    # 2) 表型（exact）
    pheno = load_phenotype_exact(args.pheno_path)
    if pheno is None:
        raise ValueError("无法读取表型（CSV/Excel）。")
    pheno_df, pheno_index = pheno
    pheno_names = set(pheno_index.keys())
    inter_names = set(paired_names) & pheno_names

    auditor.write_sets(dti_names, fmri_names, pheno_names, inter_names)
    auditor.write_list_csv("audit_paired_ids.csv", ["paired_main_name"], [(n,) for n in paired_names])

    # 来源缺失（审计）
    for n in sorted(dti_names - fmri_names): auditor.add(n, "missing_in_fMRI", "DTI有，fMRI缺")
    for n in sorted(fmri_names - dti_names): auditor.add(n, "missing_in_DTI", "fMRI有，DTI缺")
    for n in sorted((dti_names & fmri_names) - pheno_names): auditor.add(n, "missing_in_pheno", "影像都有，表型缺")

    # 3) 逐主名构图
    label_map = json.loads(args.label_map_json)
    valid_classes = set(label_map.keys())  # {"NC","EMCI","LMCI"}

    subjects, W_fused, feat_list, labels = [], [], [], []
    metas = {"sex":[], "age":[], "center":[], "status":[], "site":[]}
    used_names = []
    R = None

    for main in paired_names:
        sc = dti_name2mat.get(main); ts = fmri_name2mat.get(main)
        cls_dir = normalize_class_name(dti_name2cls.get(main, fmri_name2cls.get(main, "")))
        # exact 表型命中
        row = pheno_df.iloc[pheno_index[main]] if main in pheno_index else None

        # 形状检查
        if sc is None or ts is None or ts.ndim != 2:
            auditor.add(main, "load_error_or_bad_ts", "sc为None或ts.ndim!=2")
            continue
        T, R_cur = ts.shape
        if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
            auditor.add(main, "sc_not_square", f"sc shape={getattr(sc,'shape',None)}")
            continue
        if sc.shape[0] != R_cur:
            auditor.add(main, "roi_mismatch", f"DTI R={sc.shape[0]} vs fMRI R={R_cur}")
            continue
        if R is None: R = R_cur
        if R != R_cur:
            auditor.add(main, "roi_inconsistent_across_subjects", f"expect R={R}, got {R_cur}")
            continue

        # 个体融合网络
        try:
            W = fused_fcn_from_timeseries(ts, sc, args.win, args.step, args.lam, args.n_iters, args.lr)
        except Exception as e:
            auditor.add(main, "fuse_fail", f"{type(e).__name__}: {e}")
            continue
        W_fused.append(W)

        # 特征
        tri = upper_tri_vector(W, 1)
        if args.feature == "wfunc_plus_sc":
            tri_sc = upper_tri_vector(symmetrize(sc), 1)
            xi = np.concatenate([tri, tri_sc], axis=0)
        else:
            xi = tri
        feat_list.append(xi)
        subjects.append(main)
        used_names.append(main)

        # 三分类标签：表型优先；不在三类则用目录回退；最终必须在三类，否则剔除
        dx_ph = row["dx"] if (row is not None and "dx" in row) else None
        dx_dir = cls_dir if cls_dir in valid_classes else None
        dx = dx_ph if dx_ph in valid_classes else dx_dir
        if dx not in valid_classes:
            # 回滚
            subjects.pop(); used_names.pop(); W_fused.pop(); feat_list.pop()
            auditor.add(main, "label_not_in_mapping", f"dx_pheno={dx_ph}, dx_dir={cls_dir}")
            continue
        metas["status"].append(dx)

        # 人口学
        sex = row["sex"] if (row is not None and "sex" in row) else None
        age = row["age"] if (row is not None and "age" in row) else None
        ctr = row["center"] if (row is not None and "center" in row) else None
        site= row["site"] if (row is not None and "site" in row) else None
        metas["sex"].append(sex if (sex is None or pd.isna(sex))==False else None)
        metas["age"].append(float(age) if (age is not None and pd.notna(age)) else None)
        metas["center"].append(ctr if (ctr is not None and str(ctr).strip()!="") else None)
        metas["site"].append(site if (site is not None and str(site).strip()!="") else None)
        labels.append(int(label_map[dx]))

    if len(subjects) == 0:
        auditor.write_missing()
        auditor.write_summary(len(dti_names), len(fmri_names), len(pheno_names),
                              len(inter_names), len(paired_names), 0)
        raise ValueError("没有可用受试者（查看审计文件）。")

    auditor.write_list_csv("audit_used_ids.csv", ["used_main_name"], [(n,) for n in used_names])

    # 4) 组装矩阵/特征 & PCA
    N = len(subjects)
    W_fused = np.stack(W_fused, axis=0)
    X_raw   = np.stack(feat_list, axis=0)
    if args.pca_dim and args.pca_dim > 0:
        k = min(args.pca_dim, X_raw.shape[1], N)
        X_nodes = PCA(n_components=k, random_state=args.seed).fit_transform(X_raw)
    else:
        X_nodes = X_raw

    # 5) 被试级稀疏图
    def r_gender(gi, gj):  return 0.0 if (gi is None or gj is None) else (1.0 if gi==gj else 0.0)
    def r_age(ai, aj, a_std):
        if ai is None or aj is None: return 0.0
        try: return 1.0 if abs(float(ai)-float(aj)) <= float(a_std) else 0.0
        except: return 0.0
    def r_center(ci, cj):  return 0.0 if (not ci or not cj) else (1.0 if str(ci)==str(cj) else 0.0)
    def r_status(si, sj):  return 0.0 if (si is None or sj is None) else (1.0 if str(si)==str(sj) else 0.0)
    def r_site(si, sj):    return 0.0 if (not si or not sj) else (1.0 if str(si)==str(sj) else 0.0)

    age_vals = np.array([a for a in metas["age"] if a is not None], dtype=float)
    a_std = float(age_vals.std()) if age_vals.size > 0 else 0.0

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        xi = X_nodes[i]; xi = (xi - xi.mean())/(xi.std()+1e-8)
        for j in range(i+1, N):
            xj = X_nodes[j]; xj = (xj - xj.mean())/(xj.std()+1e-8)
            r  = float(np.dot(xi, xj)/max(len(xi)-1, 1)); r = max(min(r,1.0),-1.0)
            rho = 1.0 - r
            sim_ij = math.exp( - (rho**2) / (2.0*(args.sigma**2)) )
            rsum = (
                r_gender(metas["sex"][i], metas["sex"][j]) +
                r_age   (metas["age"][i], metas["age"][j], a_std) +
                r_center(metas["center"][i], metas["center"][j]) +
                r_status(metas["status"][i], metas["status"][j]) +
                r_site  (metas["site"][i], metas["site"][j])
            )
            Aij = sim_ij * (1.0 + rsum)
            A[i,j] = A[j,i] = Aij

    A_sparse  = row_topk_sparsify(A, args.topk)
    A_subject = row_normalize(A_sparse)

    # 6) 保存
    np.save(os.path.join(args.out_dir, "A_subject.npy"), A_subject)
    np.save(os.path.join(args.out_dir, "W_fused.npy"), W_fused)
    np.save(os.path.join(args.out_dir, "X_nodes.npy"), X_nodes)
    np.save(os.path.join(args.out_dir, "labels.npy"), np.array(labels, dtype=int))
    np.save(os.path.join(args.out_dir, "subject_ids.npy"), np.array(subjects))
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(json.loads(args.label_map_json), f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "build_graph_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dti_dir": args.dti_dir, "fmri_dir": args.fmri_dir, "pheno_path": args.pheno_path,
            "out_dir": args.out_dir, "win": args.win, "step": args.step, "lam": args.lam,
            "n_iters": args.n_iters, "lr": args.lr, "sigma": args.sigma, "topk": args.topk,
            "feature": args.feature, "pca_dim": args.pca_dim, "seed": args.seed,
            "label_map": json.loads(args.label_map_json),
            "note": "exact 文件名匹配；三分类（NC/EMCI/LMCI）；SMC自动排除；重复主名优先 .npy > .txt，其次文件更大。"
        }, f, ensure_ascii=False, indent=2)

    auditor.write_missing()
    auditor.write_summary(len(dti_names), len(fmri_names), len(pheno_names),
                          len(inter_names), len(paired_names), len(subjects))

    if args.export_per_subject:
        subdir = os.path.join(args.out_dir, "per_subject")
        os.makedirs(subdir, exist_ok=True)
        for i, sid in enumerate(subjects):
            np.save(os.path.join(subdir, f"{sid}_Wfused.npy"), W_fused[i])

    print(f"[OK] used={len(subjects)} / paired={len(paired_names)} | R={W_fused.shape[1]} | out -> {args.out_dir}")

if __name__ == "__main__":
    main()