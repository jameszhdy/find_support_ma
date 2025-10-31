#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_support_ma.py

为仓库中 config.json 的每个组合，寻找“支撑性最强”的移动平均线 MA。
输出：
  - support_ma_results/<portfolio>_support.csv  （每个 MA 的指标）
  - support_ma_summary.csv （每个组合推荐 Top-3 MA）
参数可通过脚本顶部的 DEFAULTS 调整。
"""
import os
import math
import json
import time
import logging
import argparse
import datetime as dt
from io import StringIO
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

LOG = logging.getLogger("find_support_ma")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")
OUT_DIR = os.path.join(THIS_DIR, "support_ma_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ====== DEFAULTS (可调) ======
MA_GRID = list(range(20, 301, 5))   # 测试 MA 范围
RECOVER_WINDOW = 60                 # 触及后观察 recovery 的天数（交易日）
MIN_TOUCHES = 2                     # 至少多少次触及才算有代表性
ALPHA_DEPTH = 4.0                   # 深度惩罚系数（越大越惩罚深回撤）
# =============================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# ---------- config loader ----------
def load_config(path=CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- fund NAV fetch (eastmoney pingzhongdata -> sina fallback) ----------
def ms_to_date(ms:int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit='ms').normalize()

def fetch_nav_from_eastmoney(code:str, timeout=8) -> pd.Series:
    url = f"https://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time()*1000)}"
    headers = HEADERS.copy()
    headers["Referer"] = f"https://fund.eastmoney.com/{code}.html"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    text = r.text

    # try AC
    m_ac = re.search(r'var\s+Data_ACWorthTrend\s*=\s*(\[.*?\])\s*;', text, flags=re.S)
    if m_ac:
        arr = json.loads(m_ac.group(1))
        dates = [ms_to_date(int(item[0])) for item in arr]
        vals = [float(item[1]) for item in arr]
        return pd.Series(vals, index=pd.to_datetime(dates)).sort_index()

    # try netWorthTrend
    m_net = re.search(r'var\s+Data_netWorthTrend\s*=\s*(\[.*?\])\s*;', text, flags=re.S)
    if m_net:
        arr = json.loads(m_net.group(1))
        dates = [ms_to_date(int(item["x"])) for item in arr]
        vals = [float(item["y"]) for item in arr]
        return pd.Series(vals, index=pd.to_datetime(dates)).sort_index()

    raise RuntimeError("EastMoney: 未匹配到净值数据")

def fetch_nav_from_sina(code:str, timeout=8) -> pd.Series:
    url = f"https://stock.finance.sina.com.cn/fundInfo/view/FundInfo_LSJZ.php?symbol={code}"
    headers = HEADERS.copy()
    headers["Referer"] = "https://finance.sina.com.cn/"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    try:
        dfs = pd.read_html(StringIO(html))
    except Exception:
        dfs = []

    target_df = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("单位净值" in c or "累计净值" in c or "净值" in c for c in cols):
            target_df = df
            break

    if target_df is None:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        for t in tables:
            try:
                df_try = pd.read_html(StringIO(str(t)))[0]
                cols = [str(c) for c in df_try.columns]
                if any("单位净值" in c or "累计净值" in c or "净值" in c for c in cols):
                    target_df = df_try
                    break
            except Exception:
                continue

    if target_df is None or target_df.empty:
        raise RuntimeError("新浪页面未解析到净值表格")

    if isinstance(target_df.columns, pd.MultiIndex):
        target_df.columns = ["".join([str(x) for x in tup]) for tup in target_df.columns]
    date_col = None
    nav_col = None
    for c in target_df.columns:
        s = str(c)
        if "日期" in s or "净值日期" in s or "时间" in s:
            date_col = c
        if "单位净值" in s or "累计净值" in s or "净值" in s:
            nav_col = c
    if date_col is None or nav_col is None:
        raise RuntimeError("新浪解析：未找到日期/净值列")

    df = target_df[[date_col, nav_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[nav_col] = pd.to_numeric(df[nav_col].astype(str).str.replace('%',''), errors='coerce')
    df = df.dropna(subset=[date_col, nav_col])
    ser = pd.Series(df[nav_col].values, index=df[date_col].dt.normalize())
    ser = ser.sort_index()
    return ser

def fetch_single_fund_nav(code:str, retries:int=2) -> pd.Series:
    last_err = None
    for _ in range(retries):
        try:
            return fetch_nav_from_eastmoney(code)
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    for _ in range(retries):
        try:
            return fetch_nav_from_sina(code)
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"未能获取基金 {code} 的净值（{last_err}）")

def fetch_nav_df_for_codes(codes:List[str], days_back:int=2000) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    series_list = []
    for c in codes:
        LOG.info("拉取 %s", c)
        try:
            s = fetch_single_fund_nav(c)
            s.name = c
            series_list.append(s)
        except Exception as e:
            LOG.error(" 拉取 %s 失败: %s", c, e)
        time.sleep(0.15)
    if not series_list:
        raise ValueError("没有任何基金成功获取数据。")
    df = pd.concat(series_list, axis=1).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df

# ---------- portfolio NAV combine ----------
def make_portfolio_nav(nav_df:pd.DataFrame, fund_weights:Dict[str,float], initial_capital:float=100000.0) -> pd.Series:
    weights = pd.Series(fund_weights, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-8:
        weights = weights / weights.sum()
    sub = nav_df.reindex(columns=list(weights.index)).ffill().dropna(how='any')
    if sub.empty:
        raise RuntimeError("组合数据不足")
    start_prices = sub.iloc[0]
    units = {c: initial_capital * weights[c] / start_prices[c] for c in sub.columns}
    units_s = pd.Series(units)
    value = (sub * units_s).sum(axis=1)
    nv = value / value.iloc[0]
    return nv

# ---------- support analysis on a single port_nv and a given MA ----------
def analyze_support_for_ma(port_nv:pd.Series, M:int, recover_window:int=RECOVER_WINDOW) -> Dict[str,Any]:
    # port_nv: normalized series (index date)
    series = port_nv.copy().sort_index()
    ma = series.rolling(window=M, min_periods=M).mean()
    # find touch events: day where today <= ma and yesterday > ma
    cond_today = (series <= ma)
    cond_yesterday = (series.shift(1) > ma.shift(1))
    events = cond_today & cond_yesterday
    events = events.fillna(False)
    dates = list(series.index[events])
    touch_count = len(dates)
    depths = []
    recovered_flags = []
    times_to_recover = []
    for d in dates:
        # slice from d to d+recover_window (or end)
        start_idx = series.index.get_loc(d)
        end_idx = min(start_idx + recover_window, len(series)-1)
        window_idx = series.index[start_idx:end_idx+1]
        window_prices = series.loc[window_idx]
        # find min price after touch (including touch day)
        min_price = window_prices.min()
        touch_price = series.loc[d]
        depth = (min_price / touch_price) - 1.0  # negative if down
        depths.append(depth)
        # find recovery day: first day after touch where price >= ma (ma computed across series)
        recovered = False
        days_to_recover = None
        for i, idx in enumerate(window_idx):
            if series.loc[idx] >= ma.loc[idx]:
                recovered = True
                days_to_recover = i  # 0 if same day
                break
        recovered_flags.append(recovered)
        times_to_recover.append(days_to_recover if recovered else np.nan)
    # aggregate stats
    if touch_count == 0:
        return {
            "ma": M,
            "touch_count": 0,
            "median_depth": None,
            "mean_depth": None,
            "pct_recovered_within_W": None,
            "median_time_to_recover": None,
            "avg_time_to_recover": None,
            "support_score": None
        }
    median_depth = float(np.median(depths))
    mean_depth = float(np.mean(depths))
    pct_recovered = float(np.nanmean([1.0 if f else 0.0 for f in recovered_flags]))
    median_ttr = float(np.nanmedian([t for t in times_to_recover if t is not None and not np.isnan(t)])) if any([not np.isnan(t) for t in times_to_recover]) else None
    avg_ttr = float(np.nanmean([t for t in times_to_recover if t is not None and not np.isnan(t)])) if any([not np.isnan(t) for t in times_to_recover]) else None
    # scoring
    # penalty for deep negative median_depth: depth is negative; use max(0, -median_depth)
    depth_penalty = math.exp(-ALPHA_DEPTH * max(0.0, -median_depth))
    size_factor = math.log(1.0 + touch_count)
    recover_factor = pct_recovered
    time_factor = 1.0 / (1.0 + (median_ttr if median_ttr is not None else (recover_window*1.0))/30.0)
    score = size_factor * (recover_factor if recover_factor is not None else 0.0) * depth_penalty * time_factor
    return {
        "ma": M,
        "touch_count": touch_count,
        "median_depth": median_depth,
        "mean_depth": mean_depth,
        "pct_recovered_within_W": pct_recovered,
        "median_time_to_recover": median_ttr,
        "avg_time_to_recover": avg_ttr,
        "support_score": score
    }

# ---------- run for each portfolio ----------
def run_all(portfolios:Dict[str,Any], ma_grid:List[int]=MA_GRID, recover_window:int=RECOVER_WINDOW, outdir:str=OUT_DIR):
    # collect unique funds
    fund_codes = sorted({c for p in portfolios.values() for c in p.get('funds', {})})
    LOG.info("需要拉取基金数量: %d", len(fund_codes))
    nav_df = fetch_nav_df_for_codes(fund_codes, days_back=2000)
    LOG.info("拿到净值表: rows=%d cols=%d", len(nav_df), len(nav_df.columns))
    summary_rows = []
    for pname, pconf in portfolios.items():
        LOG.info("分析组合: %s", pname)
        funds = pconf.get('funds', {})
        initial_cap = float(pconf.get('initial_capital', 100000.0))
        try:
            port_nv = make_portfolio_nav(nav_df, funds, initial_cap)
        except Exception as e:
            LOG.error(" 组合 %s 数据不足: %s", pname, e)
            continue
        # ensure enough length
        if len(port_nv) < max(ma_grid) + 10:
            LOG.warning(" 组合 %s 历史数据少于最大 MA，可能导致 NaN", pname)
        results = []
        for m in ma_grid:
            res = analyze_support_for_ma(port_nv, m, recover_window=recover_window)
            results.append(res)
        df = pd.DataFrame(results)
        # add derived fields: require at least MIN_TOUCHES to be considered; otherwise null score
        df['valid'] = df['touch_count'] >= MIN_TOUCHES
        # if touch_count small, set support_score to NaN to avoid false positive
        df.loc[~df['valid'], 'support_score'] = np.nan
        # sort by support_score desc
        df_sorted = df.sort_values(by='support_score', ascending=False)
        # save csv
        fname = os.path.join(outdir, f"{pname.replace(' ','_')}_support.csv")
        df_sorted.to_csv(fname, index=False, float_format="%.6f")
        LOG.info(" 保存 %s", fname)
        # pick top-3 valid
        topk = df_sorted[df_sorted['valid'] & df_sorted['support_score'].notna()].head(3)
        if topk.empty:
            # fallback: choose MA with max support_score even if touch_count < MIN_TOUCHES (but warn)
            fallback = df.sort_values(by='support_score', ascending=False).head(3)
            LOG.warning("组合 %s 没有满足 MIN_TOUCHES=%d 的 MA，返回 fallback top MA (样本量较小)", pname, MIN_TOUCHES)
            picked = fallback
            flag_note = "fallback_low_samples"
        else:
            picked = topk
            flag_note = "ok"
        for _, row in picked.iterrows():
            summary_rows.append({
                "portfolio": pname,
                "ma": int(row['ma']),
                "touch_count": int(row['touch_count']),
                "median_depth": float(row['median_depth']) if not pd.isna(row['median_depth']) else None,
                "pct_recovered_within_W": float(row['pct_recovered_within_W']) if not pd.isna(row['pct_recovered_within_W']) else None,
                "median_time_to_recover": float(row['median_time_to_recover']) if not pd.isna(row['median_time_to_recover']) else None,
                "support_score": float(row['support_score']) if not pd.isna(row['support_score']) else None,
                "flag": flag_note
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_fp = os.path.join(outdir, "support_ma_summary.csv")
    summary_df.to_csv(summary_fp, index=False, float_format="%.6f")
    LOG.info("总体建议保存在 %s", summary_fp)
    return summary_df

# ---------- helpers reused (make_portfolio_nav and fetch_nav_df_for_codes) ----------
def fetch_nav_df_for_codes(codes:List[str], days_back:int=2000) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    series_list = []
    for c in codes:
        LOG.info("拉取 %s", c)
        try:
            s = fetch_single_fund_nav(c)
            s.name = c
            series_list.append(s)
        except Exception as e:
            LOG.error(" 拉取 %s 失败: %s", c, e)
        time.sleep(0.1)
    if not series_list:
        raise ValueError("没有任何基金成功获取数据。")
    df = pd.concat(series_list, axis=1).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df

def make_portfolio_nav(nav_df:pd.DataFrame, fund_weights:Dict[str,float], initial_capital:float=100000.0) -> pd.Series:
    weights = pd.Series(fund_weights, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-8:
        weights = weights / weights.sum()
    sub = nav_df.reindex(columns=list(weights.index)).ffill().dropna(how='any')
    if sub.empty:
        raise RuntimeError("组合数据不足")
    start_prices = sub.iloc[0]
    units = {c: initial_capital * weights[c] / start_prices[c] for c in sub.columns}
    units_s = pd.Series(units)
    val = (sub * units_s).sum(axis=1)
    nv = val / val.iloc[0]
    return nv

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ma_min", type=int, default=MA_GRID[0])
    parser.add_argument("--ma_max", type=int, default=MA_GRID[-1])
    parser.add_argument("--ma_step", type=int, default=MA_GRID[1]-MA_GRID[0])
    parser.add_argument("--recover_window", type=int, default=RECOVER_WINDOW)
    parser.add_argument("--min_touches", type=int, default=MIN_TOUCHES)
    parser.add_argument("--alpha_depth", type=float, default=ALPHA_DEPTH)
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        LOG.setLevel(logging.DEBUG)
    # rebuild globals from args
    MA_GRID = list(range(args.ma_min, args.ma_max+1, args.ma_step))
    RECOVER_WINDOW = args.recover_window
    MIN_TOUCHES = args.min_touches
    ALPHA_DEPTH = args.alpha_depth
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = load_config()
    portfolios = cfg.get("portfolios", {})
    summary = run_all(portfolios, MA_GRID, recover_window=RECOVER_WINDOW, outdir=OUT_DIR)
    print("完成。建议保存在:", os.path.join(OUT_DIR, "support_ma_summary.csv"))
