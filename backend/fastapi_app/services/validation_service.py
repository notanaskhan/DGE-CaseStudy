
import os, json, math, re, datetime as dt
from typing import Dict, Any, List, Optional

DATE_CANDIDATES = ["date","txn date","transaction date","posting date","value date","dt"]

def _parse_float(s) -> float:
    if s is None: return 0.0
    if isinstance(s, (int, float)): return float(s)
    x = str(s).strip()
    if not x: return 0.0
    neg = x.startswith("(") and x.endswith(")")
    import re
    x = re.sub(r"[^\d,.\-]", "", x)
    if "," in x and "." in x:
        if x.rfind(".") > x.rfind(","):
            x = x.replace(",", "")
        else:
            x = x.replace(".", "").replace(",", ".")
    else:
        x = x.replace(",", "")
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return -v if neg and v >= 0 else v

def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    norm = [c.strip().lower() for c in cols]
    for cand in candidates:
        c = cand.lower()
        for i, col in enumerate(norm):
            if col == c or c in col:
                return cols[i]
    return None

def _read_df(path: str):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if ext in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(path, dtype=str, engine="openpyxl")
        except Exception:
            return pd.read_excel(path, dtype=str)
    return None

def _detect_amount_columns(df):
    amount_candidates = ["amount","transaction amount","amt","value","amount (aed)","aed amount","amount aed","total amount"]
    amount_col = _find_col(list(df.columns), amount_candidates)
    credit_candidates = ["credit","cr","deposit","inflow","paid in","credits"]
    debit_candidates  = ["debit","dr","withdrawal","outflow","paid out","debits"]
    credit_col = _find_col(list(df.columns), credit_candidates)
    debit_col  = _find_col(list(df.columns), debit_candidates)
    return amount_col, credit_col, debit_col

def _coerce_amount_series(df, amount_col):
    return df[amount_col].apply(_parse_float)

def _infer_txn_amount(df):
    amount_col, credit_col, debit_col = _detect_amount_columns(df)
    if amount_col:
        s = _coerce_amount_series(df, amount_col)
        return list(s.values)
    if credit_col or debit_col:
        cr = df[credit_col].apply(_parse_float) if credit_col in df.columns else 0.0
        dr = df[debit_col].apply(_parse_float) if debit_col in df.columns else 0.0
        cr_vals = list(getattr(cr, "values", [])) if hasattr(cr, "values") else []
        dr_vals = list(getattr(dr, "values", [])) if hasattr(dr, "values") else []
        if cr_vals and dr_vals and len(cr_vals) == len(dr_vals):
            return [float(cr_vals[i]) - float(dr_vals[i]) for i in range(len(cr_vals))]
        if cr_vals and not dr_vals:
            return [float(v) for v in cr_vals]
        if dr_vals and not cr_vals:
            return [-float(v) for v in dr_vals]
    best = None; best_sum = 0.0
    for c in df.columns:
        try:
            vals = df[c].apply(_parse_float)
            score = float(vals.abs().sum())
            if score > best_sum:
                best, best_sum = list(vals.values), score
        except Exception:
            pass
    return best

def _detect_date_column(df):
    return _find_col(list(df.columns), DATE_CANDIDATES)

def _parse_dates(series):
    import pandas as pd
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    except Exception:
        return None

def summarize_inflows_by_window(df, today: dt.date) -> Dict[str, Any]:
    import pandas as pd
    if df is None or df.empty:
        return {"has_dates": False, "reason": "empty df"}

    date_col = _detect_date_column(df)
    if not date_col:
        return {"has_dates": False, "reason": "no_date_column"}

    dates = _parse_dates(df[date_col])
    if dates is None or dates.isna().all():
        return {"has_dates": False, "reason": "unparsable_dates"}

    amounts = _infer_txn_amount(df)
    if not amounts:
        return {"has_dates": True, "reason": "no_amounts"}

    tmp = pd.DataFrame({"date": dates, "amt": amounts}).dropna(subset=["date"])
    if tmp.empty:
        return {"has_dates": True, "reason": "no_dated_rows"}

    def sum_in_window(days: int) -> float:
        start = pd.Timestamp(today - dt.timedelta(days=days))
        end   = pd.Timestamp(today)
        m = (tmp["date"] >= start) & (tmp["date"] <= end)
        inflow = tmp.loc[m & (tmp["amt"] > 0), "amt"].sum()
        return float(inflow)

    return {
        "has_dates": True,
        "last_30d_inflow":  sum_in_window(30),
        "last_60d_inflow":  sum_in_window(60),
        "last_90d_inflow":  sum_in_window(90),
        "row_count": int(len(tmp))
    }

def validate_application(app_row: Dict[str, Any], doc_paths: List[str], tolerance: float = 0.25) -> Dict[str, Any]:
    import pandas as pd
    declared = float(app_row.get("declared_monthly_income", 0) or 0)
    household = int(app_row.get("household_size", 1) or 1)
    per_capita = declared / max(1, household)

    today = dt.date.today()
    inflow_signals: List[Dict[str, Any]] = []
    for p in doc_paths:
        df = _read_df(p)
        if df is None or df.empty:
            continue
        win = summarize_inflows_by_window(df, today)
        if win.get("has_dates"):
            inflow_signals.append({"path": p, **win})

    def monthly_estimate(signals: List[Dict[str, Any]]) -> float:
        vals = []
        for s in signals:
            if "last_30d_inflow" in s and isinstance(s["last_30d_inflow"], (int, float)):
                vals.append(float(s["last_30d_inflow"]))
            elif "last_60d_inflow" in s and isinstance(s["last_60d_inflow"], (int, float)):
                vals.append(float(s["last_60d_inflow"]) / 2.0)
            elif "last_90d_inflow" in s and isinstance(s["last_90d_inflow"], (int, float)):
                vals.append(float(s["last_90d_inflow"]) / 3.0)
        return float(sum(vals) / len(vals)) if vals else 0.0

    monthly_inferred = monthly_estimate(inflow_signals)

    ok_low  = declared * (1 - tolerance)
    ok_high = declared * (1 + tolerance)
    income_consistency_ok = ok_low <= monthly_inferred <= ok_high if declared > 0 else (monthly_inferred == 0)

    flags = []
    if not inflow_signals:
        flags.append("no_time_window_inflows_detected")
    if declared == 0 and monthly_inferred > 0:
        flags.append("declared_zero_but_inflows_present")
    if declared > 0 and monthly_inferred == 0:
        flags.append("no_inflows_detected_in_windows")
    if declared > 0 and not income_consistency_ok:
        flags.append("income_inconsistency")

    return {
        "declared_monthly_income": declared,
        "household_size": household,
        "per_capita_income": per_capita,
        "monthly_inferred_income": round(monthly_inferred, 2),
        "tolerance": tolerance,
        "income_consistency_ok": income_consistency_ok,
        "flags": flags,
        "sources": inflow_signals,
    }
