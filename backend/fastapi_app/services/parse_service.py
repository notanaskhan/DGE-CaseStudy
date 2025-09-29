
import os, re
from typing import Dict, Any, List, Optional

def _to_number(s: Any) -> float:
    if s is None:
        return 0.0
    if not isinstance(s, str):
        try:
            return float(s)
        except Exception:
            return 0.0
    x = s.strip()
    if not x:
        return 0.0
    neg = False
    if x.startswith("(") and x.endswith(")"):
        neg = True
        x = x[1:-1]
    x = re.sub(r"[^\d,.\-]", "", x)
    if "," in x and "." in x:
        if x.rfind(".") > x.rfind(","):
            x = x.replace(",", "")
        else:
            x = x.replace(".", "").replace(",", ".")
    else:
        x = x.replace(",", "")
    try:
        val = float(x)
    except Exception:
        val = 0.0
    return -val if neg and val >= 0 else val

def _find_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    norm = [c.strip().lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(norm):
            if c == cand_l or cand_l in c:
                return cols[i]
    return None

def _summarize_from_amount(df, amount_col: str) -> Dict[str, Any]:
    vals = df[amount_col].apply(_to_number)
    total_in = float(vals[vals > 0].sum())
    total_out = float((-vals[vals < 0]).sum())
    return {"ok": True, "rows": int(len(df)), "amount_column": amount_col, "total_inflow": total_in, "total_outflow": total_out}

def _summarize_from_credit_debit(df, credit_col: str, debit_col: str) -> Dict[str, Any]:
    cr = df[credit_col].apply(_to_number) if credit_col in df.columns else 0.0
    dr = df[debit_col].apply(_to_number) if debit_col in df.columns else 0.0
    total_in = float(getattr(cr, "sum", lambda: 0.0)())
    total_out = float(getattr(dr, "sum", lambda: 0.0)())
    return {"ok": True, "rows": int(len(df)), "amount_column": f"{credit_col}/{debit_col}", "total_inflow": total_in, "total_outflow": total_out}

def parse_csv(filepath: str) -> Dict[str, Any]:
    try:
        import pandas as pd
    except Exception as e:
        return {"ok": False, "warning": f"pandas not installed: {e}"}
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        return summarize_transactions(df, source=os.path.basename(filepath))
    except Exception as e:
        return {"ok": False, "warning": f"csv parse error: {e}"}

def parse_xlsx(filepath: str) -> Dict[str, Any]:
    try:
        import pandas as pd
    except Exception as e:
        return {"ok": False, "warning": f"pandas not installed: {e}"}
    try:
        df = pd.read_excel(filepath, dtype=str, engine="openpyxl")
        if df.empty:
            x = pd.ExcelFile(filepath, engine="openpyxl")
            frames = []
            for sh in x.sheet_names:
                d = pd.read_excel(filepath, sheet_name=sh, dtype=str, engine="openpyxl")
                if not d.empty:
                    frames.append(d)
            if frames:
                from pandas import concat
                df = concat(frames, ignore_index=True)
        return summarize_transactions(df, source=os.path.basename(filepath))
    except Exception as e:
        return {"ok": False, "warning": f"xlsx parse error: {e}"}

def parse_pdf_table(filepath: str) -> Dict[str, Any]:
    try:
        import pdfplumber, pandas as pd
    except Exception as e:
        return {"ok": False, "warning": f"pdfplumber/pandas not installed: {e}"}
    try:
        rows = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                for tbl in (page.extract_tables() or []):
                    rows.extend(tbl)
        if not rows:
            return {"ok": False, "warning": "no tables found in PDF"}
        header = rows[0]
        header_ok = header is not None and all(h is not None and str(h).strip() for h in header) and len(set(map(str, header))) == len(header)
        import pandas as pd
        df = (pd.DataFrame(rows[1:], columns=header) if header_ok else pd.DataFrame(rows))
        return summarize_transactions(df, source=os.path.basename(filepath))
    except Exception as e:
        return {"ok": False, "warning": f"pdf parse error: {e}"}

def summarize_transactions(df, source: str) -> Dict[str, Any]:
    import pandas as pd
    if df is None or df.empty:
        return {"ok": False, "warning": f"empty data in {source}"}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    amount_candidates = ["amount","transaction amount","amt","value","amount (aed)","aed amount","amount aed","total amount"]
    amount_col = _find_column(list(df.columns), amount_candidates)
    if amount_col:
        return _summarize_from_amount(df, amount_col)
    credit_candidates = ["credit","cr","deposit","inflow","paid in","credits"]
    debit_candidates  = ["debit","dr","withdrawal","outflow","paid out","debits"]
    credit_col = _find_column(list(df.columns), credit_candidates)
    debit_col  = _find_column(list(df.columns), debit_candidates)
    if credit_col or debit_col:
        if credit_col and debit_col:
            return _summarize_from_credit_debit(df, credit_col, debit_col)
        if credit_col and not debit_col:
            cr = df[credit_col].apply(_to_number)
            return {"ok": True, "rows": int(len(df)), "amount_column": credit_col, "total_inflow": float(cr.sum()), "total_outflow": 0.0}
        if debit_col and not credit_col:
            dr = df[debit_col].apply(_to_number)
            return {"ok": True, "rows": int(len(df)), "amount_column": debit_col, "total_inflow": 0.0, "total_outflow": float(dr.sum())}
    numeric_like = []
    for c in df.columns:
        try:
            vals = df[c].apply(_to_number)
            if (vals != 0).sum() >= max(1, int(0.5 * len(vals))):
                numeric_like.append((c, float(vals.abs().sum())))
        except Exception:
            pass
    if numeric_like:
        numeric_like.sort(key=lambda x: x[1], reverse=True)
        return _summarize_from_amount(df, numeric_like[0][0])
    return {"ok": False, "warning": f"no numeric/amount columns recognized in {source}"}

def parse_document(filepath: str) -> Dict[str, Any]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return parse_csv(filepath)
    if ext in [".xls", ".xlsx"]:
        return parse_xlsx(filepath)
    if ext == ".pdf":
        return parse_pdf_table(filepath)
    return {"ok": False, "warning": f"unsupported for parsing: {ext}"}
