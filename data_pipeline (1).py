"""
CACEIS Human Capital Intelligence Cockpit - data pipeline.

This module powers the Streamlit interface. It is intentionally designed as a
prototype pipeline: robust enough for a project demo, transparent enough to be
explained during a defense, and conservative enough to avoid individual ranking.

All employee identifiers are used only as anonymous technical keys and are never
returned in user-facing tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_YEARS = [2023, 2024, 2025]
FINANCIAL_UNIT_LABEL = "financial units"

PERFORMANCE_WEIGHT = 0.25
POTENTIAL_WEIGHT = 0.25
ENGAGEMENT_WEIGHT = 0.20
FINANCIAL_WEIGHT = 0.15
CRITICALITY_WEIGHT = 0.15
RISK_PENALTY_WEIGHT = 0.20

PROTECTED_ABSENCE_GROUPS = {
    "maternite et paternite",
    "maternité et paternité",
    "legal / conventionnel familial",
    "légal / conventionnel familial",
}

NEUTRAL_ABSENCE_GROUPS = {
    "conges",
    "congés",
    "absences non suivi",
    "recuperation",
    "récupération",
}

RISK_RELEVANT_ABSENCE_KEYWORDS = [
    "maladie",
    "accident",
    "absence non autorisee",
    "absence non autorisée",
]

SEGMENT_LABELS = [
    "Strategic Value Contributors",
    "Stable Core Contributors",
    "Developing Talent Pools",
    "Critical Capability Holders",
    "Organizational Vigilance Areas",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def normalize_text(value) -> str:
    """Return a normalized text used for robust matching."""
    if pd.isna(value):
        return ""
    value = str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("\n", " ").replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\n", " ").replace("\xa0", " ")).strip() for c in df.columns]
    return df


def find_file(data_dir: Path, *keywords: str) -> Optional[Path]:
    """Find a file in data_dir using keywords. Returns shortest matching name."""
    files = [p for p in data_dir.glob("*") if p.is_file()]
    matches = []
    for path in files:
        n = normalize_text(path.name)
        if all(normalize_text(k) in n for k in keywords):
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda p: len(p.name))[0]


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    normalized = {normalize_text(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_text(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = normalize_text(candidate)
        for norm, original in normalized.items():
            if key and key in norm:
                return original
    if required:
        raise KeyError(f"Missing required column among: {candidates}")
    return None


def standardize_employee_id(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "XX": pd.NA, "xx": pd.NA})
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def extract_year(series: pd.Series) -> pd.Series:
    """Robust year extraction for numeric years, Excel serials, strings and dates."""
    s = pd.Series(series).copy()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    # 1) Numeric values that are already years.
    num = pd.to_numeric(s, errors="coerce")
    mask_year = num.between(1900, 2100)
    out.loc[mask_year] = num.loc[mask_year].round().astype("Int64")

    # 2) Excel serial dates.
    mask_serial = num.between(25000, 60000) & out.isna()
    if mask_serial.any():
        dt_serial = pd.to_datetime(num.loc[mask_serial], unit="D", origin="1899-12-30", errors="coerce")
        valid = dt_serial.dt.year.between(2000, 2100)
        out.loc[dt_serial.index[valid]] = dt_serial.loc[valid].dt.year.astype("Int64")

    # 3) Regex extraction from string.
    text = s.astype("string")
    regex_year = text.str.extract(r"(20\d{2})")[0]
    mask_regex = out.isna() & regex_year.notna()
    out.loc[mask_regex] = regex_year.loc[mask_regex].astype("Int64")

    # 4) Datetime parser fallback, avoiding invalid 1970 conversions.
    mask_remaining = out.isna()
    if mask_remaining.any():
        dt = pd.to_datetime(text.loc[mask_remaining], errors="coerce", dayfirst=True)
        valid = dt.dt.year.between(2000, 2100)
        out.loc[dt.index[valid]] = dt.loc[valid].dt.year.astype("Int64")

    return out


def minmax_score(series: pd.Series, higher_is_better: bool = True, clip_q: Optional[float] = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if clip_q is not None and x.notna().sum() > 10:
        lo = x.quantile(1 - clip_q) if not higher_is_better else x.quantile(0)
        hi = x.quantile(clip_q)
        x = x.clip(lower=lo, upper=hi)
    mn, mx = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.where(x.notna(), 0.5, np.nan), index=series.index)
    score = (x - mn) / (mx - mn)
    if not higher_is_better:
        score = 1 - score
    return score.clip(0, 1)


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where(b.abs() > 0, a / b, np.nan)


def role_family(title) -> str:
    t = normalize_text(title)
    if any(k in t for k in ["it", "information", "system", "data", "digital", "developer", "technology", "cyber"]):
        return "Technology & Data"
    if any(k in t for k in ["risk", "control", "compliance", "audit", "inspection"]):
        return "Risk, Control & Compliance"
    if any(k in t for k in ["fund", "account", "nav", "custody", "cash", "middle", "back office", "operations"]):
        return "Operations & Fund Services"
    if any(k in t for k in ["client", "sales", "coverage", "business development", "relationship"]):
        return "Client & Business Development"
    if any(k in t for k in ["legal", "jurid"]):
        return "Legal"
    if any(k in t for k in ["human resources", "hr", "talent"]):
        return "Human Resources"
    if not t:
        return "Unknown"
    return "Other / Support Functions"


def training_category(course_name, organization="") -> str:
    t = normalize_text(str(course_name) + " " + str(organization))
    if any(k in t for k in ["compliance", "reglement", "réglement", "fatca", "aifmd", "mifid", "aml", "kyc", "bale", "bâle", "sanction"]):
        return "Regulatory & Compliance"
    if any(k in t for k in ["data", "python", "ai", "ia", "digital", "cloud", "aws", "cyber", "finops", "power bi", "sql"]):
        return "Technology, Data & AI"
    if any(k in t for k in ["manager", "leadership", "management", "team", "feedback", "coaching"]):
        return "Management & Leadership"
    if any(k in t for k in ["fund", "custody", "finance", "accounting", "cash", "nav", "private equity", "middle office"]):
        return "Business Expertise"
    if any(k in t for k in ["english", "langue", "language", "communication", "soft", "presentation"]):
        return "Transversal Skills"
    return "Other Training"


# -----------------------------------------------------------------------------
# Finance layer
# -----------------------------------------------------------------------------

def parse_finance_pl(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Synthese_PL", header=None, engine="openpyxl")
    years = [2022, 2023, 2024, 2025]
    records = []
    for scope, metric_col, value_cols in [("Consolidated", 0, [1, 2, 3, 4]), ("Europe", 6, [7, 8, 9, 10])]:
        for _, row in df.iterrows():
            metric = row.get(metric_col)
            if pd.isna(metric):
                continue
            for year, col in zip(years, value_cols):
                records.append({"scope": scope, "year": year, "metric": str(metric).strip(), "value": pd.to_numeric(row.get(col), errors="coerce")})
    return pd.DataFrame(records)


def parse_fte(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Synthese_ETP", header=None, engine="openpyxl")
    records = []
    row_map = {"Consolidated": 2, "Europe": 6}
    year_cols = {2022: (2, 3), 2023: (4, 5), 2024: (6, 7), 2025: (8, 9)}
    for scope, row_idx in row_map.items():
        for year, (end_col, avg_col) in year_cols.items():
            records.append({
                "scope": scope,
                "year": year,
                "fte_end": pd.to_numeric(df.iloc[row_idx, end_col], errors="coerce"),
                "fte_avg": pd.to_numeric(df.iloc[row_idx, avg_col], errors="coerce"),
            })
    return pd.DataFrame(records)


def load_finance(data_dir: Path) -> pd.DataFrame:
    file_finance = find_file(data_dir, "PL-FTE")
    if file_finance is None:
        return pd.DataFrame()
    pl_long = parse_finance_pl(file_finance)
    fte = parse_fte(file_finance)
    pl_wide = pl_long.pivot_table(index=["scope", "year"], columns="metric", values="value", aggfunc="first").reset_index()
    finance = pl_wide.merge(fte, on=["scope", "year"], how="left")
    pnb = pick_col(finance, ["Net Banking Income (PNB)"], required=True)
    personnel = pick_col(finance, ["Total Personnel Costs"], required=True)
    rbe = pick_col(finance, ["Gross Operating Income (RBE)"], required=False)
    training_cost = pick_col(finance, ["Formation (training costs)"], required=False)
    finance["human_capital_roi"] = finance[pnb] / finance[personnel].abs()
    finance["value_per_fte"] = finance[pnb] / finance["fte_avg"]
    finance["personnel_cost_per_fte"] = finance[personnel].abs() / finance["fte_avg"]
    finance["training_cost_per_fte"] = finance[training_cost].abs() / finance["fte_avg"] if training_cost else np.nan
    finance["gross_operating_income_per_fte"] = finance[rbe] / finance["fte_avg"] if rbe else np.nan
    finance = finance.rename(columns={pnb: "net_banking_income", personnel: "total_personnel_costs"})
    return finance


# -----------------------------------------------------------------------------
# HR, performance, training and absenteeism layers
# -----------------------------------------------------------------------------

def load_employee_master(data_dir: Path, fast_mode: bool = True) -> pd.DataFrame:
    file_data = find_file(data_dir, "Data.xlsx") or find_file(data_dir, "Data")
    if file_data is None:
        return pd.DataFrame()
    nrows = 80000 if fast_mode else None
    df = pd.read_excel(file_data, sheet_name="Sheet1", engine="openpyxl", nrows=nrows)
    df = clean_columns(df)
    id_col = pick_col(df, ["ID Employee"], required=True)
    country_col = pick_col(df, ["COUNTRY_GROUP_LABEL_EN", "Country"], required=False)
    period_col = pick_col(df, ["PERIOD"], required=False)
    age_col = pick_col(df, ["Age range"], required=False)
    gender_col = pick_col(df, ["SEXE_GROUP_LABEL_EN", "Gender"], required=False)
    contract_col = pick_col(df, ["CONTRACT_GROUP_LABEL_EN", "Contract"], required=False)
    degree_col = pick_col(df, ["DEGREE_LEVEL_GROUP_LABEL_EN", "Degree"], required=False)
    entry_col = pick_col(df, ["DATE_ENTRY_CACEIS", "DATE_ENTRY_GROUP"], required=False)
    job_col = pick_col(df, ["POSTE_LABEL_LOCAL", "Job"], required=False)
    entity_col = pick_col(df, ["ENTITY_LABEL_LOCAL", "Entity"], required=False)

    employee = pd.DataFrame({
        "employee_id": standardize_employee_id(df[id_col]),
        "country": df[country_col] if country_col else pd.NA,
        "period": pd.to_datetime(df[period_col], errors="coerce") if period_col else pd.NaT,
        "age_range": df[age_col] if age_col else pd.NA,
        "gender": df[gender_col] if gender_col else pd.NA,
        "contract_type": df[contract_col] if contract_col else pd.NA,
        "degree_level": df[degree_col] if degree_col else pd.NA,
        "entry_date": pd.to_datetime(df[entry_col], errors="coerce") if entry_col else pd.NaT,
        "job_title": df[job_col] if job_col else pd.NA,
        "entity": df[entity_col] if entity_col else pd.NA,
    }).dropna(subset=["employee_id"])

    employee["year"] = employee["period"].dt.year.astype("Int64")
    employee["country_scope"] = np.select(
        [employee["country"].astype("string").str.contains("France", case=False, na=False),
         employee["country"].astype("string").str.contains("Luxembourg", case=False, na=False)],
        ["France", "Luxembourg"],
        default="Other"
    )
    employee = employee[employee["year"].isin(DEFAULT_YEARS)].copy()
    employee = employee.sort_values(["employee_id", "year", "period"])
    employee_year = employee.drop_duplicates(["employee_id", "year"], keep="last").copy()
    employee_year["role_family"] = employee_year["job_title"].map(role_family)
    employee_year["tenure_years"] = (pd.to_datetime(employee_year["year"].astype(str) + "-12-31") - employee_year["entry_date"]).dt.days / 365.25
    employee_year["tenure_years"] = employee_year["tenure_years"].clip(lower=0)
    return employee_year


def load_performance_2023(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Feuil1", engine="openpyxl")
    df = clean_columns(df)
    df["Pays"] = df["Pays"].ffill()
    out = pd.DataFrame({
        "employee_id": standardize_employee_id(df[pick_col(df, ["IUG"], required=True)]),
        "country": df[pick_col(df, ["Pays"], required=False)] if pick_col(df, ["Pays"], required=False) else pd.NA,
        "year": 2023,
        "performance_note": to_numeric(df[pick_col(df, ["Note"], required=True)]),
        "performance_source": path.name,
    })
    return out.dropna(subset=["employee_id"])


def load_performance_database(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Database", engine="openpyxl")
    df = clean_columns(df)
    id_col = pick_col(df, ["IUG"], required=True)
    year_col = pick_col(df, ["Année", "Annee"], required=False)
    note_col = pick_col(df, ["Note de performance", "Note"], required=False)
    country_col = pick_col(df, ["Country", "Pays"], required=False)
    service_col = pick_col(df, ["Code et libellé Service", "Service"], required=False)
    job_col = pick_col(df, ["Libellé emploi", "Libelle emploi"], required=False)
    if note_col is None:
        return pd.DataFrame(columns=["employee_id", "country", "year", "performance_note", "service", "job_title_perf", "performance_source"])
    note = df[note_col].astype("string").str.extract(r"(\d+(?:[\.,]\d+)?)")[0].str.replace(",", ".", regex=False)
    out = pd.DataFrame({
        "employee_id": standardize_employee_id(df[id_col]),
        "country": df[country_col] if country_col else pd.NA,
        "year": extract_year(df[year_col]) if year_col else pd.NA,
        "performance_note": to_numeric(note),
        "service": df[service_col] if service_col else pd.NA,
        "job_title_perf": df[job_col] if job_col else pd.NA,
        "performance_source": path.name,
    })
    return out.dropna(subset=["employee_id"])

def first_non_null(series):
        values = series.dropna()
        if len(values) == 0:
            return pd.NA
        return values.astype(str).iloc[0]

def load_performance(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = [
        find_file(data_dir, "Notes evaluation 2023"),
        find_file(data_dir, "Version Définitive"),
        find_file(data_dir, "Vretraitement"),
    ]
    frames = []
    if files[0] is not None:
        frames.append(load_performance_2023(files[0]))
    for f in files[1:]:
        if f is not None:
            frames.append(load_performance_database(f))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["country_scope"] = np.select(
        [raw["country"].astype("string").str.contains("FR|France", case=False, na=False),
         raw["country"].astype("string").str.contains("LU|Luxembourg", case=False, na=False)],
        ["France", "Luxembourg"],
        default=pd.NA
    )
    quality = raw.groupby("year", dropna=False).agg(
        performance_rows=("employee_id", "size"),
        employees_with_performance=("employee_id", "nunique"),
        avg_note=("performance_note", "mean"),
        missing_note=("performance_note", lambda s: s.isna().sum()),
    ).reset_index()
    perf = raw.dropna(subset=["year", "performance_note"]).copy()
    perf = perf[perf["year"].isin(DEFAULT_YEARS)]
    perf = perf.groupby(["employee_id", "year"], as_index=False).agg(
        performance_note=("performance_note", "mean"),
        country_scope_perf=("country_scope", first_non_null),
        service=("service", first_non_null),
        job_title_perf=("job_title_perf", first_non_null),
    )
    return perf, quality


def load_training(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file_training = find_file(data_dir, "Training_Records")
    if file_training is None:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_excel(file_training, sheet_name="Final_CSV", engine="openpyxl")
    df = clean_columns(df)
    id_col = pick_col(df, ["Employee Code"], required=True)
    year_col = pick_col(df, ["Year"], required=False)
    status_col = pick_col(df, ["Status"], required=False)
    hours_col = pick_col(df, ["Total_Training_Hours"], required=False)
    cert_col = pick_col(df, ["Certifications"], required=False)
    course_col = pick_col(df, ["Attended_Courses"], required=False)
    org_col = pick_col(df, ["Organization"], required=False)
    entity_col = pick_col(df, ["Entity"], required=False)
    direction_col = pick_col(df, ["Direction"], required=False)
    start_col = pick_col(df, ["Seesion_Start_Date", "Session_Start_Date"], required=False)
    training = pd.DataFrame({
        "employee_id": standardize_employee_id(df[id_col]),
        "year": extract_year(df[year_col]) if year_col else extract_year(df[start_col]),
        "status": df[status_col] if status_col else pd.NA,
        "training_hours": to_numeric(df[hours_col]) if hours_col else 0,
        "certification": df[cert_col] if cert_col else pd.NA,
        "course": df[course_col] if course_col else pd.NA,
        "organization": df[org_col] if org_col else pd.NA,
        "entity_training": df[entity_col] if entity_col else pd.NA,
        "direction_training": df[direction_col] if direction_col else pd.NA,
    })
    training["training_hours"] = training["training_hours"].fillna(0)
    training["is_completed"] = training["status"].astype("string").str.contains("réalis|realis|completed", case=False, na=False)
    training["certification_flag"] = training["certification"].astype("string").str.contains("yes|oui|true|1", case=False, na=False)
    training["training_category"] = [training_category(c, o) for c, o in zip(training["course"], training["organization"])]
    training = training.dropna(subset=["employee_id", "year"])
    training = training[training["year"].isin(DEFAULT_YEARS)]
    features = training.groupby(["employee_id", "year"], as_index=False).agg(
        training_hours=("training_hours", "sum"),
        completed_courses=("is_completed", "sum"),
        total_training_records=("status", "size"),
        certifications=("certification_flag", "sum"),
        strategic_training_records=("training_category", lambda s: s.isin(["Technology, Data & AI", "Regulatory & Compliance", "Management & Leadership", "Business Expertise"]).sum()),
        entity_training=("entity_training", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else pd.NA),
        direction_training=("direction_training", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else pd.NA),
    )
    features["completion_rate"] = safe_divide(features["completed_courses"], features["total_training_records"])
    taxonomy = training.groupby(["year", "training_category"], as_index=False).agg(records=("course", "size"), hours=("training_hours", "sum"))
    return features, taxonomy


def response_to_score(value) -> float:
    t = normalize_text(value)
    if not t:
        return np.nan
    if "tout a fait d'accord" in t or "oui, tout a fait" in t or t == "excellent":
        return 5
    if t == "d'accord" or "oui, en partie" in t or t in {"tres bien", "très bien"}:
        return 4
    if "ni d'accord" in t or "moyen" in t:
        return 3
    if "pas vraiment" in t or (("pas d'accord" in t) and ("pas du tout" not in t)):
        return 2
    if "pas du tout" in t or t in {"non", "no"}:
        return 1
    return np.nan


def load_review_file(path: Optional[Path], prefix: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Data", engine="openpyxl")
    df = clean_columns(df)
    id_col = pick_col(df, ["Matricule"], required=True)
    date_col = pick_col(df, ["Date"], required=False)
    meta = {normalize_text(c) for c in [
        "Date", "Matricule", "Formation", "Mode de formation", "Organisme", "Organization", "Formateur",
        "ID de session", "Session_ID", "Date de début de session", "Date de fin de session", "Lieu de session", "Status"
    ]}
    qcols = [c for c in df.columns if normalize_text(c) not in meta]
    if not qcols:
        scored = pd.DataFrame(index=df.index)
    else:
        scored = df[qcols].map(response_to_score)
    out = pd.DataFrame({
        "employee_id": standardize_employee_id(df[id_col]),
        "year": extract_year(df[date_col]) if date_col else pd.NA,
        f"{prefix}_review_score": scored.mean(axis=1),
        f"{prefix}_answered_questions": scored.notna().sum(axis=1),
    })
    return out.dropna(subset=["employee_id", "year"])


def load_reviews(data_dir: Path) -> pd.DataFrame:
    quick = load_review_file(find_file(data_dir, "Quick_Review"), "quick")
    cold = load_review_file(find_file(data_dir, "Cold_Review"), "cold")
    if quick.empty and cold.empty:
        return pd.DataFrame()
    out = quick.merge(cold, on=["employee_id", "year"], how="outer")
    out = out[out["year"].isin(DEFAULT_YEARS)]
    out = out.groupby(["employee_id", "year"], as_index=False).agg(
        quick_review_score=("quick_review_score", "mean"),
        cold_review_score=("cold_review_score", "mean"),
        quick_answered_questions=("quick_answered_questions", "sum"),
        cold_answered_questions=("cold_answered_questions", "sum"),
    )
    out["training_impact_index_raw"] = out[["quick_review_score", "cold_review_score"]].mean(axis=1)
    return out


def load_absence_detail(data_dir: Path, fast_mode: bool = True) -> pd.DataFrame:
    files = [
        (data_dir / "Absentéisme_-_détail_affectation_-_Bilan_social.xlsx", 2023, "Rapport 1", 1),
        (data_dir / "Absentéisme_-_détail_affectation_-_Bilan_social (1).xlsx", 2024, "Rapport 1", 1),
        (find_file(data_dir, "Absentéisme", "2025"), 2025, "extract", 0),
    ]
    frames = []
    wanted = {
        "Employee Code", "Société", "Niveau 6", "Niveau 7", "Regroupement Jour Absences",
        "Motif Jour Absence", "Jours Ouvrés Absence", "Jours Ouvrables Absence", "Jour Calendaires Absence"
    }
    nrows = 60000 if fast_mode else None
    for path, year, sheet, header in files:
        if path is None or not Path(path).exists():
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=header, engine="openpyxl", nrows=nrows, usecols=lambda c: str(c).strip() in wanted)
        except Exception:
            continue
        df = clean_columns(df)
        id_col = pick_col(df, ["Employee Code"], required=True)
        group_col = pick_col(df, ["Regroupement Jour Absences"], required=False)
        days_col = pick_col(df, ["Jours Ouvrés Absence", "Jours Ouvrables Absence", "Jour Calendaires Absence"], required=False)
        company_col = pick_col(df, ["Société"], required=False)
        lvl6_col = pick_col(df, ["Niveau 6"], required=False)
        lvl7_col = pick_col(df, ["Niveau 7"], required=False)
        out = pd.DataFrame({
            "employee_id": standardize_employee_id(df[id_col]),
            "year": year,
            "absence_group": df[group_col] if group_col else pd.NA,
            "absence_days": to_numeric(df[days_col]) if days_col else 0,
            "company": df[company_col] if company_col else pd.NA,
            "org_level_6": df[lvl6_col] if lvl6_col else pd.NA,
            "org_level_7": df[lvl7_col] if lvl7_col else pd.NA,
        })
        out["absence_group_norm"] = out["absence_group"].map(normalize_text)
        out["is_protected_absence"] = out["absence_group_norm"].isin(PROTECTED_ABSENCE_GROUPS)
        out["is_neutral_absence"] = out["absence_group_norm"].isin(NEUTRAL_ABSENCE_GROUPS)
        out["is_risk_absence"] = (~out["is_protected_absence"]) & (~out["is_neutral_absence"])
        out["risk_absence_days"] = np.where(out["is_risk_absence"], out["absence_days"].fillna(0), 0)
        out["protected_absence_days"] = np.where(out["is_protected_absence"], out["absence_days"].fillna(0), 0)
        frames.append(out.dropna(subset=["employee_id"]))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    features = raw.groupby(["employee_id", "year"], as_index=False).agg(
        total_absence_days=("absence_days", "sum"),
        risk_absence_days=("risk_absence_days", "sum"),
        protected_absence_days=("protected_absence_days", "sum"),
        absence_records=("absence_days", "size"),
        main_absence_group=("absence_group", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().shape[0] else pd.NA),
        company=("company", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else pd.NA),
        org_level_6=("org_level_6", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else pd.NA),
        org_level_7=("org_level_7", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else pd.NA),
    )
    return features


# -----------------------------------------------------------------------------
# Scoring, segmentation, scenario and recommendations
# -----------------------------------------------------------------------------

def build_valuation_table(data_dir: Path, fast_mode: bool = True) -> Dict[str, pd.DataFrame]:
    finance = load_finance(data_dir)
    employee = load_employee_master(data_dir, fast_mode=fast_mode)
    performance, performance_quality = load_performance(data_dir)
    training, training_taxonomy = load_training(data_dir)
    reviews = load_reviews(data_dir)
    absence = load_absence_detail(data_dir, fast_mode=fast_mode)

    if employee.empty:
        raise ValueError("Employee master could not be loaded. Check Data.xlsx availability.")

    valuation = employee.copy()
    valuation = valuation[valuation["country_scope"].isin(["France", "Luxembourg"])].copy()
    valuation = valuation.merge(performance, on=["employee_id", "year"], how="left") if not performance.empty else valuation
    valuation = valuation.merge(training, on=["employee_id", "year"], how="left") if not training.empty else valuation
    valuation = valuation.merge(reviews, on=["employee_id", "year"], how="left") if not reviews.empty else valuation
    valuation = valuation.merge(absence, on=["employee_id", "year"], how="left") if not absence.empty else valuation

    for col in ["training_hours", "completed_courses", "total_training_records", "certifications", "strategic_training_records", "risk_absence_days", "protected_absence_days", "absence_records"]:
        if col in valuation.columns:
            valuation[col] = valuation[col].fillna(0)
        else:
            valuation[col] = 0
    if "completion_rate" not in valuation:
        valuation["completion_rate"] = 0
    valuation["completion_rate"] = valuation["completion_rate"].fillna(0)

    if "training_impact_index_raw" not in valuation:
        valuation["training_impact_index_raw"] = np.nan
    med_impact = valuation["training_impact_index_raw"].median()
    valuation["training_impact_index_raw"] = valuation["training_impact_index_raw"].fillna(med_impact if pd.notna(med_impact) else 3)

    # Harmonize country and job fields after merge.
    if "country_scope_perf" in valuation.columns:
        valuation["country_scope"] = valuation["country_scope"].combine_first(valuation["country_scope_perf"])
    if "job_title_perf" in valuation.columns:
        valuation["job_title_final"] = valuation["job_title"].combine_first(valuation["job_title_perf"])
    else:
        valuation["job_title_final"] = valuation["job_title"]
    valuation["role_family"] = valuation["job_title_final"].map(role_family)

    # Department view used by the Streamlit filters.
    # Fallback logic validated with the project team:
    # business group if available; service; entity; role family.
    if "business_group" not in valuation.columns:
        valuation["business_group"] = pd.NA
    if "org_level_6" in valuation.columns:
        valuation["business_group"] = valuation["business_group"].combine_first(valuation["org_level_6"])
    if "direction_training" in valuation.columns:
        valuation["business_group"] = valuation["business_group"].combine_first(valuation["direction_training"])

    valuation["department_view"] = (
        valuation["business_group"]
        .combine_first(valuation.get("service", pd.Series(pd.NA, index=valuation.index)))
        .combine_first(valuation.get("entity", pd.Series(pd.NA, index=valuation.index)))
        .combine_first(valuation["role_family"])
        .fillna("Unknown department")
        .astype(str)
    )

    # Performance score.
    perf_med = valuation["performance_note"].median() if "performance_note" in valuation else np.nan
    valuation["performance_note_imputed"] = valuation.get("performance_note", pd.Series(np.nan, index=valuation.index)).fillna(perf_med if pd.notna(perf_med) else 3)
    valuation["performance_index"] = minmax_score(valuation["performance_note_imputed"], higher_is_better=True).fillna(0.5)

    # Potential score.
    valuation["training_hours_index"] = minmax_score(np.log1p(valuation["training_hours"]), higher_is_better=True).fillna(0.0)
    valuation["training_impact_index"] = minmax_score(valuation["training_impact_index_raw"], higher_is_better=True).fillna(0.5)
    valuation["strategic_training_index"] = minmax_score(valuation["strategic_training_records"], higher_is_better=True).fillna(0.0)
    valuation["potential_index"] = (0.55 * valuation["training_hours_index"] + 0.25 * valuation["training_impact_index"] + 0.20 * valuation["strategic_training_index"]).clip(0, 1)

    # Engagement / risk score.
    valuation["engagement_proxy_index"] = minmax_score(valuation["risk_absence_days"], higher_is_better=False, clip_q=0.95).fillna(0.5)

    # Internal strategic criticality.
    role_freq = valuation["role_family"].value_counts(normalize=True)
    valuation["internal_rarity_index"] = valuation["role_family"].map(lambda r: 1 - role_freq.get(r, 0))
    strategic_roles = ["Technology & Data", "Risk, Control & Compliance", "Operations & Fund Services", "Legal"]
    valuation["strategic_role_index"] = valuation["role_family"].isin(strategic_roles).astype(float)
    valuation["tenure_index"] = minmax_score(valuation["tenure_years"], higher_is_better=True).fillna(0.5)
    valuation["certification_index"] = minmax_score(valuation["certifications"], higher_is_better=True).fillna(0.0)
    valuation["criticality_index"] = (
        0.35 * valuation["internal_rarity_index"] +
        0.25 * valuation["strategic_role_index"] +
        0.20 * valuation["certification_index"] +
        0.20 * valuation["tenure_index"]
    ).clip(0, 1)

    # Financial contribution is annual and Europe-based.
    fin_europe = finance[finance["scope"].eq("Europe")][["year", "value_per_fte", "human_capital_roi"]].copy() if not finance.empty else pd.DataFrame()
    if not fin_europe.empty:
        fin_europe["financial_contribution_index"] = minmax_score(fin_europe["value_per_fte"], higher_is_better=True)
        valuation = valuation.merge(fin_europe[["year", "financial_contribution_index", "value_per_fte", "human_capital_roi"]], on="year", how="left")
    else:
        valuation["financial_contribution_index"] = 0.5
        valuation["value_per_fte"] = np.nan
        valuation["human_capital_roi"] = np.nan
    valuation["financial_contribution_index"] = valuation["financial_contribution_index"].fillna(0.5)

    valuation["organizational_risk_index"] = (
        0.40 * (1 - valuation["engagement_proxy_index"]) +
        0.25 * (1 - valuation["potential_index"]) +
        0.20 * (1 - valuation["performance_index"]) +
        0.15 * valuation["criticality_index"] * (1 - valuation["potential_index"])
    ).clip(0, 1)

    valuation["strategic_value_score"] = 100 * (
        PERFORMANCE_WEIGHT * valuation["performance_index"] +
        POTENTIAL_WEIGHT * valuation["potential_index"] +
        ENGAGEMENT_WEIGHT * valuation["engagement_proxy_index"] +
        FINANCIAL_WEIGHT * valuation["financial_contribution_index"] +
        CRITICALITY_WEIGHT * valuation["criticality_index"]
    )
    valuation["risk_adjusted_hcv_score"] = (valuation["strategic_value_score"] * (1 - RISK_PENALTY_WEIGHT * valuation["organizational_risk_index"])).clip(0, 100)

    # Clustering on profile patterns. Individual IDs are not exported.
    cluster_features = ["performance_index", "potential_index", "engagement_proxy_index", "criticality_index", "organizational_risk_index", "financial_contribution_index"]
    cluster_base = valuation.dropna(subset=cluster_features).copy()
    if len(cluster_base) >= 30:
        X = cluster_base[cluster_features].fillna(cluster_base[cluster_features].median())
        Xs = StandardScaler().fit_transform(X)
        candidate_k = [3, 4, 5]
        scores = {}
        for k in candidate_k:
            if len(cluster_base) > k:
                labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xs)
                scores[k] = silhouette_score(Xs, labels)
        best_k = max(scores, key=scores.get) if scores else 4
        model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        cluster_base["cluster"] = model.fit_predict(Xs)
        valuation = valuation.merge(cluster_base[["employee_id", "year", "cluster"]], on=["employee_id", "year"], how="left")
    else:
        valuation["cluster"] = 0

    # Segment labels.
    segment_profile_tmp = valuation.groupby("cluster", dropna=False).agg(
        population=("employee_id", "nunique"),
        risk_adjusted_hcv_score=("risk_adjusted_hcv_score", "mean"),
        strategic_value_score=("strategic_value_score", "mean"),
        potential_index=("potential_index", "mean"),
        organizational_risk_index=("organizational_risk_index", "mean"),
        criticality_index=("criticality_index", "mean"),
    ).reset_index()
    label_map = assign_segment_labels(segment_profile_tmp)
    valuation["profile_segment"] = valuation["cluster"].map(label_map).fillna("Stable Core Contributors")

    segment_summary = build_segment_summary(valuation)
    portfolio = build_portfolio(segment_summary)
    recommendations = build_recommendations(segment_summary)
    country_kpis = build_country_kpis(valuation)
    department_summary = build_department_summary(valuation, min_population=20)
    data_quality = build_data_quality(employee, performance, training, reviews, absence, valuation, performance_quality)

    return {
        "finance": finance,
        "employee_year": employee,
        "performance_quality": performance_quality,
        "training_taxonomy": training_taxonomy,
        "valuation": valuation,
        "segment_summary": segment_summary,
        "portfolio": portfolio,
        "recommendations": recommendations,
        "country_kpis": country_kpis,
        "department_summary": department_summary,
        "data_quality": data_quality,
    }


def assign_segment_labels(profile: pd.DataFrame) -> Dict:
    labels = {}
    remaining = set(profile["cluster"])
    if profile.empty:
        return labels
    strategic = profile.sort_values(["risk_adjusted_hcv_score", "strategic_value_score"], ascending=False).iloc[0]["cluster"]
    labels[strategic] = "Strategic Value Contributors"
    remaining.discard(strategic)
    if remaining:
        critical = profile[profile["cluster"].isin(remaining)].sort_values(["criticality_index", "organizational_risk_index"], ascending=False).iloc[0]["cluster"]
        labels[critical] = "Critical Capability Holders"
        remaining.discard(critical)
    if remaining:
        vigilance = profile[profile["cluster"].isin(remaining)].sort_values(["organizational_risk_index"], ascending=False).iloc[0]["cluster"]
        labels[vigilance] = "Organizational Vigilance Areas"
        remaining.discard(vigilance)
    if remaining:
        develop = profile[profile["cluster"].isin(remaining)].sort_values(["potential_index"], ascending=False).iloc[0]["cluster"]
        labels[develop] = "Developing Talent Pools"
        remaining.discard(develop)
    for cluster in remaining:
        labels[cluster] = "Stable Core Contributors"
    return labels


def build_segment_summary(valuation: pd.DataFrame) -> pd.DataFrame:
    seg = valuation.groupby("profile_segment", as_index=False).agg(
        population=("employee_id", "nunique"),
        avg_hcv_score=("risk_adjusted_hcv_score", "mean"),
        strategic_value_score=("strategic_value_score", "mean"),
        performance_index=("performance_index", "mean"),
        potential_index=("potential_index", "mean"),
        engagement_proxy_index=("engagement_proxy_index", "mean"),
        criticality_index=("criticality_index", "mean"),
        organizational_risk_index=("organizational_risk_index", "mean"),
        avg_training_hours=("training_hours", "mean"),
        avg_risk_absence_days=("risk_absence_days", "mean"),
        france_share=("country_scope", lambda s: (s == "France").mean()),
        luxembourg_share=("country_scope", lambda s: (s == "Luxembourg").mean()),
    )
    total = seg["population"].sum()
    seg["population_share"] = seg["population"] / total if total else np.nan
    seg = seg.sort_values("avg_hcv_score", ascending=False)
    return seg


def build_country_kpis(valuation: pd.DataFrame) -> pd.DataFrame:
    out = valuation.groupby(["country_scope", "year"], as_index=False).agg(
        population=("employee_id", "nunique"),
        avg_hcv_score=("risk_adjusted_hcv_score", "mean"),
        avg_performance=("performance_note", "mean"),
        avg_training_hours=("training_hours", "mean"),
        avg_risk_absence_days=("risk_absence_days", "mean"),
        avg_criticality=("criticality_index", "mean"),
        avg_risk_index=("organizational_risk_index", "mean"),
    )
    total = out["population"].sum()
    out["population_share"] = out["population"] / total if total else np.nan
    return out


def build_department_summary(valuation: pd.DataFrame, min_population: int = 20) -> pd.DataFrame:
    if "department_view" not in valuation.columns:
        return pd.DataFrame()
    dept = valuation.groupby("department_view", as_index=False).agg(
        population=("employee_id", "nunique"),
        avg_hcv_score=("risk_adjusted_hcv_score", "mean"),
        strategic_value_score=("strategic_value_score", "mean"),
        organizational_risk_index=("organizational_risk_index", "mean"),
        performance_index=("performance_index", "mean"),
        potential_index=("potential_index", "mean"),
        engagement_proxy_index=("engagement_proxy_index", "mean"),
        criticality_index=("criticality_index", "mean"),
    )
    dept = dept[dept["population"] >= min_population].copy()
    total = dept["population"].sum()
    dept["population_share"] = dept["population"] / total if total else np.nan
    return dept.sort_values("avg_hcv_score", ascending=False)


def build_portfolio(segment_summary: pd.DataFrame) -> pd.DataFrame:
    p = segment_summary.copy()
    conditions = [
        (p["strategic_value_score"] >= p["strategic_value_score"].median()) & (p["organizational_risk_index"] <= p["organizational_risk_index"].median()),
        (p["strategic_value_score"] >= p["strategic_value_score"].median()) & (p["organizational_risk_index"] > p["organizational_risk_index"].median()),
        (p["potential_index"] >= p["potential_index"].median()) & (p["strategic_value_score"] < p["strategic_value_score"].median()),
        (p["organizational_risk_index"] > p["organizational_risk_index"].median()),
    ]
    choices = ["Productive & Resilient Human Capital", "Critical At-Risk Human Capital", "Growth Human Capital", "At-Risk Human Capital"]
    p["asset_class"] = np.select(conditions, choices, default="Defensive Core Human Capital")
    return p


def top_drivers(row: pd.Series) -> List[str]:
    metrics = {
        "performance": row.get("performance_index", np.nan),
        "potential": row.get("potential_index", np.nan),
        "engagement": row.get("engagement_proxy_index", np.nan),
        "criticality": row.get("criticality_index", np.nan),
    }
    return [k for k, _ in sorted(metrics.items(), key=lambda x: (np.nan_to_num(x[1]), x[0]), reverse=True)[:3]]


def top_risks(row: pd.Series) -> List[str]:
    risks = []
    if row.get("organizational_risk_index", 0) > 0.55:
        risks.append("elevated organizational risk")
    if row.get("potential_index", 1) < 0.45:
        risks.append("capability renewal gap")
    if row.get("engagement_proxy_index", 1) < 0.45:
        risks.append("availability / engagement pressure")
    if row.get("criticality_index", 0) > 0.60 and row.get("potential_index", 1) < 0.55:
        risks.append("critical skills dependency")
    if not risks:
        risks.append("no major structural risk detected")
    return risks[:3]


def recommendation_for(row: pd.Series) -> str:
    seg = row.get("profile_segment", "")
    risk = row.get("organizational_risk_index", 0)
    potential = row.get("potential_index", 0)
    criticality = row.get("criticality_index", 0)
    engagement = row.get("engagement_proxy_index", 0)
    performance = row.get("performance_index", 0)
    actions = []
    if criticality > 0.60 and risk > 0.45:
        actions.append("launch targeted retention and succession planning for critical capabilities")
    if potential < 0.45:
        actions.append("increase targeted upskilling and certification coverage")
    if engagement < 0.45:
        actions.append("review workload, well-being signals and managerial support")
    if performance >= 0.60 and potential < 0.55:
        actions.append("protect current contribution by renewing skills and career paths")
    if potential >= 0.60 and performance < 0.55:
        actions.append("convert learning investment into business impact through mentoring and mobility")
    if not actions:
        actions.append("maintain current investment and monitor value/risk trajectory")
    return "; ".join(actions[:3]).capitalize() + "."


def build_recommendations(segment_summary: pd.DataFrame) -> pd.DataFrame:
    if segment_summary.empty:
        return pd.DataFrame()
    rec = segment_summary.copy()
    rec["top_drivers"] = rec.apply(lambda r: ", ".join(top_drivers(r)), axis=1)
    rec["top_risks"] = rec.apply(lambda r: ", ".join(top_risks(r)), axis=1)
    rec["recommended_actions"] = rec.apply(recommendation_for, axis=1)
    rec["copilot_narrative"] = rec.apply(generate_segment_narrative, axis=1)
    if "population_share" not in rec.columns:
        total = rec["population"].sum()
        rec["population_share"] = rec["population"] / total if total else np.nan
    return rec[["profile_segment", "population_share", "avg_hcv_score", "top_drivers", "top_risks", "recommended_actions", "copilot_narrative"]]


def generate_segment_narrative(row: pd.Series) -> str:
    return (
        f"{row['profile_segment']} represents {row.get('population_share', 0):.1%} of the analysed population. "
        f"Its average Human Capital Value score is {row.get('avg_hcv_score', np.nan):.1f}/100. "
        f"The main value drivers are {', '.join(top_drivers(row))}. "
        f"The main watch-outs are {', '.join(top_risks(row))}. "
        f"Recommended action: {recommendation_for(row)}"
    )


def build_data_quality(employee, performance, training, reviews, absence, valuation, performance_quality) -> pd.DataFrame:
    rows = []
    sources = {
        "Employee master": employee,
        "Performance": performance,
        "Training": training,
        "Training reviews": reviews,
        "Absenteeism": absence,
        "Final valuation table": valuation,
    }
    for name, df in sources.items():
        if df is None or df.empty:
            rows.append({"source": name, "rows": 0, "anonymous_ids": 0, "years": "", "status": "Missing or empty"})
        else:
            rows.append({
                "source": name,
                "rows": len(df),
                "anonymous_ids": df["employee_id"].nunique() if "employee_id" in df.columns else np.nan,
                "years": ", ".join(map(str, sorted([int(y) for y in df["year"].dropna().unique()]))) if "year" in df.columns else "",
                "status": "Loaded",
            })
    q = pd.DataFrame(rows)
    return q


# -----------------------------------------------------------------------------
# Caching and optional CSV loading/export
# -----------------------------------------------------------------------------

def output_dir(data_dir: Path) -> Path:
    out = data_dir / "streamlit_outputs"
    out.mkdir(exist_ok=True)
    return out


def export_outputs(data: Dict[str, pd.DataFrame], data_dir: Path) -> None:
    out = output_dir(data_dir)
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty and name != "valuation":
            df.to_csv(out / f"{name}.csv", index=False)

    # Public analytical view for the interface filters.
    # It deliberately excludes anonymous employee identifiers.
    if "valuation" in data and isinstance(data["valuation"], pd.DataFrame) and not data["valuation"].empty:
        public_cols = [
            "year", "country_scope", "department_view", "profile_segment", "role_family",
            "risk_adjusted_hcv_score", "strategic_value_score", "organizational_risk_index",
            "performance_index", "potential_index", "engagement_proxy_index", "criticality_index",
            "training_hours", "risk_absence_days"
        ]
        public_cols = [c for c in public_cols if c in data["valuation"].columns]
        data["valuation"][public_cols].to_csv(out / "valuation_view.csv", index=False)


def load_from_cached_csv(data_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    out = output_dir(data_dir)
    expected = ["finance", "segment_summary", "portfolio", "recommendations", "country_kpis", "data_quality"]
    if not all((out / f"{x}.csv").exists() for x in expected):
        return None
    data = {name: pd.read_csv(out / f"{name}.csv") for name in expected}
    # Optional tables.
    for name in ["training_taxonomy", "performance_quality", "department_summary", "valuation_view"]:
        p = out / f"{name}.csv"
        data[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    # Keep a consistent key name for the app even when loaded from CSV cache.
    if "valuation_view" in data and not data["valuation_view"].empty:
        data["valuation"] = data["valuation_view"]
    return data


def load_pipeline(data_dir: Path, prefer_cache: bool = True, fast_mode: bool = True, rebuild_cache: bool = False) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    if prefer_cache and not rebuild_cache:
        cached = load_from_cached_csv(data_dir)
        if cached is not None:
            cached["_source"] = pd.DataFrame({"mode": ["cached_csv"]})
            return cached
    data = build_valuation_table(data_dir, fast_mode=fast_mode)
    export_outputs(data, data_dir)
    data["_source"] = pd.DataFrame({"mode": ["excel_pipeline_fast" if fast_mode else "excel_pipeline_full"]})
    return data


# -----------------------------------------------------------------------------
# Scenario engine
# -----------------------------------------------------------------------------

def run_scenario(segment_summary: pd.DataFrame, training_uplift_pct: float, risk_absence_reduction_pct: float, performance_uplift_points: float, strategic_investment: bool) -> pd.DataFrame:
    """Run sensitivity scenarios at segment level."""
    if segment_summary.empty:
        return pd.DataFrame()
    out = segment_summary.copy()
    base = out["avg_hcv_score"].copy()

    # Sensitivity assumptions. They are deliberately transparent and conservative.
    potential_delta = (training_uplift_pct / 100) * 12 * (1 - out["potential_index"])
    risk_delta = (risk_absence_reduction_pct / 100) * 10 * out["organizational_risk_index"]
    performance_delta = performance_uplift_points * 4.0 * (1 - out["performance_index"])
    critical_delta = np.where((strategic_investment) & (out["criticality_index"] > out["criticality_index"].median()), 3.0 * (1 - out["organizational_risk_index"]), 0)

    out["scenario_score_delta"] = potential_delta + risk_delta + performance_delta + critical_delta
    out["scenario_hcv_score"] = (base + out["scenario_score_delta"]).clip(0, 100)
    out["scenario_change_pct"] = (out["scenario_hcv_score"] / base - 1).replace([np.inf, -np.inf], np.nan) * 100
    if "population_share" not in out.columns:
        total = out["population"].sum()
        out["population_share"] = out["population"] / total if total else np.nan
    return out[["profile_segment", "population_share", "avg_hcv_score", "scenario_hcv_score", "scenario_score_delta", "scenario_change_pct"]]
