# app.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Business HR App", page_icon="💼", layout="wide")
st.title("💼 Business HR — Prediction & Clustering")


# Config

CLUSTER_MEANINGS = {
    0: "High performance, low salary",
    1: "Balanced, risk of stagnation",
    2: "High earners, low performance",
}

CLUSTER_MESSAGES_SUFFIX = {
    0: "the High Performers. Watch out if they feel unfair of salary base.",
    1: "the Work Life Balance! Watch for risk of stagnation & lack of motivation.",
    2: "our high earners but not too effective.",
}

CLUSTER_RECOMMENDATIONS_BASE = {
    0: "Offer financial security, recognition & growth.",
    1: "Offer flexible schedule, team building and development.",
    2: "Prioritize mental health, focus on their work-life balance, and intensive based on performance focus.",
}

FEATURES = [
    "age",
    "marital_status",
    "years_experience",
    "education_level",
    "department",
    "bonus_percentage",
    "overtime_hours",
    "monthly_income",
    "income_class",       
    "performance_score",  
]
CATEGORICAL = ["marital_status", "education_level", "department", "income_class"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

MARITAL_OPTS = ["Single", "Married", "Divorced", "Widowed"]
EDU_OPTS = ["High School", "Diploma", "Bachelor", "Master", "Doctorate"]
DEPT_OPTS = ["Sales", "Tech", "Finance", "HR", "Operations", "Marketing"]


PERF_HIGH_CUT = 75.0

RISK_FROM_CLUSTER = {0: "Low", 1: "Low", 2: "High"}


QUADRANT_ORDER = [
    "Low Performance, Low Risk",   
    "Low Performance, High Risk",  
    "High Performance, Low Risk",  
    "High Performance, High Risk", 
]
def quadrant_id(name: str) -> int:
    try:
        return QUADRANT_ORDER.index(name) + 1
    except ValueError:
        return 0  # fallback


MEANING_BY_CQ = {
    (0, 3): "These are your role models. Strong, motivated, and stable.",
    (0, 4): "Your “flight risks.” Strong performers but tempted by outside opportunities or burnout.",
    (1, 1): "Solid but stagnant. Reliable, but not pushing boundaries.",
    (1, 2): "Underperformers who may leave — and their exit won’t hurt productivity much.",
    (2, 1): "Comfortable but unproductive; can drain resources.",
    (2, 2): "High disengagement + poor output = red flag.",
    (2, 3): "Sometimes highly paid but finally delivering consistently.",
    (2, 4): "Costly but capable employees who could leave. Losing them could hit both performance and budgets.",
}
RECO_BY_CQ = {
    (0, 3): "Recognize achievements, provide growth paths, and assign them as mentors.",
    (0, 4): "Prioritize retention — offer tailored incentives, career development, flexible work, and personal check-ins.",
    (1, 1): "Provide structured training, rotation opportunities, and small stretch goals to unlock hidden potential.",
    (1, 2): "Address with coaching and performance plans. If no improvement, consider role realignment or managed exit.",
    (2, 1): "Tighten accountability, provide measurable KPIs, and evaluate long-term fit.",
    (2, 2): "Direct coaching conversation. If issues persist, prepare for exit. Use as a learning point to refine hiring/placement.",
    (2, 3): "Ensure ROI is clear. Balance rewards with accountability to sustain value creation.",
    (2, 4): "Reassess compensation alignment, provide clear career paths, and strengthen purpose/culture connection.",
}


# Helpers

def derive_income_class(income: float) -> str:
    if income <= 6_000_000:
        return "Low"
    elif income <= 12_000_000:
        return "Mid"
    return "High"

def default_demo_dataset(n=400, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 60, size=n),
            "marital_status": rng.choice(MARITAL_OPTS, size=n, p=[0.4, 0.45, 0.1, 0.05]),
            "years_experience": rng.integers(0, 35, size=n),
            "education_level": rng.choice(EDU_OPTS, size=n, p=[0.15, 0.15, 0.45, 0.2, 0.05]),
            "department": rng.choice(DEPT_OPTS, size=n),
            "bonus_percentage": rng.normal(8, 3, size=n).clip(0, 30).round(1),
            "overtime_hours": rng.normal(6, 3, size=n).clip(0, 40).round(1),
            "monthly_income": rng.normal(8_000_000, 2_500_000, size=n)
                               .clip(3_000_000, 25_000_000).round(-3),
            "performance_score": rng.normal(75, 12, size=n).clip(0, 100).round(1),
        }
    )
    df["income_class"] = df["monthly_income"].apply(derive_income_class)
    return df

def cluster_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )
    km = KMeans(n_clusters=3, n_init=20, random_state=42)
    pipe = Pipeline([("prep", pre), ("kmeans", km)])
    return pipe

@st.cache_resource(show_spinner=False)
def get_reference_pipe():
    df_ref = default_demo_dataset()
    pipe = cluster_pipeline()
    pipe.fit(df_ref[FEATURES])
    return pipe

def project_pca(X_prepped, n_components=2):
    p = PCA(n_components=n_components, random_state=42)
    coords = p.fit_transform(X_prepped)
    return coords, p.explained_variance_ratio_

def predict_cluster_single(row_df: pd.DataFrame, pipe: Pipeline) -> int:
    raw = pipe.predict(row_df[FEATURES])
    return int(raw[0])

def assign_quadrant(perf: float, risk_bucket: str) -> str:
    perf_band = "High" if perf >= PERF_HIGH_CUT else "Low"
    risk_band = "High" if risk_bucket == "High" else "Low"

    if perf_band == "Low" and risk_band == "Low":
        return "Low Performance, Low Risk"
    if perf_band == "Low" and risk_band == "High":
        return "Low Performance, High Risk"
    if perf_band == "High" and risk_band == "Low":
        return "High Performance, Low Risk"
    if perf_band == "High" and risk_band == "High":
        return "High Performance, High Risk"
    return "Uncategorized"

# Tabs

PERF_HIGH_CUT = 75.0
tab_pred, tab_cluster = st.tabs(["🔮 Prediction", "🧩 Clustering"])

# 🔮 Prediction

with tab_pred:
    st.subheader("Model Inference (Clustering: k=3)")

    st.sidebar.header("Employee Snapshot")
    age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=28, step=1)
    marital_status = st.sidebar.selectbox("Marital Status", MARITAL_OPTS, index=0)
    years_experience = st.sidebar.number_input("Years of Experience", min_value=0, max_value=45, value=3, step=1)
    education_level = st.sidebar.selectbox("Education Level", EDU_OPTS, index=2)
    department = st.sidebar.selectbox("Department", DEPT_OPTS, index=0)
    bonus_percentage = st.sidebar.slider("Bonus Percentage (%)", 0.0, 50.0, 8.0, 0.5)
    overtime_hours = st.sidebar.slider("Overtime Hours / month", 0.0, 80.0, 6.0, 0.5)
    monthly_income = st.sidebar.number_input(
        "Monthly Income (IDR)", min_value=1_000_000, max_value=100_000_000,
        value=8_000_000, step=100_000
    )
    performance_score = st.sidebar.slider("Performance Score", 0.0, 100.0, 75.0, 0.5)

    income_class = derive_income_class(monthly_income)

    row = pd.DataFrame([{
        "age": age,
        "marital_status": marital_status,
        "years_experience": years_experience,
        "education_level": education_level,
        "department": department,
        "bonus_percentage": bonus_percentage,
        "overtime_hours": overtime_hours,
        "monthly_income": monthly_income,
        "income_class": income_class,
        "performance_score": performance_score,
    }])

    st.write("#### Preview of Input")
    st.dataframe(row, use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        try:
            ref_pipe = get_reference_pipe()
            c = predict_cluster_single(row, ref_pipe)
            st.success(
                f"This employee belong to **Cluster {c}**.\n"
                f"Cluster {c}, {CLUSTER_MESSAGES_SUFFIX[c]}\n\n"
                f"**Income Class:** {income_class}\n"
                f"**Recommendation:** {CLUSTER_RECOMMENDATIONS_BASE[c]}"
            )
        except Exception as e:
            st.exception(e)


# 🧩 Clustering

with tab_cluster:
    st.subheader("Employee Segmentation (K-Means, k=3)")

    data_src = st.radio("Choose data source", ["Upload CSV", "Use Demo Data"], horizontal=True)

    if data_src == "Upload CSV":
        upl = st.file_uploader("Upload CSV (must include columns above)", type=["csv"], key="cluster_csv")
        df_clu = None
        if upl is not None:
            try:
                df_clu = pd.read_csv(upl)
                if "income_class" not in df_clu.columns and "monthly_income" in df_clu.columns:
                    df_clu["income_class"] = df_clu["monthly_income"].apply(derive_income_class)
                if "performance_score" not in df_clu.columns:
                    df_clu["performance_score"] = 70.0
            except Exception:
                st.error("Could not read the uploaded file.")
    else:
        df_clu = default_demo_dataset()

    if df_clu is not None:
        if "employee_id" not in df_clu.columns:
            df_clu.insert(0, "employee_id", np.arange(1, len(df_clu) + 1))

        df_use = df_clu[FEATURES].copy()
        pipe = cluster_pipeline()
        X_prepped = pipe.named_steps["prep"].fit_transform(df_use)
        labels = pipe.named_steps["kmeans"].fit_predict(X_prepped)

        df_plot = df_use.copy()
        df_plot["employee_id"] = df_clu["employee_id"].values
        df_plot["cluster"] = labels.astype(int)
        df_plot["cluster_name"] = df_plot["cluster"].map(CLUSTER_MEANINGS)
        df_plot["risk_bucket"] = df_plot["cluster"].map(RISK_FROM_CLUSTER)

        # Quadrants
        df_plot["quadrant"] = df_plot.apply(
            lambda r: assign_quadrant(r["performance_score"], r["risk_bucket"]), axis=1
        )

        # ===== Overview =====
        st.markdown("### Overview")
        total_emp = len(df_plot)
        avg_perf = float(np.nanmean(df_plot["performance_score"])) if total_emp else 0.0
        high_risk_count = int((df_plot["quadrant"] == "High Performance, High Risk").sum())  # Quadrant 4
        low_risk_count = int((df_plot["quadrant"] == "High Performance, Low Risk").sum())   # Quadrant 3

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Employees", f"{total_emp:,}")
        cB.metric("Avg Performance", f"{avg_perf:.1f}")
        cC.metric("High Risk to Resign", f"{high_risk_count:,}")
        cD.metric("Low Risk to Resign", f"{low_risk_count:,}")

        # Risk bucket distribution
        risk_counts = (
            df_plot["risk_bucket"].value_counts(dropna=False)
            .rename_axis("risk_bucket")
            .reset_index(name="count")
        )
        risk_counts["percent"] = (risk_counts["count"] / total_emp * 100).round(1)
        fig_risk = px.bar(
            risk_counts, x="risk_bucket", y="count",
            color="risk_bucket", text="percent",
            title="Risk Bucket Distribution"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Quadrant summary
        st.markdown("### Quadrant Summary")
        quad_counts = (
            df_plot["quadrant"].value_counts(dropna=False)
            .rename_axis("quadrant")
            .reset_index(name="count")
        )
        quad_counts["percent"] = (quad_counts["count"] / total_emp * 100).round(1)

        q1, q2 = st.columns([2, 3])
        with q1:
            st.dataframe(quad_counts.sort_values("quadrant"), use_container_width=True)
        with q2:
            fig_quad = px.pie(
                quad_counts, names="quadrant", values="count",
                title="Quadrant Share"
            )
            st.plotly_chart(fig_quad, use_container_width=True)

        # PCA visualization
        coords, expl = project_pca(X_prepped, n_components=2)
        df_plot["PC1"] = coords[:, 0]
        df_plot["PC2"] = coords[:, 1]
        fig = px.scatter(
            df_plot,
            x="PC1", y="PC2",
            color="cluster",
            hover_data=["employee_id"] + FEATURES + ["cluster_name", "risk_bucket", "quadrant"],
            title=f"PCA Projection (k=3) — PC1 {expl[0]:.1%}, PC2 {expl[1]:.1%}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar Analysis
        st.markdown("### Radar Analysis (Numeric Features)")
        prof = df_plot.groupby("cluster")[NUMERIC].mean()
        prof_norm = (prof - prof.min()) / (prof.max() - prof.min()).replace(0, 1)

        radar = go.Figure()
        categories = prof_norm.columns.tolist()
        for cluster_id in prof_norm.index:
            radar.add_trace(
                go.Scatterpolar(
                    r=prof_norm.loc[cluster_id].tolist(),
                    theta=categories,
                    fill="toself",
                    name=f"Cluster {cluster_id} — {CLUSTER_MEANINGS[cluster_id]}"
                )
            )
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(radar, use_container_width=True)

        # ===== Employee Drilldown =====
        st.markdown("### Employee Drilldown")
        ids_sorted = df_plot["employee_id"].tolist()
        sel_id = st.selectbox("Choose Employee ID", ids_sorted)
        row = df_plot.loc[df_plot["employee_id"] == sel_id].iloc[0]

        qid = quadrant_id(row["quadrant"])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Cluster", int(row["cluster"]))
        d2.metric("Performance", f"{row['performance_score']:.1f}")
        d3.metric("Risk Bucket", row["risk_bucket"])
        d4.metric("Quadrant", f"{qid}")

        # Meaning & Recommendation based on Cluster + Quadrant
        key = (int(row["cluster"]), qid)
        meaning_text = MEANING_BY_CQ.get(key)
        reco_text = RECO_BY_CQ.get(key)

        st.caption(f"Quadrant {qid}: {row['quadrant']}")
        st.caption(f"Meaning: {CLUSTER_MEANINGS[int(row['cluster'])]}")

        if meaning_text:
            st.write(f"**Meaning:** {meaning_text}")
        if reco_text:
            st.write(f"**Recommendations:** {reco_text}")

        with st.expander("Row details"):
            st.dataframe(row.to_frame().rename(columns={row.name: "value"}))

        st.markdown("### Clustered Data Preview")
        st.dataframe(df_plot.head(30), use_container_width=True)
