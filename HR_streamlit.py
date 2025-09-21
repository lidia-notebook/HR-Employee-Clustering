# =========================
# Tabs: Dashboard & Simulator
# =========================
tab_dash, tab_sim = st.tabs(["📊 Dashboard", "🧪 Simulator"])

# -------------------------------------------------------
# TAB 1 — 📊 Dashboard (transform → overview → drilldown)
# -------------------------------------------------------
with tab_dash:
    # ---------- Transform & enrich ----------
    NUM_COLS = ["age","years_experience","weekly_hours","bonus_percentage",
                "performance_score","overtime_hours","monthly_income"]
    CAT_COLS = ["gender","marital_status","city","education_level","department","income_class"]

    # df_input must be set earlier in Section 4 (upload/sample)
    feature_cols = [c for c in NUM_COLS + CAT_COLS if c in df_input.columns]
    if not feature_cols:
        st.error("No usable feature columns found in the input. Check your column names.")
        st.stop()

    try:
        raw_clusters = PIPELINE_FULL.predict(df_input[feature_cols])
    except Exception as e:
        st.error(f"Prediction failed. Missing or mismatched columns: {e}")
        st.stop()

    # Map raw → business cluster ids (0,1,2) using label_map in MODEL_META
    LABEL_MAP = MODEL_META.get("clustering", {}).get("label_map", {})
    if LABEL_MAP:
        clusters = pd.Series(raw_clusters).map(LABEL_MAP).fillna(raw_clusters).astype(int).values
    else:
        clusters = raw_clusters

    df_enriched = df_input.copy()
    df_enriched["cluster"] = clusters

    # --- Risk & Quadrants (functions defined earlier in file) ---
    risk_df = compute_risk_scores(df_enriched)
    quad_df = assign_quadrants(df_enriched, risk_df)

    joined = (
        df_enriched
        .merge(risk_df, on=ID_COL, how="left")
        .merge(quad_df[[ID_COL, "performance_band", "quadrant"]], on=ID_COL, how="left")
    )

    # ---------- Overview ----------
    st.subheader("Overview")

    CUTS = MODEL_META["thresholds"]["values"]
    PERF_HI_VAL   = CUTS["perf_hi_val"]
    RISK_HIGH_CUT = MODEL_META["thresholds"]["risk_bucket_cuts"]["high_cut"]
    RISK_LOW_CUT  = MODEL_META["thresholds"]["risk_bucket_cuts"]["low_cut"]

    colA, colB, colC, colD = st.columns(4)
    with colA: st.metric("Employees", f"{len(joined):,}")
    with colB: st.metric("High Performance Cutoff", f"{PERF_HI_VAL:.2f}")
    with colC: st.metric("High Risk ≥", f"{RISK_HIGH_CUT:.0f}")
    with colD: st.metric("Low Risk ≤", f"{RISK_LOW_CUT:.0f}")

    # Risk distribution
    risk_counts = (
        joined["risk_bucket"]
        .value_counts(dropna=False)
        .rename_axis("risk_bucket")
        .reset_index(name="count")
    )
    risk_counts["percent"] = (risk_counts["count"] / len(joined) * 100).round(1)
    fig_risk = px.bar(
        risk_counts, x="risk_bucket", y="count",
        color="risk_bucket", text="percent",
        title="Risk Bucket Distribution"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Quadrant summary
    quad_counts = (
        joined["quadrant"]
        .value_counts(dropna=False)
        .rename_axis("quadrant")
        .reset_index(name="count")
    )
    quad_counts["percent"] = (quad_counts["count"] / len(joined) * 100).round(1)

    col1, col2 = st.columns([2,3])
    with col1:
        st.write("### Quadrant Summary")
        st.dataframe(quad_counts.sort_values("quadrant"), use_container_width=True)
    with col2:
        fig_quad = px.pie(
            quad_counts, names="quadrant", values="count",
            title="Quadrant Share"
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    # Cluster numeric profile (use saved artifact if available; else compute)
    st.write("### Cluster Numeric Profile")
    CLUSTER_PROFILE_NUMERIC = st.session_state.get("CLUSTER_PROFILE_NUMERIC", None)
    if CLUSTER_PROFILE_NUMERIC is None:
        try:
            CLUSTER_PROFILE_NUMERIC = joblib.load(ARTIFACTS_PATH / "cluster_profile_numeric.joblib")
            st.session_state["CLUSTER_PROFILE_NUMERIC"] = CLUSTER_PROFILE_NUMERIC
        except Exception:
            CLUSTER_PROFILE_NUMERIC = None

    if CLUSTER_PROFILE_NUMERIC is not None:
        st.dataframe(CLUSTER_PROFILE_NUMERIC, use_container_width=True)
    else:
        prof = (
            joined.groupby("cluster")[["performance_score","monthly_income","bonus_percentage","overtime_hours"]]
            .agg(["mean","median","count"])
            .round(2)
        )
        st.dataframe(prof, use_container_width=True)

    # ---------- Employee Drilldown ----------
    st.write("### Employee Drilldown")
    ids_sorted = joined[ID_COL].tolist()
    if not ids_sorted:
        st.info("No rows to display.")
        st.stop()

    selected_id = st.selectbox("Choose employee ID", ids_sorted)
    row = joined.loc[joined[ID_COL] == selected_id].iloc[0]

    CLUSTER_DESCRIPTIONS = MODEL_META.get("clustering", {}).get("descriptions", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cluster", int(row["cluster"]))
    c2.metric("Performance", f"{row['performance_score']:.1f}" if pd.notna(row["performance_score"]) else "—")
    c3.metric("Risk Score", f"{row['risk_score']:.1f}" if pd.notna(row["risk_score"]) else "—")
    c4.metric("Risk Bucket", row["risk_bucket"])
    st.write(f"**Quadrant:** {row['quadrant']}")
    desc = CLUSTER_DESCRIPTIONS.get(str(int(row["cluster"])), "")
    if desc: st.caption(f"Cluster meaning: {desc}")
    st.write(f"**Drivers:** {row['drivers']}")

    with st.expander("Row details"):
        st.dataframe(row.to_frame().rename(columns={0:"value"}))

    # ---------- Download processed results ----------
    st.write("### Download Processed Results")
    out_cols = [ID_COL, "cluster", "risk_score", "risk_bucket", "performance_band", "quadrant"]
    out = joined[out_cols].copy()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (clusters + risk + quadrant)",
        data=csv,
        file_name="hr_results.csv",
        mime="text/csv"
    )

# -------------------------------------------------------
# TAB 2 — 🧪 Simulator (single employee + batch clustering)
# -------------------------------------------------------
with tab_sim:
    st.subheader("Employee Simulator & Quick Clustering")

    # shared config
    CLUSTER_RISK_LEVEL = {0: "Low", 1: "Moderate", 2: "High"}
    CLUSTER_REASON = {
        0: "High Performers. Watch out for perceived salary unfairness; offer financial security, recognition & growth, and motivation.",
        1: "Balanced Group. Ensure work-life balance; watch stagnation & motivation. Offer flexible schedules, team building, and development.",
        2: "High Earners, Not Effective. High salary & overtime but low performance/bonus; watch burnout. Prioritize mental health, WLB, and performance focus.",
    }

    def derive_income_class(monthly_income: float, df_reference: pd.DataFrame | None) -> str:
        # Low/Mid/High via tertiles, else fallback to low-cut
        try:
            if df_reference is not None and "monthly_income" in df_reference.columns and len(df_reference) >= 10:
                p33, p66 = np.nanpercentile(df_reference["monthly_income"], [33, 66])
                if monthly_income <= p33: return "Low"
                if monthly_income <= p66: return "Mid"
                return "High"
            if monthly_income <= CUTS.get("income_lo_val", 0):
                return "Low"
            return "High"
        except Exception:
            return "Low"

    def predict_cluster_for_row(row_df: pd.DataFrame) -> int:
        feat_cols = [c for c in NUM_COLS + CAT_COLS if c in row_df.columns]
        raw = PIPELINE_FULL.predict(row_df[feat_cols])
        if LABEL_MAP:
            mapped = pd.Series(raw).map(LABEL_MAP).fillna(raw).astype(int).values
        else:
            mapped = raw
        return int(mapped[0])

    def ensure_income_class(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "income_class" not in df.columns and "monthly_income" in df.columns:
            try:
                q = pd.qcut(df["monthly_income"], q=3, labels=["Low","Mid","High"], duplicates="drop")
                df["income_class"] = q.astype(str)
            except Exception:
                med = np.nanmedian(df["monthly_income"])
                df["income_class"] = np.where(df["monthly_income"] <= med, "Low", "High")
        return df

    # Dynamic department options
    if 'joined' in globals() and "department" in joined.columns:
        dept_options = sorted(joined["department"].dropna().astype(str).unique().tolist())
    elif 'df_input' in globals() and "department" in df_input.columns:
        dept_options = sorted(df_input["department"].dropna().astype(str).unique().tolist())
    else:
        dept_options = ["Sales","Tech","Finance","HR","Operations"]

    tab_single, tab_batch = st.tabs(["🔮 Single Employee", "📦 Batch"])

    # ----- Single Employee -----
    with tab_single:
        colL, colR = st.columns(2)
        with colL:
            age_bin = st.select_slider("Age Range", options=[f"{a}-{a+4}" for a in range(20, 81, 5)], value="25-29")
            age_val = int(age_bin.split("-")[0]) + 2  # midpoint
            edu = st.selectbox("Education Level", ["SMA","D3","S1","S2"], index=2)
            dept = st.selectbox("Department", dept_options)
            weekly_hours = st.slider("Weekly Hours", 20, 80, 40, 1)
            bonus_pct = st.slider("Bonus Percentage (%)", 0.0, 50.0, 10.0, 0.5)
            monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=50000.0, value=8000.0, step=100.0)
            years_exp = st.slider("Years of Experience", 1, 45, 6, 1)

            gender = "Male"
            marital_status = "Single"
            city = "Jakarta"

        with colR:
            st.info("Performance and overtime default to reasonable values. Expose them later if needed.")
            performance_score = 75.0
            overtime_hours = max(0, weekly_hours - 40)

            row = pd.DataFrame([{
                ID_COL: 0,
                "age": age_val,
                "education_level": edu,
                "department": dept,
                "weekly_hours": weekly_hours,
                "bonus_percentage": bonus_pct,
                "monthly_income": monthly_income,
                "years_experience": years_exp,
                "gender": gender,
                "marital_status": marital_status,
                "city": city,
                "performance_score": performance_score,
                "overtime_hours": overtime_hours,
            }])
            row["income_class"] = derive_income_class(monthly_income, df_reference=None)

            if st.button("Predict Cluster & Risk", use_container_width=True):
                try:
                    c = predict_cluster_for_row(row)
                    risk = CLUSTER_RISK_LEVEL.get(c, "Unknown")
                    reason = CLUSTER_REASON.get(c, "")
                    st.success(
                        f"This Employee is in **Cluster {c}** → resignation chance: **{risk}**.\n\n"
                        f"- Overtime hours: **{overtime_hours:.1f}**\n"
                        f"- Income class: **{row.loc[0,'income_class']}**\n\n"
                        f"**Reason:** {reason}"
                    )
                    desc = MODEL_META.get("clustering", {}).get("descriptions", {}).get(str(c), "")
                    if desc: st.caption(f"Cluster meaning: {desc}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # ----- Batch (Dummy/Upload) -----
    with tab_batch:
        mode = st.radio("Choose input source", ["Generate Dummy Data", "Upload CSV"], horizontal=True)

        if mode == "Generate Dummy Data":
            n_rows = st.slider("How many dummy employees?", 20, 500, 100, 10)
            rng = np.random.default_rng(42)
            dummy = pd.DataFrame({
                ID_COL: np.arange(1, n_rows + 1),
                "age": rng.integers(20, 60, n_rows),
                "gender": rng.choice(["Male", "Female"], n_rows),
                "marital_status": rng.choice(["Single", "Married", "Divorced"], n_rows, p=[0.5, 0.45, 0.05]),
                "city": rng.choice(["Jakarta","Bandung","Surabaya"], n_rows),
                "education_level": rng.choice(["SMA","D3","S1","S2"], n_rows, p=[0.2, 0.2, 0.45, 0.15]),
                "years_experience": rng.integers(1, 30, n_rows),
                "weekly_hours": rng.integers(30, 60, n_rows),
                "department": rng.choice(dept_options, n_rows),
                "bonus_percentage": rng.normal(10, 5, n_rows).clip(0, 40),
                "performance_score": rng.normal(75, 12, n_rows).clip(0, 100),
                "monthly_income": rng.normal(9000, 3000, n_rows).clip(1000, 40000),
            })
            dummy["overtime_hours"] = np.maximum(0, dummy["weekly_hours"] - 40)
            dummy = ensure_income_class(dummy)

            if st.button("Cluster Dummy Data", use_container_width=True):
                try:
                    feat_cols = [c for c in NUM_COLS + CAT_COLS if c in dummy.columns]
                    raw = PIPELINE_FULL.predict(dummy[feat_cols])
                    mapped = pd.Series(raw).map(LABEL_MAP).fillna(raw).astype(int).values
                    dummy_out = dummy.copy()
                    dummy_out["cluster"] = mapped
                    dummy_out["resign_risk"] = dummy_out["cluster"].map(CLUSTER_RISK_LEVEL)
                    st.dataframe(dummy_out.head(30), use_container_width=True)
                    csv = dummy_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download clustered dummy CSV", data=csv,
                                       file_name="dummy_clustered.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

        else:
            up = st.file_uploader("Upload CSV to cluster", type=["csv"], key="batch_uploader_tab")
            if up is not None:
                try:
                    dfu = pd.read_csv(up)
                    dfu = ensure_income_class(dfu)
                    feat_cols = [c for c in NUM_COLS + CAT_COLS if c in dfu.columns]
                    raw = PIPELINE_FULL.predict(dfu[feat_cols])
                    mapped = pd.Series(raw).map(LABEL_MAP).fillna(raw).astype(int).values
                    dfu_out = dfu.copy()
                    dfu_out["cluster"] = mapped
                    dfu_out["resign_risk"] = dfu_out["cluster"].map(CLUSTER_RISK_LEVEL)
                    st.dataframe(dfu_out.head(30), use_container_width=True)
                    csv = dfu_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download clustered CSV", data=csv,
                                       file_name="clustered_upload.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Failed to process uploaded CSV: {e}")
