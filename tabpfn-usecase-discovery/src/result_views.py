"""Plain-language result views; graphs only consume measured runner artifacts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .schemas import AnalysisResult, BenchmarkResult


def _chart(figure):
    figure.update_layout(margin=dict(t=55, b=30), legend_title_text="", height=390)
    st.plotly_chart(figure, width="stretch")


def render_benchmark(result: BenchmarkResult, *, forecasting=False):
    st.subheader("Traditional models vs TabPFN")
    scores = pd.DataFrame(result.metrics).T
    scores.index.name = "Model"
    scores.columns = [f"{metric} {'↓' if metric in {'MAE', 'RMSE'} else '↑'}" for metric in scores.columns]
    st.dataframe(scores.style.format("{:.3f}").highlight_max(subset=[c for c in scores if c.endswith('↑')], props="background-color: #d9ead3; color: #16351e;")
                 .highlight_min(subset=[c for c in scores if c.endswith('↓')], props="background-color: #d9ead3; color: #16351e;"), width="stretch")
    st.caption("↑ Higher is better · ↓ Lower is better. All models use the same unseen test records.")
    render_explanations(result)
    with st.expander("How to interpret this evaluation"):
        if "classification" in result.plan.task_type:
            st.write("Accuracy measures correct predictions overall. Balanced accuracy gives each outcome equal weight. Precision measures how often alerts are correct; recall measures how many important outcomes are found. F1 balances precision and recall; AUC scores assess ranking quality.")
        else:
            st.write("MAE is the typical error in the outcome's units. RMSE gives larger mistakes more weight. R² compares the estimates with predicting the average; it can be negative when the model performs poorly.")
        if forecasting:
            st.write("The latest dates are held out together. This is a historical forecast test using supplied inputs; those inputs must also be available when making real forecasts.")
        for note in result.notes:
            st.write(note)


def render_explanations(result: BenchmarkResult):
    explanations = result.case_explanations
    if explanations is None or explanations.empty:
        st.info("Enable “Explain the predictions” before running the evaluation to inspect an individual result.")
        return
    if "SHAP value" not in explanations.columns:
        st.info("Run the evaluation again with “Explain the predictions” enabled to generate SHAP values.")
        return
    if explanations is not None and not explanations.empty:
        record = st.selectbox("Inspect an individual result", explanations["Record"].unique(), key="explanation_record")
        case = explanations[explanations["Record"] == record].copy()
        case = case.loc[case["SHAP value"].abs().sort_values(ascending=False).index]
        if result.predictions is not None:
            observed = result.predictions[result.predictions["Record"] == record].iloc[0]
            st.write(f"Observed outcome: **{observed['Observed']}** · Predicted outcome: **{observed['Predicted']}**")
        classification = "classification" in result.plan.task_type
        baseline = float(case.iloc[0]["Base value"])
        prediction = float(case.iloc[0]["Prediction"])
        outcome = case.iloc[0]["Explained outcome"]
        if classification:
            st.write(f"Probability of **{outcome}**: reference **{baseline:.1%}** → this record **{prediction:.1%}**.")
        else:
            st.write(f"Estimated **{outcome}**: reference **{baseline:.3f}** → this record **{prediction:.3f}**.")
        scale = 100 if classification else 1
        contributions = case["SHAP value"].to_numpy() * scale
        labels = [f"{row['Field']} = {str(row['Field value'])[:24]}" for _, row in case.iterrows()]
        figure = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * len(case) + ["total"],
            y=["Reference prediction", *labels, "This prediction"], x=[baseline * scale, *contributions, 0],
            text=[f"{baseline * scale:.3f}", *[f"{v:+.3f}" for v in contributions], f"{prediction * scale:.3f}"],
            textposition="outside", connector=dict(line=dict(color="#8593a8")),
            increasing=dict(marker=dict(color="#df5367")), decreasing=dict(marker=dict(color="#5271e6")),
            totals=dict(marker=dict(color="#6b7b94")),
        ))
        figure.update_layout(title="SHAP contributions to this prediction", showlegend=False,
                             xaxis_title="Probability (%) — SHAP changes in percentage points" if classification else "Predicted outcome (original units)",
                             yaxis=dict(autorange="reversed"))
        _chart(figure)
        st.caption("Red contributions raise the prediction; blue contributions lower it. The reference prediction plus all SHAP values equals this record's prediction. SHAP explains model behavior, not cause and effect.")
        st.caption(f"{case.iloc[0]['Method']} · Reference: {int(case.iloc[0]['Background records'])} sampled learning records.")


def render_analysis(result: AnalysisResult):
    artifacts = result.artifacts
    st.subheader("Results of the evaluation")
    if result.capability == "imputation":
        scores = result.metrics
        st.dataframe(scores.style.format({"TabPFN": "{:.3f}", "Simple baseline": "{:.3f}"}), width="stretch", hide_index=True)
        st.caption("Compare recovery quality for each field. Lower error is better for numbers; higher accuracy is better for categories.")
        field = st.selectbox("Field to inspect", scores["Field"].unique(), key="imputation_field")
        recovered = artifacts["recovery"]
        recovered = recovered[recovered["Field"] == field]
        record = st.selectbox("Inspect an individual result", recovered["Record"], key="imputation_record")
        case = recovered[recovered["Record"] == record].iloc[0]
        if bool(case["Numeric"]):
            _chart(px.bar(x=["Known value", "Recovered value"], y=[case["Known value"], case["Recovered value"]], labels={"x": "", "y": field}, title=f"Recovery for record {record}"))
            st.caption("The known value was temporarily hidden. Compare it with the model's recovered value.")
        else:
            st.write(f"Known category: **{case['Known value']}** · Recovered category: **{case['Recovered value']}**")
        download = artifacts["completed"].copy()
        for column in download:
            download[f"Originally missing: {column}"] = artifacts["missing_flags"][column]
        if "identifiers" in artifacts:
            download = download.join(artifacts["identifiers"])
        download.insert(0, "Source record", download.index.astype(str))
        st.download_button("Download completed evaluation records", download.to_csv(index=False), "completed-records.csv", "text/csv")
    elif result.capability == "anomaly_detection":
        queue = artifacts["queue"]
        budget = min(10, len(queue))
        baseline_top = set(queue.nlargest(budget, "Isolation Forest")["Record"])
        agreement = len(set(queue.head(budget)["Record"]) & baseline_top) / budget
        comparison = pd.DataFrame({"Model": ["Isolation Forest", "TabPFN conditional models"], "Records evaluated": [len(queue)] * 2,
                                   "Top-10 alert agreement": [agreement] * 2, "Verified precision / recall": ["Review labels required"] * 2})
        st.dataframe(comparison.style.format({"Top-10 alert agreement": "{:.0%}"}), hide_index=True, width="stretch")
        st.caption("Alert agreement compares the models' review lists. It does not measure accuracy; verified outcomes are needed for precision and recall.")
        record = st.selectbox("Inspect an individual result", queue["Record"], key="anomaly_record")
        factors = artifacts["contributions"].set_index("Record").loc[record].rename_axis("Field").reset_index(name="Surprise percentile")
        _chart(px.bar(factors.sort_values("Surprise percentile"), x="Surprise percentile", y="Field", orientation="h", title="Fields that made this record unusual"))
        st.caption("A high field percentile means its recorded value was especially surprising given the other fields. This is a reason to investigate, not proof that something is wrong.")
        st.download_button("Download the review queue", queue.to_csv(index=False), "unusual-records.csv", "text/csv")
    elif result.capability == "synthetic_generation":
        generated = artifacts["generated"]
        fidelity = artifacts["fidelity"]
        comparison = fidelity.pivot(index="Model", columns="Field", values="Distance")
        comparison.columns = [f"{column} distance ↓" for column in comparison]
        st.dataframe(comparison.style.format("{:.3f}").highlight_min(props="background-color: #d9ead3; color: #16351e;"), width="stretch")
        st.caption("Lower distribution distance means closer to untouched original records. Compare each field separately; different fields use different scales.")
        st.download_button("Download generated records", generated.to_csv(index=False), "generated-records.csv", "text/csv")
    with st.expander("How to interpret this evaluation"):
        for note in result.notes:
            st.write(note)
