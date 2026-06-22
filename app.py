"""
AMI Mortality Risk Predictor — Streamlit App
部署: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import catboost as cb
import matplotlib.pyplot as plt

# ── 页面配置 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMI Mortality Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── 加载模型和配置 ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """加载 CatBoost 模型"""
    model = cb.CatBoost()
    model.load_model("ami_catboost_model.cbm")
    return model

@st.cache_data
def load_config():
    """加载模型配置"""
    with open("model_config.json", "r") as f:
        return json.load(f)

model = load_model()
config = load_config()

# ── 侧边栏 ───────────────────────────────────────────────────────────
st.sidebar.title("🫀 AMI Mortality Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Risk Prediction", "Feature Importance", "Model Info"],
    label_visibility="collapsed"
)

# ── 预测函数 ─────────────────────────────────────────────────────────
def predict_risk(inputs: dict) -> float:
    """使用 CatBoost 模型预测死亡风险"""
    # 构建输入数据框
    df = pd.DataFrame([inputs])

    # 应用 z-score 缩放
    if config.get("scale_params"):
        center = config["scale_params"]["center"]
        scale = config["scale_params"]["scale"]
        for col in center:
            if col in df.columns and scale.get(col, 0) > 0:
                df[col] = (df[col] - center[col]) / scale[col]

    # CatBoost 预测
    pool = cb.Pool(df)
    prob = model.predict(pool, prediction_type="Probability")[:, 1]
    return float(prob[0])

# ── 风险分层 ─────────────────────────────────────────────────────────
def risk_level(prob: float) -> tuple:
    cutoff = config.get("best_cutoff", 0.084)
    if prob < cutoff:
        return "🟢 Low Risk", f"< {cutoff*100:.1f}%"
    elif prob < 0.25:
        return "🟡 Intermediate Risk", "8.4% – 25%"
    elif prob < 0.50:
        return "🟠 High Risk", "25% – 50%"
    else:
        return "🔴 Very High Risk", "≥ 50%"

# ======================================================================
# 页面 1: 风险预测
# ======================================================================
if page == "Risk Prediction":
    st.title("One-Year Mortality Risk Prediction")
    st.markdown(
        "For patients with **Acute Myocardial Infarction** complicated by "
        "**Aortic Valve Stenosis**"
    )
    st.markdown("---")

    # 输入区域
    inputs = {}
    features = config["features"]
    disp = config["display_names"]

    # 分成 3 列布局
    cols_per_row = 4
    feat_items = list(features.items())
    rows = [feat_items[i:i + cols_per_row] for i in range(0, len(feat_items), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for i, (feat_name, meta) in enumerate(row):
            label = disp.get(feat_name, feat_name)

            if meta["type"] == "numeric":
                inputs[feat_name] = cols[i].number_input(
                    label,
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=float(meta.get("step", 1)),
                    key=feat_name
                )
            else:
                options = meta.get("options", [])
                inputs[feat_name] = cols[i].selectbox(
                    label,
                    options=options,
                    index=0,
                    key=feat_name
                )

    st.markdown("---")

    # 预测按钮
    col_btn, col_result = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button(
            "🧮 Calculate Risk",
            type="primary",
            use_container_width=True
        )

    if predict_clicked:
        with st.spinner("Calculating..."):
            prob = predict_risk(inputs)
            level, level_desc = risk_level(prob)

        with col_result:
            # 大号概率显示
            st.markdown(
                f"<h1 style='text-align:center; color:#d9534f;'>"
                f"Mortality Risk: {prob*100:.1f}%</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h3 style='text-align:center;'>Risk Category: {level}</h3>",
                unsafe_allow_html=True
            )

# ======================================================================
# 页面 2: SHAP 特征重要性
# ======================================================================
elif page == "Feature Importance":
    st.title("SHAP Feature Importance")
    st.markdown("mean(|SHAP|) — higher values = greater contribution to mortality prediction")
    st.markdown("---")

    # 优先加载 R 导出的 SHAP 值，否则用 CatBoost 内置重要性
    try:
        with open("shap_importance.json", "r") as f:
            shap_data = json.load(f)
        df_imp = pd.DataFrame({
            "Feature": shap_data["display_names"],
            "Importance": shap_data["importance"]
        })
    except FileNotFoundError:
        fi = config["feature_importance"]
        disp = config["display_names"]
        df_imp = pd.DataFrame({
            "Feature": [disp.get(f, f) for f in fi["features"]],
            "Importance": fi["importance"]
        })

    df_imp = df_imp.sort_values("Importance", ascending=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"],
                    color="#E64B35", alpha=0.85)
    for bar, val in zip(bars, df_imp["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, color="#4d4d4d")

    ax.set_xlabel("mean(|SHAP|)", fontsize=12, fontweight="bold")
    ax.set_title("SHAP Feature Importance (KernelSHAP, Test Set)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, df_imp["Importance"].max() * 1.15)

    st.pyplot(fig)

    # 可选: 如果 SHAP 值 CSV 存在，显示 beeswarm
    import os
    if os.path.exists("shap_values.csv"):
        with st.expander("SHAP Beeswarm (click to expand)"):
            st.caption("Full SHAP value distribution across test set patients")
            # 简化版 beeswarm：每特征的 SHAP 分布
            df_shap = pd.read_csv("shap_values.csv").drop(columns=["row_id"])
            disp = config["display_names"]

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            # 按 mean(|SHAP|) 排序
            order = df_imp["Feature"].tolist()
            plot_data = []
            for feat_name, disp_name in zip(shap_data["features"], shap_data["display_names"]):
                if feat_name in df_shap.columns:
                    for val in df_shap[feat_name]:
                        plot_data.append({"Feature": disp_name, "SHAP": val})

            df_plot = pd.DataFrame(plot_data)
            import numpy as np
            # 只显示 top 10
            top_features = df_imp["Feature"].tail(10).tolist()
            df_plot_top = df_plot[df_plot["Feature"].isin(top_features)]

            colors = ["#E64B35" if v > 0 else "#4DBBD5" for v in df_plot_top["SHAP"]]
            y_pos = [top_features.index(f) for f in df_plot_top["Feature"]]
            ax2.scatter(df_plot_top["SHAP"],
                       [top_features.index(f) for f in df_plot_top["Feature"]],
                       c=colors, alpha=0.3, s=8)
            ax2.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features)
            ax2.set_xlabel("SHAP Value", fontsize=12, fontweight="bold")
            ax2.set_title("SHAP Beeswarm (Test Set)", fontsize=14, fontweight="bold")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            st.pyplot(fig2)

# ======================================================================
# 页面 3: 模型信息
# ======================================================================
else:
    st.title("Model Information")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Algorithm")
        st.markdown("**CatBoost** (Categorical Boosting)")
        st.markdown(
            "Gradient boosting algorithm with native categorical feature "
            "support and ordered boosting for robustness to overfitting."
        )

        st.subheader("Performance (Test Set)")
        perf = config.get("performance", {})
        perf_df = pd.DataFrame({
            "Metric": list(perf.keys()),
            "Value": [f"{v:.4f}" for v in perf.values()]
        })
        st.table(perf_df)

    with col2:
        st.subheader("Study Population")
        st.markdown("""
        - **980 patients** with AMI + moderate-to-severe AS
        - **82 hospitals** in Tianjin, China
        - **2010–2024**
        - One-year mortality rate: **15.2%**
        """)

        st.subheader("Risk Cutoff")
        cutoff = config.get("best_cutoff", 0.084)
        st.markdown(f"Youden-optimal threshold: **{cutoff*100:.1f}%**")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "This tool is for **research and educational purposes only**. "
        "It should **not** be used as the sole basis for clinical decision-making."
    )

# ── Footer ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("AMI Mortality Predictor v1.0 | CatBoost Model")
st.sidebar.caption("© 2024 Research Purpose Only")
