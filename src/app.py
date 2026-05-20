import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="ChurnSight | E-Commerce Dashboard",
    page_icon="📦",
    layout="wide",
)


# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["features"]

@st.cache_data
def load_data():
    return pd.read_csv("customer_data.csv")

try:
    model, features = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Make sure you run ChurnModel.py first.")
    st.stop()


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.title("ChurnSight")
st.markdown("#### E-Commerce Customer Churn Prediction Dashboard")
st.markdown(f"Model: **{type(model).__name__}**")
st.markdown("---")


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab1, tab2 = st.tabs(["Predict Customer", "Dataset Overview"])


# ─────────────────────────────────────────
# TAB 1 - PREDICT
# ─────────────────────────────────────────

with tab1:
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        recency_days = st.slider(
            "Days since last purchase", 1, 365, 90,
            help="How many days ago did this customer last buy something?"
        )
        frequency = st.slider(
            "Number of purchases (last 6 months)", 1, 50, 10,
            help="Total number of orders placed in the last 6 months."
        )
        monetary = st.number_input(
            "Total amount spent (USD)", min_value=10.0, max_value=5000.0,
            value=500.0, step=10.0,
            help="Cumulative spend across all orders."
        )
        avg_order_value = round(monetary / frequency, 2)
        st.metric("Avg. Order Value (auto-calculated)", f"${avg_order_value}")

    with col2:
        num_returns = st.slider(
            "Number of returned items", 0, 10, 1,
            help="How many items has this customer returned?"
        )
        email_open_rate = st.slider(
            "Email open rate", 0.0, 1.0, 0.5, step=0.01,
            help="Fraction of marketing emails this customer has opened (0 = none, 1 = all)."
        )
        days_since_signup = st.slider(
            "Days since account creation", 30, 1825, 365,
            help="How long has this customer had an account?"
        )

    st.markdown("---")

    if st.button("Predict Churn Risk", use_container_width=True):
        input_data = pd.DataFrame([{
            "recency_days":      recency_days,
            "frequency":         frequency,
            "monetary":          monetary,
            "avg_order_value":   avg_order_value,
            "num_returns":       num_returns,
            "email_open_rate":   email_open_rate,
            "days_since_signup": days_since_signup,
        }])[features]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("### Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if prediction == 1:
                st.error("HIGH CHURN RISK")
            else:
                st.success("LOW CHURN RISK")

        with res_col2:
            st.metric("Churn Probability", f"{probability:.1%}")

        with res_col3:
            if probability > 0.7:
                risk_level = "High"
            elif probability > 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            st.metric("Risk Level", risk_level)

        st.markdown("#### Recommended Action")
        if probability > 0.7:
            st.warning(
                "This customer is very likely to churn. Consider sending a **personalised discount voucher** "
                "or a **re-engagement email** as soon as possible."
            )
        elif probability > 0.4:
            st.info(
                "This customer shows some signs of disengagement. A **targeted promotion** or "
                "**loyalty reward** could help retain them."
            )
        else:
            st.success(
                "This customer appears engaged and healthy. Continue standard **email marketing** "
                "and monitor their activity."
            )


# ─────────────────────────────────────────
# TAB 2 - DATASET OVERVIEW
# ─────────────────────────────────────────

with tab2:
    st.subheader("Dataset Overview")
    st.markdown(f"Training dataset: **{len(df):,} synthetic customers**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Churned", f"{df['churned'].sum():,}", f"{df['churned'].mean():.1%}")
    m3.metric("Avg. Recency (days)", f"{df['recency_days'].mean():.0f}")
    m4.metric("Avg. Purchases / 6mo", f"{df['frequency'].mean():.1f}")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Churn Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df["churned"].value_counts()
        ax.pie(
            counts, labels=["Not Churned", "Churned"],
            autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

    with chart_col2:
        st.markdown("**Recency by Churn Status**")
        fig, ax = plt.subplots(figsize=(4, 3))
        df.boxplot(column="recency_days", by="churned", ax=ax, patch_artist=True)
        ax.set_title("")
        ax.set_xlabel("Churned (0 = No, 1 = Yes)")
        ax.set_ylabel("Days since last purchase")
        plt.suptitle("")
        st.pyplot(fig)
        plt.close()

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("**Purchase Frequency Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        df.groupby("churned")["frequency"].plot(
            kind="hist", alpha=0.6, bins=20, ax=ax,
            legend=True, color=["#4CAF50", "#F44336"]
        )
        ax.set_xlabel("Number of purchases (6 months)")
        ax.legend(["Not Churned", "Churned"])
        st.pyplot(fig)
        plt.close()

    with chart_col4:
        st.markdown("**Email Open Rate by Churn Status**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.violinplot(data=df, x="churned", y="email_open_rate",
                       palette=["#4CAF50", "#F44336"], ax=ax)
        ax.set_xlabel("Churned (0 = No, 1 = Yes)")
        ax.set_ylabel("Email open rate")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Raw Data Sample**")
    st.dataframe(df.sample(10, random_state=42).reset_index(drop=True), use_container_width=True)