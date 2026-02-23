"""
Streamlit Application for Mobile Money Fraud Detection
Dashboard with Real-Time Monitoring, Fraud Alerts, and Analytics
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from preprocessing import (
    preprocess_data,
    engineer_features,
    select_features,
)
from models import HybridModel, save_models, load_models
from core.session_state import (
    initialize_session_state,
    reset_for_logout,
    clear_investigation_queue,
    reset_dashboard_statistics,
)
from core.risk import get_risk_level, get_recommended_action, calculate_objective_status
from ui.styles import load_custom_css

DATA_DIR = "data"
DEFAULT_TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")


# Page configuration
st.set_page_config(
    page_title="Mobile Money Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure all required session keys exist before any page logic runs.
initialize_session_state()


# Custom CSS for styling - Light White and Blue Theme
def dashboard_page():
    """Main Dashboard Page - aligned with project objectives."""
    load_custom_css()

    st.title("Fraud Detection Dashboard")

    if not st.session_state.dashboard_data_loaded:
        with st.spinner("Loading models and initializing dashboard..."):
            model_path = "trained_models"
            if os.path.exists(model_path) and os.path.exists(
                os.path.join(model_path, "scaler.pkl")
            ):
                try:
                    hybrid_model, scaler = load_models(model_path)
                    st.session_state.hybrid_model = hybrid_model
                    st.session_state.scaler = scaler
                    st.session_state.models_loaded = True
                except Exception as e:
                    st.warning(f"Could not load models: {e}")
            # Removed auto-loading of test data - transactions should start at 0
            st.session_state.dashboard_data_loaded = True

    fraud_rate = (
        st.session_state.fraudulent_count / max(st.session_state.total_transactions, 1)
    ) * 100
    active_count = len(
        [
            t
            for t in st.session_state.flagged_transactions
            if t.get("transaction_id", t.get("nameOrig", "N/A"))
            not in st.session_state.resolved_alerts
        ]
    )
    st.session_state.active_alerts = active_count
    objective_status = calculate_objective_status()
    completed = sum(objective_status.values())
    progress = completed / len(objective_status)
    resolved_count = len(st.session_state.resolved_alerts)

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">Hybrid Unsupervised Fraud Intelligence</div>
            <div class="hero-subtitle">Real-time and batch detection using Isolation Forest, Autoencoder, and DBSCAN.</div>
            <span class="soft-tag">Objectives Completed: {completed}/5</span>
            <span class="soft-tag">Fraud Rate: {fraud_rate:.2f}%</span>
            <span class="soft-tag">Active Alerts: {active_count}</span>
            <span class="soft-tag">Resolved Alerts: {resolved_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "TOTAL TRANSACTIONS",
            f"{st.session_state.total_transactions:,}",
            f"+{len(st.session_state.recent_transactions)}",
        )
    with col2:
        st.metric(
            "FRAUDULENT TRANSACTIONS",
            f"{st.session_state.fraudulent_count}",
            delta_color="inverse",
        )
    with col3:
        st.metric("FRAUD RATE", f"{fraud_rate:.2f}%", "Real-time")
    with col4:
        st.metric("ACTIVE ALERTS", f"{active_count}", "Needs review")

    exec_col1, exec_col2 = st.columns(2)
    with exec_col1:
        st.markdown("### Executive Risk Gauge")
        risk_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fraud_rate,
                title={"text": "Current Fraud Rate (%)"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2d9bf0"},
                    "steps": [
                        {"range": [0, 2], "color": "#eaf7ef"},
                        {"range": [2, 5], "color": "#fff6e5"},
                        {"range": [5, 100], "color": "#fdecec"},
                    ],
                },
            )
        )
        risk_gauge.update_layout(
            template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=260
        )
        st.plotly_chart(risk_gauge, width="stretch")

    with exec_col2:
        st.markdown("### Objective Completion Gauge")
        objective_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=completed,
                title={"text": "Objectives Completed"},
                number={"suffix": "/5"},
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": "#1f5f99"},
                    "steps": [
                        {"range": [0, 2], "color": "#fdecec"},
                        {"range": [2, 4], "color": "#fff6e5"},
                        {"range": [4, 5], "color": "#eaf7ef"},
                    ],
                },
            )
        )
        objective_gauge.update_layout(
            template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=260
        )
        st.plotly_chart(objective_gauge, width="stretch")

    st.markdown(
        "<div class='section-header'>Objectives Progress</div>", unsafe_allow_html=True
    )
    st.progress(progress, text=f"{completed}/5 objectives currently covered")

    for objective, done in objective_status.items():
        state_text = "Completed" if done else "Pending"
        state_class = "objective-card objective-complete" if done else "objective-card"
        st.markdown(
            f"<div class='{state_class}'><strong>{objective}</strong><br><small>Status: {state_text}</small></div>",
            unsafe_allow_html=True,
        )

    objective_rows = []
    objective_rows.append(
        {
            "Objective": "1. Upload mobile money dataset for analysis",
            "Status": (
                "Completed"
                if objective_status["1. Upload mobile money dataset for analysis"]
                else "Pending"
            ),
            "Evidence": f"{len(st.session_state.uploaded_datasets)} dataset(s) uploaded",
            "How Answered": "Real-Time and Batch upload modules",
        }
    )
    objective_rows.append(
        {
            "Objective": "2. Analyze transactions against normal behavior",
            "Status": (
                "Completed"
                if objective_status["2. Analyze transactions against normal behavior"]
                else "Pending"
            ),
            "Evidence": f"{st.session_state.analysis_runs} analysis run(s)",
            "How Answered": "Hybrid score from Isolation Forest + Autoencoder + DBSCAN",
        }
    )
    objective_rows.append(
        {
            "Objective": "3. Flag fraudulent activities",
            "Status": (
                "Completed"
                if objective_status["3. Flag fraudulent activities"]
                else "Pending"
            ),
            "Evidence": f"{len(st.session_state.flagged_transactions)} flagged transaction(s)",
            "How Answered": "Threshold-based anomaly blocking",
        }
    )
    objective_rows.append(
        {
            "Objective": "4. Validate flagged transactions via manual inspection",
            "Status": (
                "Completed"
                if objective_status["4. Validate flagged transactions via manual inspection"]
                else "Pending"
            ),
            "Evidence": f"{len(st.session_state.manual_reviews)} reviewed transaction(s)",
            "How Answered": "Investigation page decision workflow",
        }
    )
    objective_rows.append(
        {
            "Objective": "5. Recommend actions after manual analysis",
            "Status": (
                "Completed"
                if objective_status["5. Recommend actions after manual analysis"]
                else "Pending"
            ),
            "Evidence": "Recommendation engine active",
            "How Answered": "Risk-aware action suggestion for each alert",
        }
    )
    st.dataframe(pd.DataFrame(objective_rows), width="stretch")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("### Transaction Volume and Detection Trend")
        if len(st.session_state.transaction_history) > 0:
            history_df = pd.DataFrame(st.session_state.transaction_history[-120:])
            history_df = history_df.reset_index().rename(columns={"index": "sequence"})
            fig = px.line(
                history_df,
                x="sequence",
                y="amount",
                color="is_fraud",
                color_discrete_map={False: "#2d9bf0", True: "#e74c3c"},
                labels={
                    "sequence": "Transaction Sequence",
                    "amount": "Amount",
                    "is_fraud": "Flagged",
                },
                markers=True,
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_orientation="h",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Run Real-Time Monitor or Batch Analysis to populate this chart.")

    with chart_col2:
        st.markdown("### Risk Level Distribution")
        if len(st.session_state.flagged_transactions) > 0:
            risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            for txn in st.session_state.flagged_transactions:
                score = txn.get("fraud_score", txn.get("hybrid_score", 0.5))
                risk_level, _ = get_risk_level(score)
                risk_counts[risk_level] += 1
            risk_df = pd.DataFrame(
                {
                    "risk_level": list(risk_counts.keys()),
                    "count": list(risk_counts.values()),
                }
            )
            risk_df = risk_df[risk_df["count"] > 0]
            if not risk_df.empty:
                fig = px.pie(
                    risk_df,
                    values="count",
                    names="risk_level",
                    hole=0.55,
                    color="risk_level",
                    color_discrete_map={
                        "LOW": "#17a673",
                        "MEDIUM": "#f39c12",
                        "HIGH": "#e67e22",
                        "CRITICAL": "#e74c3c",
                    },
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend_orientation="h",
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No flagged transactions yet.")

    deep_col1, deep_col2 = st.columns(2)
    with deep_col1:
        st.markdown("### Amount Distribution by Decision")
        if len(st.session_state.transaction_history) > 0:
            dist_df = pd.DataFrame(st.session_state.transaction_history[-300:]).copy()
            dist_df["decision"] = dist_df["is_fraud"].map(
                {True: "Blocked", False: "Approved"}
            )
            amount_fig = px.violin(
                dist_df,
                x="decision",
                y="amount",
                box=True,
                points="outliers",
                color="decision",
                color_discrete_map={"Approved": "#2d9bf0", "Blocked": "#e74c3c"},
            )
            amount_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(amount_fig, width="stretch")
        else:
            st.info("No transaction history yet for amount analytics.")

    with deep_col2:
        st.markdown("### Hourly Fraud Pattern")
        if len(st.session_state.transaction_history) > 0:
            hour_df = pd.DataFrame(st.session_state.transaction_history[-500:]).copy()
            hour_df["hour"] = pd.to_datetime(hour_df["timestamp"]).dt.hour
            hour_df["blocked"] = hour_df["is_fraud"].astype(int)
            hour_summary = hour_df.groupby("hour", as_index=False).agg(
                blocked_rate=("blocked", "mean"), tx_count=("blocked", "count")
            )
            hour_summary["blocked_rate"] = hour_summary["blocked_rate"] * 100
            hourly_fig = px.bar(
                hour_summary,
                x="hour",
                y="blocked_rate",
                color="tx_count",
                color_continuous_scale=["#d6ebff", "#2d9bf0"],
                labels={
                    "blocked_rate": "Blocked Rate (%)",
                    "hour": "Hour of Day",
                    "tx_count": "Transactions",
                },
            )
            hourly_fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(hourly_fig, width="stretch")
        else:
            st.info("No timestamp pattern available yet.")

    smart_col1, smart_col2 = st.columns(2)
    with smart_col1:
        st.markdown("### Hybrid Score Trend and Cumulative Fraud Rate")
        if len(st.session_state.anomaly_scores_history) > 0:
            score_df = pd.DataFrame(st.session_state.anomaly_scores_history).copy()
            score_df["cumulative_fraud_rate"] = (
                score_df["blocked"].astype(int).cumsum() / (score_df.index + 1)
            ) * 100
            score_df["rolling_score"] = (
                score_df["score"].rolling(window=5, min_periods=1).mean()
            )

            trend_fig = px.line(
                score_df,
                x="transaction_id",
                y=["score", "rolling_score"],
                labels={
                    "value": "Hybrid Score",
                    "transaction_id": "Transaction Sequence",
                    "variable": "Metric",
                },
                color_discrete_sequence=["#2d9bf0", "#1f5f99"],
            )
            trend_fig.add_hline(
                y=float(
                    getattr(st.session_state.hybrid_model, "default_threshold", 0.75)
                ),
                line_dash="dash",
                line_color="#e74c3c",
            )
            trend_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(trend_fig, width="stretch")

            fraud_rate_fig = px.area(
                score_df,
                x="transaction_id",
                y="cumulative_fraud_rate",
                labels={
                    "transaction_id": "Transaction Sequence",
                    "cumulative_fraud_rate": "Fraud Rate (%)",
                },
                color_discrete_sequence=["#7fb3e6"],
            )
            fraud_rate_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fraud_rate_fig, width="stretch")
        else:
            st.info("Run detection to populate hybrid score intelligence.")

    with smart_col2:
        st.markdown("### Review Outcomes and Action Intelligence")
        if len(st.session_state.flagged_transactions) > 0:
            review_df = pd.DataFrame(st.session_state.flagged_transactions).copy()
            if "transaction_id" not in review_df.columns:
                review_df["transaction_id"] = review_df.index.astype(str)
            review_df["review_status"] = (
                review_df["transaction_id"]
                .astype(str)
                .apply(lambda tid: st.session_state.manual_reviews.get(tid, "Pending"))
            )

            review_counts = (
                review_df["review_status"]
                .value_counts()
                .rename_axis("review_status")
                .reset_index(name="count")
            )
            outcome_fig = px.pie(
                review_counts,
                names="review_status",
                values="count",
                hole=0.5,
                color="review_status",
                color_discrete_map={
                    "Confirmed Fraud": "#e74c3c",
                    "False Positive": "#3498db",
                    "Pending": "#f39c12",
                },
            )
            outcome_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(outcome_fig, width="stretch")

            if "recommended_action" in review_df.columns:
                action_counts = (
                    review_df["recommended_action"]
                    .fillna("No action")
                    .value_counts()
                    .head(8)
                    .rename_axis("recommended_action")
                    .reset_index(name="count")
                )
                action_fig = px.bar(
                    action_counts,
                    x="count",
                    y="recommended_action",
                    orientation="h",
                    color="count",
                    color_continuous_scale=["#d6ebff", "#2d9bf0"],
                )
                action_fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    coloraxis_showscale=False,
                    yaxis_title="Recommended Action",
                    xaxis_title="Frequency",
                )
                st.plotly_chart(action_fig, width="stretch")
        else:
            st.info("No flagged transactions yet for review analytics.")

    st.markdown(
        "<div class='section-header'>Recent Fraud Alerts and Recommended Actions</div>",
        unsafe_allow_html=True,
    )
    fraud_alerts = [
        t
        for t in st.session_state.flagged_transactions
        if t.get("transaction_id", t.get("nameOrig", "N/A"))
        not in st.session_state.resolved_alerts
    ]

    if len(fraud_alerts) == 0:
        st.info(
            "No active fraud alerts. Run Real-Time Monitor or Batch Analysis to detect transactions."
        )
    else:
        fraud_alerts_to_show = fraud_alerts[-10:]
        header_cols = st.columns([2, 2, 2, 2, 2, 3, 3])
        headers = [
            "TRANSACTION ID",
            "CUSTOMER ID",
            "AMOUNT",
            "RISK",
            "REVIEW",
            "RECOMMENDED ACTION",
            "ACTIONS",
        ]
        for col, header in zip(header_cols, headers):
            col.markdown(f"**{header}**")
        st.markdown("---")

        for idx, alert in enumerate(fraud_alerts_to_show):
            score = alert.get("fraud_score", alert.get("hybrid_score", 0.5))
            risk_level, risk_class = get_risk_level(score)
            txn_id = alert.get("transaction_id", alert.get("nameOrig", f"TXN_{idx}"))[:20]
            review_status = st.session_state.manual_reviews.get(txn_id, "Pending")
            recommendation = get_recommended_action(score, review_status)

            cols = st.columns([2, 2, 2, 2, 2, 3, 3])
            cols[0].markdown(f"**{txn_id}**")
            cols[1].markdown(
                alert.get("customer_id", alert.get("nameOrig", "N/A"))[:15]
            )
            cols[2].markdown(f"${alert.get('amount', 0):,.2f}")
            cols[3].markdown(
                f'<span class="badge-{risk_class}">{risk_level}</span>',
                unsafe_allow_html=True,
            )
            cols[4].markdown(review_status)
            cols[5].markdown(recommendation)

            with cols[6]:
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("Confirm", key=f"confirm_{txn_id}_{idx}"):
                        st.session_state.manual_reviews[txn_id] = "Confirmed Fraud"
                        st.rerun()
                with action_col2:
                    if st.button("Safe", key=f"safe_{txn_id}_{idx}"):
                        st.session_state.manual_reviews[txn_id] = "False Positive"
                        st.rerun()
                with action_col3:
                    if st.button("Resolve", key=f"resolve_{txn_id}_{idx}"):
                        st.session_state.resolved_alerts.add(txn_id)
                        st.rerun()

            with st.expander(f"View details: {txn_id}", expanded=False):
                st.json(alert)

    st.markdown(
        "<div class='section-header'>Recent Transactions</div>", unsafe_allow_html=True
    )
    transactions = (
        st.session_state.recent_transactions[:10]
        if len(st.session_state.recent_transactions) > 0
        else []
    )
    if len(transactions) == 0:
        st.info(
            "Process transactions in Real-Time Monitor or Batch Analysis to see them here."
        )
    else:
        for i in range(0, len(transactions), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(transactions):
                    txn = transactions[i + j]
                    status_badge = "FRAUD" if txn.get("is_fraud") else "CLEAN"
                    badge_class = (
                        "badge-fraud" if txn.get("is_fraud") else "badge-clean"
                    )
                    with col:
                        st.markdown(
                            f"""
                        <div class="transaction-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                <span style="color:#7f8c8d;font-size:13px;font-weight:600;">{txn.get('transaction_id', 'N/A')}</span>
                                <span class="{badge_class}">{status_badge}</span>
                            </div>
                            <div style="font-size:24px;font-weight:700;color:#2c3e50;margin-bottom:8px;">
                                ${txn.get('amount', 0):,.2f}
                            </div>
                            <div style="color:#7f8c8d;font-size:13px;margin-bottom:4px;">
                                Customer: {txn.get('customer_id', 'N/A')}
                            </div>
                            <div style="color:#95a5a6;font-size:12px;">
                                {txn.get('timestamp', 'N/A')}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )


def login_page():
    """Page 1: Login Authentication"""
    load_custom_css()
    
    # Add CSS to style the form container with border
    st.markdown(
        """
        <style>
        /* Style the form container */
        div[data-testid="stForm"] {
            border: 2px solid #c0c0c0 !important;
            border-radius: 12px !important;
            padding: 30px !important;
            background-color: #f8f9fa !important;
            margin: 20px 0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        }
        
        /* Hide the form submit button since we'll use custom buttons */
        div[data-testid="stForm"] button[kind="primary"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create centered layout with three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Header section with shield emoji
        st.markdown(
            "<h1 style='text-align: center;'>🛡️ Mobile Money Fraud Detection Portal</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h3 style='text-align: center; color: #666;'>Authorized Personnel Only</h3>",
            unsafe_allow_html=True,
        )

        st.write("")  # Empty space for visual separation
        
        # Use form to create a bordered container
        with st.form("login_form", clear_on_submit=False):
            # Input fields
            username = st.text_input(
                "Username", key="login_username", placeholder="Enter your username"
            )
            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="Enter your password",
            )

            st.write("")  # Space before buttons

            # Create two-column layout for buttons
            btn_col1, btn_col2 = st.columns([1, 1])

            with btn_col1:
                login_button = st.form_submit_button("Login", type="primary", use_container_width=True)

            with btn_col2:
                request_access_button = st.form_submit_button("Request Access", use_container_width=True)

        # Handle login button click
        if login_button:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

        # Handle request access button click
        if request_access_button:
            st.info(
                "Access request submitted. An administrator will review your request shortly."
            )


def load_or_train_models(X_train=None, scaler=None):
    """Helper function to load saved models or train and persist new models."""
    model_path = "trained_models"

    if os.path.exists(model_path) and os.path.exists(
        os.path.join(model_path, "scaler.pkl")
    ):
        try:
            st.info("Loading pre-trained models...")
            hybrid_model, scaler = load_models(model_path)
            st.session_state.hybrid_model = hybrid_model
            st.session_state.scaler = scaler
            st.session_state.models_loaded = True
            st.success("✓ Models loaded successfully!")
            return hybrid_model, scaler
        except Exception as e:
            st.warning(f"Could not load models: {e}. Training new models...")

    # Train new models (X_train must be provided if models don't exist)
    if X_train is None or scaler is None:
        raise ValueError(
            "Both X_train and scaler must be provided when training new models"
        )

    st.info("Training models... This may take a few minutes.")
    progress_bar = st.progress(0)

    input_dim = X_train.shape[1]
    hybrid_model = HybridModel(input_dim=input_dim)

    progress_bar.progress(25)
    with st.spinner("Training Isolation Forest..."):
        hybrid_model.iso_forest.train(X_train)

    progress_bar.progress(50)
    with st.spinner("Training Autoencoder..."):
        hybrid_model.autoencoder.train(X_train, epochs=30)

    progress_bar.progress(75)
    with st.spinner("Fitting DBSCAN..."):
        hybrid_model.dbscan.train(X_train)

    progress_bar.progress(100)
    st.success("✓ All models trained successfully!")

    # Persist trained models for reuse in later sessions
    save_models(hybrid_model, scaler, model_path)
    st.success("✓ Trained models saved successfully!")

    return hybrid_model, scaler


def ensure_models_ready(df_reference: pd.DataFrame):
    """Load existing models or train new ones from the provided dataset."""
    if (
        st.session_state.models_loaded
        and st.session_state.hybrid_model is not None
        and st.session_state.scaler is not None
    ):
        return

    required_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing_cols = [col for col in required_cols if col not in df_reference.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Please upload a CSV with the correct PaySim format.")
        st.stop()

    with st.spinner("Preparing models..."):
        model_path = "trained_models"
        scaler_path = os.path.join(model_path, "scaler.pkl")

        if os.path.exists(scaler_path):
            hybrid_model, scaler = load_or_train_models()
            st.session_state.hybrid_model = hybrid_model
            st.session_state.scaler = scaler
        else:
            st.info("Training new scaler and models...")
            df_train = df_reference.sample(
                min(10000, len(df_reference)), random_state=42
            )
            X_train, scaler, _ = preprocess_data(df_train.copy())
            hybrid_model, fitted_scaler = load_or_train_models(X_train, scaler)
            st.session_state.hybrid_model = hybrid_model
            st.session_state.scaler = fitted_scaler

        st.session_state.models_loaded = True



def real_time_monitor():
    """Page 2: Real-Time Transaction Monitor"""
    st.title("Real-Time Transaction Monitor")
    st.markdown("Monitor transactions as they arrive and detect fraud in real-time.")

    # File uploader for test dataset
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)", type=["csv"], key="realtime_upload"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_datasets.add(uploaded_file.name)
        # Load and prepare data
        df_test = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df_test)} transactions for simulation")

        # Prepare models once for this session
        ensure_models_ready(df_test)

        # Simulation controls
        col1, col2 = st.columns([1, 1])
        
        # 🔧 FIX: Use 0.21 as default threshold (matches Batch Analysis working threshold)
        suggested_threshold = 0.21  # Override model default to use empirically calibrated value

        with col1:
            num_transactions = st.number_input(
                "Transactions to simulate", min_value=1, max_value=100, value=20
            )

        with col2:
            threshold = st.slider(
                "Block Threshold",
                min_value=0.0,
                max_value=1.0,
                value=suggested_threshold,  # Use 0.21 as default
                step=0.01,
            )
            st.caption(
                f"Recommended threshold: {suggested_threshold:.2f}"
            )

        # Check if scaler is available before simulation
        if st.session_state.scaler is None:
            st.error(
                "Scaler not loaded! Please ensure models are loaded first or run 'python scripts/generate_sample_data.py' to prepare data and train models."
            )
            st.stop()

        # Start simulation button
        if st.button("▶ Start Simulation", type="primary"):
            st.session_state.analysis_runs += 1
            st.session_state.anomaly_scores_history = []
            st.session_state.flagged_transactions = []

            # Create placeholders for live updates
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

            # Sample transactions
            transactions_to_process = df_test.sample(
                n=min(num_transactions, len(df_test))
            ).reset_index(drop=True)

            # 🔧 BATCH PREPROCESSING: Process ALL transactions upfront using fixed pipeline
            with st.spinner("Preprocessing transactions..."):
                # Use the EXACT same pipeline as Batch Analysis
                X_batch, _, df_batch_processed = preprocess_data(
                    transactions_to_process.copy(), 
                    scaler=st.session_state.scaler  # Use pre-loaded scaler (TRANSFORM ONLY)
                )
                
                # Get predictions for entire batch using shock-absorbed model
                predictions_batch, hybrid_scores_batch, individual_scores_batch = (
                    st.session_state.hybrid_model.predict(
                        X_batch, threshold=threshold
                    )
                )

            total_processed = 0
            total_blocked = 0

            # Simulate transactions one-by-one using PRE-CALCULATED scores
            for idx in range(len(transactions_to_process)):
                transaction = transactions_to_process.iloc[idx]
                hybrid_score = hybrid_scores_batch[idx]  # Use pre-calculated score
                is_blocked = predictions_batch[idx] == 1
                
                # Extract individual component scores for this transaction
                iso_score = individual_scores_batch['isolation_forest'][idx]
                ae_score = individual_scores_batch['autoencoder'][idx]
                dbscan_score = individual_scores_batch['dbscan'][idx]

                total_processed += 1

                if is_blocked:
                    total_blocked += 1

                # Store anomaly score
                st.session_state.anomaly_scores_history.append(
                    {
                        "transaction_id": idx,
                        "score": hybrid_score,
                        "blocked": is_blocked,
                    }
                )

                # Display status
                with status_placeholder.container():
                    if is_blocked:
                        st.error(
                            f"**TRANSACTION BLOCKED** - Hybrid Score: {hybrid_score:.3f}"
                        )
                        st.markdown(
                            f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        - Isolation Forest Score: {iso_score:.3f}
                        - Autoencoder Score: {ae_score:.3f}
                        - DBSCAN Score: {dbscan_score:.3f}
                        """
                        )

                        # Store flagged transaction
                        flagged_data = transaction.to_dict()
                        flagged_data["transaction_id"] = f"TXN{2000 + idx}"
                        flagged_data["customer_id"] = transaction.get(
                            "nameOrig", f"CUST{300 + idx}"
                        )[:10]
                        flagged_data["hybrid_score"] = hybrid_score
                        flagged_data["fraud_score"] = hybrid_score
                        flagged_data["risk_level"] = get_risk_level(hybrid_score)[0]
                        flagged_data["timestamp"] = datetime.now().strftime(
                            "%m/%d/%Y, %I:%M:%S %p"
                        )
                        flagged_data["reason"] = (
                            f"High hybrid score ({hybrid_score:.3f})"
                        )
                        flagged_data["recommended_action"] = get_recommended_action(
                            hybrid_score
                        )
                        if ae_score > 0.5:
                            flagged_data["reason"] += ", High reconstruction error"
                        if iso_score > 0.5:
                            flagged_data["reason"] += ", Isolation Forest anomaly"
                        st.session_state.flagged_transactions.append(flagged_data)

                        # Update active alerts count
                        st.session_state.active_alerts = len(
                            st.session_state.flagged_transactions
                        )
                        st.session_state.fraudulent_count += 1
                    else:
                        st.success(
                            f"**TRANSACTION APPROVED** - Hybrid Score: {hybrid_score:.3f}"
                        )
                        st.markdown(
                            f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        """
                        )

                    # Add to recent transactions
                    recent_txn = {
                        "transaction_id": f"TXN{2000 + idx}",
                        "customer_id": transaction.get("nameOrig", f"CUST{300 + idx}")[:10],
                        "amount": transaction.get("amount", 0),
                        "is_fraud": is_blocked,
                        "timestamp": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
                    }
                    st.session_state.recent_transactions.insert(0, recent_txn)
                    st.session_state.recent_transactions = (
                        st.session_state.recent_transactions[:20]
                    )
                    st.session_state.total_transactions += 1

                    # Track for charts
                    st.session_state.transaction_history.append(
                        {
                            "amount": transaction.get("amount", 0),
                            "timestamp": datetime.now(),
                            "is_fraud": is_blocked,
                        }
                    )

                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Processed", total_processed)
                    m2.metric("Blocked", total_blocked)
                    m3.metric("Approved", total_processed - total_blocked)
                    m4.metric(
                        "Block Rate", f"{(total_blocked/total_processed*100):.1f}%"
                    )

                # Update chart
                if len(st.session_state.anomaly_scores_history) > 0:
                    with chart_placeholder.container():
                        scores_df = pd.DataFrame(
                            st.session_state.anomaly_scores_history
                        )
                        fig = px.scatter(
                            scores_df,
                            x="transaction_id",
                            y="score",
                            color="blocked",
                            color_discrete_map={False: "#2d9bf0", True: "#e74c3c"},
                            labels={
                                "transaction_id": "Transaction Number",
                                "score": "Hybrid Anomaly Score",
                                "blocked": "Blocked",
                            },
                        )
                        fig.add_hline(
                            y=threshold, line_dash="dash", line_color="#f39c12"
                        )
                        fig.update_layout(
                            template="plotly_white",
                            margin=dict(l=10, r=10, t=30, b=10),
                            title="Real-Time Anomaly Scores",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Simulate delay
                time.sleep(1)

            st.success(
                f"Simulation complete! Processed {total_processed} transactions, blocked {total_blocked}."
            )
            if total_blocked == 0:
                st.warning(
                    "No transactions crossed the threshold in this sample. Lower threshold slightly or increase sample size."
                )



def batch_analysis():
    """Page 3: Batch Analysis"""
    st.title("Batch Analysis")
    st.markdown("Upload a CSV file for bulk fraud detection analysis.")
    st.caption("Upload one CSV file per analysis run.")

    uploaded_file = st.file_uploader(
        "Upload CSV File", type=["csv"], key="batch_upload"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_datasets.add(uploaded_file.name)
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} transactions")

        # Show preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))

        # Use 0.21 as default threshold (matches Real-Time Monitor)
        suggested_threshold = 0.21
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=suggested_threshold,
            step=0.01,
            key="batch_threshold",
        )
        st.caption(f"Suggested from training distribution: {suggested_threshold:.3f}")

        if st.button("Analyze Transactions", type="primary"):
            st.session_state.analysis_runs += 1
            ensure_models_ready(df)

            # 🔧 CORRECTED PREPROCESSING AND PREDICTION PIPELINE
            with st.spinner("Processing transactions..."):
                # Step 1: Preprocess using TRANSFORM ONLY (never fit)
                X_processed, _, df_processed = preprocess_data(
                    df.copy(), 
                    scaler=st.session_state.scaler  # Use pre-loaded scaler
                )
                
                # Step 2: Get predictions from hybrid model
                predictions, hybrid_scores, individual_scores = (
                    st.session_state.hybrid_model.predict(
                        X_processed, threshold=threshold
                    )
                )

            # 🔍 DIAGNOSTIC PANEL - Add before displaying results
            with st.expander("Model Diagnostics", expanded=False):
                st.write("#### Scaled Features of Row 0:")
                st.code(X_processed[0])
                
                st.write("#### Anomaly Score Statistics:")
                diag_col1, diag_col2, diag_col3 = st.columns(3)
                
                with diag_col1:
                    st.metric("Minimum Score", f"{hybrid_scores.min():.4f}")
                
                with diag_col2:
                    st.metric("Maximum Score", f"{hybrid_scores.max():.4f}")
                
                with diag_col3:
                    st.metric("Average Score", f"{hybrid_scores.mean():.4f}")
                
                st.write("#### Individual Model Scores:")
                st.write(f"- **Isolation Forest**: Min={individual_scores['isolation_forest'].min():.4f}, Max={individual_scores['isolation_forest'].max():.4f}, Avg={individual_scores['isolation_forest'].mean():.4f}")
                st.write(f"- **AutoEncoder**: Min={individual_scores['autoencoder'].min():.4f}, Max={individual_scores['autoencoder'].max():.4f}, Avg={individual_scores['autoencoder'].mean():.4f}")
                st.write(f"- **DBSCAN**: Min={individual_scores['dbscan'].min():.4f}, Max={individual_scores['dbscan'].max():.4f}, Avg={individual_scores['dbscan'].mean():.4f}")
                
                st.write("#### Feature Statistics (After Scaling):")
                # X_processed is already a scaled numpy array with all 16 features
                scaled_df = pd.DataFrame(X_processed)
                st.dataframe(scaled_df.describe())
                
                st.write("#### Categorical Encoding Check:")
                type_cols = [col for col in df_processed.columns if col.startswith('type_')]
                if type_cols:
                    st.write(f"Type columns present: {type_cols}")
                    st.write(df_processed[type_cols].head())

            # Calculate statistics
            num_anomalies = predictions.sum()
            num_normal = len(predictions) - num_anomalies

            # Update session state for dashboard metrics
            st.session_state.total_transactions += len(df)
            st.session_state.fraudulent_count += int(num_anomalies)

            # Display metrics
            st.markdown("### Analysis Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Anomalies Detected", num_anomalies)
            with col3:
                st.metric("Normal Transactions", num_normal)

            # Visualization
            st.markdown("### Anomaly Distribution")

            viz_df = pd.DataFrame(
                {
                    "transaction_index": np.arange(len(hybrid_scores)),
                    "hybrid_score": hybrid_scores,
                    "prediction": np.where(predictions == 1, "Anomaly", "Normal"),
                    "iso_score": individual_scores["isolation_forest"],
                    "ae_score": individual_scores["autoencoder"],
                    "dbscan_score": individual_scores["dbscan"],
                }
            )
            if "amount" in df.columns:
                viz_df["amount"] = df["amount"].values
            if "type" in df.columns:
                viz_df["type"] = df["type"].astype(str).values

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                score_fig = px.scatter(
                    viz_df,
                    x="transaction_index",
                    y="hybrid_score",
                    color="prediction",
                    color_discrete_map={"Normal": "#2d9bf0", "Anomaly": "#e74c3c"},
                    title="Anomaly Scores by Transaction",
                )
                score_fig.add_hline(y=threshold, line_dash="dash", line_color="#f39c12")
                score_fig.update_layout(
                    template="plotly_white", margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(score_fig, width="stretch")

            with plot_col2:
                hist_fig = px.histogram(
                    viz_df,
                    x="hybrid_score",
                    color="prediction",
                    nbins=40,
                    barmode="overlay",
                    opacity=0.75,
                    color_discrete_map={"Normal": "#2d9bf0", "Anomaly": "#e74c3c"},
                    title="Score Distribution",
                )
                hist_fig.add_vline(x=threshold, line_dash="dash", line_color="#f39c12")
                hist_fig.update_layout(
                    template="plotly_white", margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(hist_fig, width="stretch")

            st.markdown("### Advanced Bulk Insights")
            adv_col1, adv_col2 = st.columns(2)

            with adv_col1:
                # Risk band distribution
                risk_bins = pd.cut(
                    viz_df["hybrid_score"],
                    bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
                    labels=[
                        "Low (0-0.25)",
                        "Medium (0.25-0.5)",
                        "High (0.5-0.75)",
                        "Critical (0.75-1.0)",
                    ],
                )
                risk_band_df = (
                    risk_bins.value_counts()
                    .rename_axis("risk_band")
                    .reset_index(name="count")
                )
                risk_band_fig = px.bar(
                    risk_band_df,
                    x="risk_band",
                    y="count",
                    color="risk_band",
                    color_discrete_map={
                        "Low (0-0.25)": "#17a673",
                        "Medium (0.25-0.5)": "#f39c12",
                        "High (0.5-0.75)": "#e67e22",
                        "Critical (0.75-1.0)": "#e74c3c",
                    },
                    title="Risk Band Distribution",
                )
                risk_band_fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                    xaxis_title="Risk Band",
                    yaxis_title="Transactions",
                )
                st.plotly_chart(risk_band_fig, width="stretch")

            with adv_col2:
                # Compare model component score distributions
                component_df = viz_df[["iso_score", "ae_score", "dbscan_score"]].melt(
                    var_name="component",
                    value_name="score",
                )
                component_fig = px.box(
                    component_df,
                    x="component",
                    y="score",
                    color="component",
                    color_discrete_map={
                        "iso_score": "#2d9bf0",
                        "ae_score": "#1f5f99",
                        "dbscan_score": "#e74c3c",
                    },
                    title="Component Score Distribution",
                )
                component_fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                    xaxis_title="Model Component",
                    yaxis_title="Normalized Score",
                )
                st.plotly_chart(component_fig, width="stretch")

            adv_col3, adv_col4 = st.columns(2)

            with adv_col3:
                # Amount-risk relationship
                if "amount" in viz_df.columns:
                    amount_score_fig = px.scatter(
                        viz_df,
                        x="amount",
                        y="hybrid_score",
                        color="prediction",
                        color_discrete_map={"Normal": "#2d9bf0", "Anomaly": "#e74c3c"},
                        opacity=0.7,
                        title="Amount vs Hybrid Risk Score",
                    )
                    amount_score_fig.add_hline(
                        y=threshold, line_dash="dash", line_color="#f39c12"
                    )
                    amount_score_fig.update_layout(
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_title="Transaction Amount",
                        yaxis_title="Hybrid Score",
                    )
                    st.plotly_chart(amount_score_fig, width="stretch")
                else:
                    st.info("Amount column not found for amount-risk visualization.")

            with adv_col4:
                # Transaction type anomaly intensity
                if "type" in viz_df.columns:
                    type_risk_df = (
                        viz_df.groupby("type", as_index=False)
                        .agg(
                            mean_hybrid_score=("hybrid_score", "mean"),
                            anomaly_rate=(
                                "prediction",
                                lambda x: (x == "Anomaly").mean() * 100,
                            ),
                            tx_count=("prediction", "count"),
                        )
                        .sort_values("mean_hybrid_score", ascending=False)
                    )
                    type_risk_fig = px.bar(
                        type_risk_df,
                        x="type",
                        y="mean_hybrid_score",
                        color="anomaly_rate",
                        color_continuous_scale=["#d6ebff", "#2d9bf0", "#1f5f99"],
                        title="Type-Level Risk Intensity (Mean Score)",
                        hover_data=["tx_count", "anomaly_rate"],
                    )
                    type_risk_fig.update_layout(
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=40, b=10),
                        coloraxis_showscale=False,
                        xaxis_title="Transaction Type",
                        yaxis_title="Mean Hybrid Score",
                    )
                    st.plotly_chart(type_risk_fig, width="stretch")
                else:
                    st.info("Type column not found for type-level analysis.")

            # Show anomalous transactions
            if num_anomalies > 0:
                st.markdown("### Detected Anomalies")
                anomaly_indices = np.where(predictions == 1)[0]
                df_anomalies = df.iloc[anomaly_indices].copy()
                df_anomalies["hybrid_score"] = hybrid_scores[anomaly_indices]
                df_anomalies["iso_score"] = individual_scores["isolation_forest"][
                    anomaly_indices
                ]
                df_anomalies["ae_score"] = individual_scores["autoencoder"][
                    anomaly_indices
                ]
                df_anomalies["dbscan_score"] = individual_scores["dbscan"][
                    anomaly_indices
                ]
                df_anomalies["recommended_action"] = df_anomalies["hybrid_score"].apply(
                    get_recommended_action
                )

                st.dataframe(df_anomalies)

                existing_ids = {
                    str(t.get("transaction_id", ""))
                    for t in st.session_state.flagged_transactions
                }
                for idx_row, row in df_anomalies.head(200).iterrows():
                    tx_id = row.get("nameOrig", f"batch_{int(idx_row)}")
                    if str(tx_id) in existing_ids:
                        continue
                    st.session_state.flagged_transactions.append(
                        {
                            "transaction_id": str(tx_id),
                            "customer_id": str(row.get("nameOrig", "N/A")),
                            "type": row.get("type", "N/A"),
                            "amount": float(row.get("amount", 0)),
                            "hybrid_score": float(row["hybrid_score"]),
                            "fraud_score": float(row["hybrid_score"]),
                            "risk_level": get_risk_level(float(row["hybrid_score"]))[0],
                            "timestamp": datetime.now().strftime(
                                "%m/%d/%Y, %I:%M:%S %p"
                            ),
                            "reason": "Batch analysis anomaly",
                            "recommended_action": get_recommended_action(
                                float(row["hybrid_score"])
                            ),
                        }
                    )
                    existing_ids.add(str(tx_id))

                # Download button
                csv = df_anomalies.to_csv(index=False)
                st.download_button(
                    label="Download Anomalies CSV",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


def investigation_reports():
    """Page 4: Investigation & Reports"""
    st.title("Investigation and Reports")
    st.markdown(
        "Validate flagged transactions manually and export investigation outcomes."
    )

    if len(st.session_state.flagged_transactions) == 0:
        st.info(
            "No flagged transactions yet. Run Real-Time Monitor or Batch Analysis first."
        )
        return

    st.success(
        f"Found {len(st.session_state.flagged_transactions)} flagged transactions"
    )
    df_flagged = pd.DataFrame(st.session_state.flagged_transactions).copy()

    if "transaction_id" not in df_flagged.columns:
        df_flagged["transaction_id"] = df_flagged.index.astype(str)
    if "risk_level" not in df_flagged.columns:
        df_flagged["risk_level"] = df_flagged.get("hybrid_score", 0.0).apply(
            lambda x: get_risk_level(float(x))[0]
        )

    df_flagged["manual_review"] = df_flagged["transaction_id"].apply(
        lambda x: st.session_state.manual_reviews.get(str(x), "Pending")
    )
    df_flagged["recommended_action"] = df_flagged.apply(
        lambda row: get_recommended_action(
            float(row.get("hybrid_score", row.get("fraud_score", 0.0))),
            row["manual_review"],
        ),
        axis=1,
    )

    # Analyst filter controls
    st.markdown("### Investigation Filters")
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    with filt_col1:
        review_filter = st.multiselect(
            "Review Status",
            ["Pending", "Confirmed Fraud", "False Positive"],
            default=["Pending", "Confirmed Fraud", "False Positive"],
        )
    with filt_col2:
        risk_options = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        present_risks = sorted(
            [r for r in risk_options if r in set(df_flagged["risk_level"].astype(str))]
        )
        risk_filter = st.multiselect(
            "Risk Level",
            present_risks if present_risks else risk_options,
            default=present_risks if present_risks else risk_options,
        )
    with filt_col3:
        min_score, max_score = st.slider(
            "Hybrid Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )
    with filt_col4:
        search_id = st.text_input("Search Transaction ID", value="")

    filtered_df = df_flagged[
        df_flagged["manual_review"].isin(review_filter)
        & df_flagged["risk_level"].isin(risk_filter)
        & (
            df_flagged.get("hybrid_score", 0.0)
            .astype(float)
            .between(min_score, max_score)
        )
    ].copy()
    if search_id.strip():
        filtered_df = filtered_df[
            filtered_df["transaction_id"]
            .astype(str)
            .str.contains(search_id.strip(), case=False, na=False)
        ]

    # Triage KPIs
    st.markdown("### Triage Metrics")
    review_counts = filtered_df["manual_review"].value_counts().to_dict()
    review_pending = review_counts.get("Pending", 0)
    review_confirmed = review_counts.get("Confirmed Fraud", 0)
    review_false = review_counts.get("False Positive", 0)
    avg_risk_score = (
        float(filtered_df.get("hybrid_score", pd.Series(dtype=float)).mean())
        if len(filtered_df) > 0
        else 0.0
    )
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Queue Size", len(filtered_df))
    kpi2.metric("Pending Review", review_pending)
    kpi3.metric("Confirmed Fraud", review_confirmed)
    kpi4.metric("Avg Hybrid Score", f"{avg_risk_score:.3f}")

    # Queue table
    st.markdown("### Review Queue")
    sort_choice = st.selectbox(
        "Sort Queue By",
        ["Highest Risk First", "Lowest Risk First", "Latest Added"],
        index=0,
    )
    if sort_choice == "Highest Risk First":
        filtered_df = filtered_df.sort_values("hybrid_score", ascending=False)
    elif sort_choice == "Lowest Risk First":
        filtered_df = filtered_df.sort_values("hybrid_score", ascending=True)
    else:
        if "timestamp" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("timestamp", ascending=False)

    display_cols = [
        "transaction_id",
        "type",
        "amount",
        "customer_id",
        "nameOrig",
        "hybrid_score",
        "risk_level",
        "manual_review",
        "recommended_action",
        "reason",
    ]
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    st.dataframe(filtered_df[available_cols], width="stretch")

    # Manual decision controls
    st.markdown("### Manual Validation")
    selectable_ids = [str(x) for x in filtered_df["transaction_id"].head(500).tolist()]
    if not selectable_ids:
        st.info("No transactions match current filters.")
    else:
        selected_txn = st.selectbox("Select transaction ID", selectable_ids)
        decision_col1, decision_col2, decision_col3 = st.columns(3)
        with decision_col1:
            if st.button("Confirm Fraud", type="primary"):
                st.session_state.manual_reviews[selected_txn] = "Confirmed Fraud"
                st.rerun()
        with decision_col2:
            if st.button("Mark False Positive"):
                st.session_state.manual_reviews[selected_txn] = "False Positive"
                st.rerun()
        with decision_col3:
            if st.button("Mark Pending"):
                st.session_state.manual_reviews[selected_txn] = "Pending"
                st.rerun()

    # Reporting visuals
    st.markdown("### Investigation Visual Analytics")
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        outcome_counts = (
            filtered_df["manual_review"]
            .value_counts()
            .rename_axis("manual_review")
            .reset_index(name="count")
        )
        if len(outcome_counts) > 0:
            outcome_fig = px.pie(
                outcome_counts,
                names="manual_review",
                values="count",
                hole=0.5,
                color="manual_review",
                color_discrete_map={
                    "Confirmed Fraud": "#e74c3c",
                    "False Positive": "#2d9bf0",
                    "Pending": "#f39c12",
                },
                title="Review Outcome Distribution",
            )
            outcome_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(outcome_fig, width="stretch")

    with vis_col2:
        if "type" in filtered_df.columns:
            type_df = (
                filtered_df.groupby(["type", "manual_review"])
                .size()
                .reset_index(name="count")
            )
            type_fig = px.bar(
                type_df,
                x="type",
                y="count",
                color="manual_review",
                barmode="group",
                color_discrete_map={
                    "Confirmed Fraud": "#e74c3c",
                    "False Positive": "#2d9bf0",
                    "Pending": "#f39c12",
                },
                title="Manual Review by Transaction Type",
            )
            type_fig.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(type_fig, width="stretch")

    if "amount" in filtered_df.columns and "manual_review" in filtered_df.columns:
        amount_fig = px.box(
            filtered_df,
            x="manual_review",
            y="amount",
            color="manual_review",
            color_discrete_map={
                "Confirmed Fraud": "#e74c3c",
                "False Positive": "#2d9bf0",
                "Pending": "#f39c12",
            },
            title="Amount Distribution by Review Status",
        )
        amount_fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(amount_fig, width="stretch")

    # Report downloads
    st.markdown("### Download Reports")
    report_col1, report_col2 = st.columns(2)
    with report_col1:
        filtered_csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Report (CSV)",
            data=filtered_csv,
            file_name=f"filtered_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with report_col2:
        full_csv = df_flagged.to_csv(index=False)
        st.download_button(
            label="Download Full Investigation Report (CSV)",
            data=full_csv,
            file_name=f"full_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )

    if st.button("Clear Flagged Transactions"):
        clear_investigation_queue()
        st.rerun()


def main():
    """Main application logic with navigation"""

    # Defensive initialization for reruns/import edge cases.
    initialize_session_state()

    # Check login status
    if not st.session_state.logged_in:
        login_page()
        return

    # Sidebar navigation with custom styling
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-card">
                <div class="sidebar-brand-title">🛡️ Fraud Detection System</div>
                <p class="sidebar-brand-sub">Hybrid unsupervised monitoring and investigation console.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Add Reset Dashboard Button at the top
        st.markdown("---")
        if st.button("Clear Dashboard Stats", type="secondary", use_container_width=True):
            reset_dashboard_statistics()
            st.success("✓ Dashboard statistics cleared!")
            st.rerun()
        st.markdown("---")

        selected = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "Real-Time Monitor",
                "Batch Analysis",
                "Investigation & Reports",
            ],
            icons=["speedometer2", "activity", "folder2-open", "file-earmark-text"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "4px 0 4px 0",
                    "background-color": "transparent",
                },
                "icon": {"color": "#d7ecff", "font-size": "17px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "4px 2px",
                    "padding": "11px 10px",
                    "border-radius": "10px",
                    "color": "#f3f9ff",
                    "background-color": "rgba(18, 58, 94, 0.55)",
                },
                "nav-link-selected": {
                    "background-color": "#ffffff",
                    "color": "#1f5f99",
                    "font-weight": "700",
                },
            },
        )

        if st.button("Logout", width="stretch"):
            reset_for_logout()
            st.rerun()

    # Route to selected page
    if selected == "Dashboard":
        dashboard_page()
    elif selected == "Real-Time Monitor":
        real_time_monitor()
    elif selected == "Batch Analysis":
        batch_analysis()
    elif selected == "Investigation & Reports":
        investigation_reports()


if __name__ == "__main__":
    main()
