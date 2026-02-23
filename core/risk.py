import streamlit as st


def get_risk_level(score):
    if score >= 0.75:
        return "CRITICAL", "critical"
    if score >= 0.6:
        return "HIGH", "high"
    if score >= 0.4:
        return "MEDIUM", "medium"
    return "LOW", "low"


def get_recommended_action(score: float, review_status: str = "Pending") -> str:
    if review_status == "Confirmed Fraud":
        return "Freeze account, block transaction, and escalate to fraud team."
    if review_status == "False Positive":
        return "Release transaction and tune threshold/feature checks."
    if score >= 0.75:
        return "Block immediately and require customer verification."
    if score >= 0.6:
        return "Hold transaction for analyst review before release."
    if score >= 0.4:
        return "Allow with monitoring and place customer on watchlist."
    return "Allow transaction and continue normal monitoring."


def calculate_objective_status():
    uploaded = len(st.session_state.uploaded_datasets) > 0
    analyzed = st.session_state.analysis_runs > 0
    flagged = len(st.session_state.flagged_transactions) > 0
    reviewed = len(st.session_state.manual_reviews) > 0
    recommended = flagged
    return {
        "1. Upload mobile money dataset for analysis": uploaded,
        "2. Analyze transactions against normal behavior": analyzed,
        "3. Flag fraudulent activities": flagged,
        "4. Validate flagged transactions via manual inspection": reviewed,
        "5. Recommend actions after manual analysis": recommended,
    }
