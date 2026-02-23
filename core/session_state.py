import streamlit as st


def initialize_session_state():
    defaults = {
        "logged_in": False,
        "flagged_transactions": [],
        "recent_transactions": [],
        "anomaly_scores_history": [],
        "models_loaded": False,
        "hybrid_model": None,
        "scaler": None,
        "total_transactions": 0,
        "fraudulent_count": 0,
        "active_alerts": 0,
        "transaction_history": [],
        "dashboard_data_loaded": False,
        "resolved_alerts": set(),
        "manual_reviews": {},
        "analysis_runs": 0,
        "uploaded_datasets": set(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_for_logout():
    """Reset session values when user logs out."""
    st.session_state.logged_in = False
    st.session_state.flagged_transactions = []
    st.session_state.recent_transactions = []
    st.session_state.anomaly_scores_history = []
    st.session_state.manual_reviews = {}
    st.session_state.analysis_runs = 0
    st.session_state.uploaded_datasets = set()
    st.session_state.dashboard_data_loaded = False
    st.session_state.total_transactions = 0
    st.session_state.fraudulent_count = 0
    st.session_state.active_alerts = 0
    st.session_state.transaction_history = []
    st.session_state.resolved_alerts = set()


def clear_investigation_queue():
    """Clear investigation-specific data."""
    st.session_state.flagged_transactions = []
    st.session_state.manual_reviews = {}
    st.session_state.resolved_alerts = set()


def reset_dashboard_statistics():
    """Reset all dashboard metrics and data stores."""
    # Clear statistical counters
    st.session_state.total_transactions = 0
    st.session_state.fraudulent_count = 0
    st.session_state.active_alerts = 0
    
    # Clear data stores
    st.session_state.flagged_transactions = []
    st.session_state.recent_transactions = []
    st.session_state.anomaly_scores_history = []
    st.session_state.transaction_history = []
    
    # Clear analysis tracking
    st.session_state.analysis_runs = 0
    st.session_state.uploaded_datasets = set()
    
    # Clear investigation data
    st.session_state.manual_reviews = {}
    st.session_state.resolved_alerts = set()
    
    # Reset dashboard loaded flag
    st.session_state.dashboard_data_loaded = False
