"""
Streamlit Application for Mobile Money Fraud Detection
Dashboard with Real-Time Monitoring, Fraud Alerts, and Analytics
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import random

# Import custom modules
from preprocessing import preprocess_data, preprocess_single_transaction, engineer_features, select_features
from models import HybridModel, save_models, load_models
import joblib


# Page configuration
st.set_page_config(
    page_title="Mobile Money Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - Light White and Blue Theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container - Light theme */
    .main {
        background-color: #f0f4f8;
    }
    
    /* Metric cards - White with blue accents */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px solid #3498db;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.1);
    }
    
    div[data-testid="metric-container"] > label {
        color: #2c3e50;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] > div {
        font-size: 32px;
        font-weight: 700;
        color: #2980b9;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
        background-color: white;
    }
    
    /* Buttons - Blue theme */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #3498db;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
        margin-bottom: 20px;
        border: 1px solid #e3f2fd;
    }
    
    /* Transaction cards */
    .transaction-card {
        background: white;
        border: 1px solid #e3f2fd;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.08);
        transition: all 0.2s ease;
    }
    
    .transaction-card:hover {
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
        transform: translateY(-2px);
    }
    
    /* Status badges */
    .badge-clean {
        background-color: #27ae60;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-fraud {
        background-color: #e74c3c;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-critical {
        background-color: #e74c3c;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-high {
        background-color: #e67e22;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-medium {
        background-color: #f39c12;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-low {
        background-color: #27ae60;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 18px;
        margin-top: 35px;
        border-bottom: 3px solid #3498db;
        padding-bottom: 8px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'flagged_transactions' not in st.session_state:
    st.session_state.flagged_transactions = []
if 'recent_transactions' not in st.session_state:
    st.session_state.recent_transactions = []
if 'anomaly_scores_history' not in st.session_state:
    st.session_state.anomaly_scores_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'total_transactions' not in st.session_state:
    st.session_state.total_transactions = 0
if 'fraudulent_count' not in st.session_state:
    st.session_state.fraudulent_count = 0
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = 0
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'dashboard_data_loaded' not in st.session_state:
    st.session_state.dashboard_data_loaded = False
if 'resolved_alerts' not in st.session_state:
    st.session_state.resolved_alerts = set()


def get_risk_level(score):
    """Determine risk level based on fraud score"""
    if score >= 0.75:
        return "CRITICAL", "critical"
    elif score >= 0.6:
        return "HIGH", "high"
    elif score >= 0.4:
        return "MEDIUM", "medium"
    else:
        return "LOW", "low"


def create_fraud_score_bar(score):
    """Create a visual fraud score bar"""
    # Color gradient from green to red
    if score < 0.3:
        color = "#27ae60"
    elif score < 0.5:
        color = "#f39c12"
    else:
        color = "#e74c3c"
    
    return f"""
    <div style="background-color: #ecf0f1; border-radius: 10px; height: 22px; width: 100%; position: relative; border: 1px solid #bdc3c7;">
        <div style="background-color: {color}; border-radius: 10px; height: 22px; width: {score*100}%; position: absolute; top: 0; left: 0;"></div>
        <span style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: 700; color: #2c3e50;">{score:.3f}</span>
    </div>
    """


def generate_sample_transactions(num_transactions=10):
    """Generate sample transactions for display"""
    transactions = []
    base_time = datetime.now()
    
    for i in range(num_transactions):
        txn_id = f"TXN{2000 + i}"
        customer_id = f"CUST{300 + i}"
        amount = round(random.uniform(100, 5000), 2)
        is_fraud = random.random() < 0.3
        timestamp = base_time - timedelta(minutes=random.randint(1, 180))
        
        transactions.append({
            'transaction_id': txn_id,
            'customer_id': customer_id,
            'amount': amount,
            'is_fraud': is_fraud,
            'timestamp': timestamp.strftime('%m/%d/%Y, %I:%M:%S %p')
        })
    
    return transactions


def dashboard_page():
    """Main Dashboard Page - Matches the screenshot layout"""
    load_custom_css()
    
    st.title("Fraud Detection Dashboard")
    
    # Top Metrics Row (4 cards)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fraud_rate = (st.session_state.fraudulent_count / st.session_state.total_transactions) * 100
        st.metric(
            label="TOTAL TRANSACTIONS",
            value=f"{st.session_state.total_transactions:,}",
            delta="+12.5% from last period",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="FRAUDULENT TRANSACTIONS",
            value=f"{st.session_state.fraudulent_count}",
            delta="+8.3% from last period",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="FRAUD RATE",
            value=f"{fraud_rate:.2f}%",
            delta="+0.2% from last period",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="ACTIVE ALERTS",
            value=f"{st.session_state.active_alerts}",
            delta="No change"
        )
    
    # Charts Row
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### Transaction Volume Over Time")
        
        # Generate sample time series data
        dates = pd.date_range(end=datetime.now(), periods=20, freq='H')
        volumes = [random.randint(500, 3000) for _ in range(20)]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, volumes, marker='o', color='#0d6efd', linewidth=2, markersize=6)
        ax.fill_between(dates, volumes, alpha=0.1, color='#0d6efd')
        ax.set_xlabel('')
        ax.set_ylabel('Amount', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    
    with chart_col2:
        st.markdown("### Risk Level Distribution")
        
        # Pie chart data
        risk_levels = ['Low 20%', 'Critical 20%', 'High 40%', 'Medium 20%']
        sizes = [20, 20, 40, 20]
        colors = ['#28a745', '#dc3545', '#fd7e14', '#ffc107']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=risk_levels, 
            colors=colors,
            autopct='',
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    
    # Recent Fraud Alerts Section
    st.markdown("<div class='section-header'>Recent Fraud Alerts</div>", unsafe_allow_html=True)
    
    # Create sample fraud alerts if none exist
    if len(st.session_state.flagged_transactions) == 0:
        sample_alerts = [
            {'transaction_id': '', 'customer_id': '', 'amount': 2081.64, 'fraud_score': 0.875, 'risk_level': 'CRITICAL', 'timestamp': '12/2/2025, 1:33:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 3477.58, 'fraud_score': 0.353, 'risk_level': 'MEDIUM', 'timestamp': '12/2/2025, 2:16:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 773.19, 'fraud_score': 0.708, 'risk_level': 'LOW', 'timestamp': '12/2/2025, 2:39:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 746.77, 'fraud_score': 0.355, 'risk_level': 'HIGH', 'timestamp': '12/2/2025, 2:15:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 3733.93, 'fraud_score': 0.217, 'risk_level': 'MEDIUM', 'timestamp': '12/2/2025, 1:16:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 2925.76, 'fraud_score': 0.801, 'risk_level': 'MEDIUM', 'timestamp': '12/2/2025, 1:57:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 3900.41, 'fraud_score': 0.385, 'risk_level': 'MEDIUM', 'timestamp': '12/2/2025, 2:12:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 1858.27, 'fraud_score': 0.219, 'risk_level': 'CRITICAL', 'timestamp': '12/2/2025, 2:09:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 859.55, 'fraud_score': 0.910, 'risk_level': 'MEDIUM', 'timestamp': '12/2/2025, 2:38:57 PM'},
            {'transaction_id': '', 'customer_id': '', 'amount': 4921.01, 'fraud_score': 0.254, 'risk_level': 'HIGH', 'timestamp': '12/2/2025, 1:52:57 PM'},
        ]
        fraud_alerts_df = pd.DataFrame(sample_alerts)
    else:
        fraud_alerts_df = pd.DataFrame(st.session_state.flagged_transactions[:10])
    
    # Create refresh button
    col_refresh1, col_refresh2 = st.columns([6, 1])
    with col_refresh2:
        if st.button("🔄 Refresh", key="refresh_alerts"):
            st.rerun()
    
    # Display fraud alerts table
    if len(fraud_alerts_df) > 0:
        # Create HTML table
        st.markdown("""
        <style>
        .fraud-table {
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .fraud-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .fraud-table th {
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-size: 12px;
            font-weight: 600;
            color: #6c757d;
            text-transform: uppercase;
            border-bottom: 1px solid #dee2e6;
        }
        .fraud-table td {
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }
        .fraud-table tr:hover {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        table_html = '<div class="fraud-table"><table><thead><tr>'
        table_html += '<th>TRANSACTION ID</th><th>CUSTOMER ID</th><th>AMOUNT</th><th>FRAUD SCORE</th><th>RISK LEVEL</th><th>TIME</th><th>ACTION</th>'
        table_html += '</tr></thead><tbody>'
        
        for idx, row in fraud_alerts_df.iterrows():
            risk_level, risk_class = get_risk_level(row.get('fraud_score', 0.5))
            amount_display = f"₦{row.get('amount', 0):,.2f}"
            
            table_html += '<tr>'
            table_html += f'<td>{row.get("transaction_id", "")}</td>'
            table_html += f'<td>{row.get("customer_id", "")}</td>'
            table_html += f'<td><strong>{amount_display}</strong></td>'
            table_html += f'<td>{create_fraud_score_bar(row.get("fraud_score", 0.5))}</td>'
            table_html += f'<td><span class="badge-{risk_class}">{risk_level}</span></td>'
            table_html += f'<td>{row.get("timestamp", "")}</td>'
            table_html += f'<td><button style="background:#28a745; color:white; border:none; padding:6px 12px; border-radius:4px; margin-right:5px; cursor:pointer; font-size:12px; font-weight:600;">Resolve</button><button style="background:#17a2b8; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px; font-weight:600;">Investigate</button></td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table></div>'
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Recent Transactions Section
    st.markdown("<div class='section-header'>Recent Transactions</div>", unsafe_allow_html=True)
    
    # Generate sample transactions if none exist
    if len(st.session_state.recent_transactions) == 0:
        st.session_state.recent_transactions = generate_sample_transactions(10)
    
    # Display transactions in grid (2 columns)
    transactions = st.session_state.recent_transactions[:10]
    
    for i in range(0, len(transactions), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(transactions):
                txn = transactions[i + j]
                with col:
                    status_badge = "FRAUD" if txn['is_fraud'] else "CLEAN"
                    badge_class = "badge-fraud" if txn['is_fraud'] else "badge-clean"
                    
                    st.markdown(f"""
                    <div class="transaction-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="color: #6c757d; font-size: 13px; font-weight: 500;">{txn['transaction_id']}</span>
                            <span class="{badge_class}">{status_badge}</span>
                        </div>
                        <div style="font-size: 24px; font-weight: 700; color: #212529; margin-bottom: 8px;">
                            ₦{txn['amount']:,.2f}
                        </div>
                        <div style="color: #6c757d; font-size: 13px; margin-bottom: 4px;">
                            Customer: {txn['customer_id']}
                        </div>
                        <div style="color: #6c757d; font-size: 13px;">
                            {txn['timestamp']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def login_page():
    """Page 1: Login Authentication"""
    st.title("Mobile Money Fraud Detection System")
    st.subheader("Login")
    
    # Create a centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please enter your credentials")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            # Hardcoded credentials for prototype
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        
        st.info("**Demo Credentials:**\n\nUsername: `admin`\n\nPassword: `admin123`")


def load_or_train_models(X_train=None):
    """Helper function to load or train models"""
    model_path = 'trained_models'
    
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'scaler.pkl')):
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
    if X_train is None:
        raise ValueError("X_train must be provided when training new models")
    
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
    
    return hybrid_model, None  # Return None for scaler as it's passed separately


def real_time_monitor():
    """Page 2: Real-Time Transaction Monitor"""
    st.title("Real-Time Transaction Monitor")
    st.markdown("Monitor transactions as they arrive and detect fraud in real-time.")
    
    # File uploader for test dataset
    uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=['csv'], key="realtime_upload")
    
    if uploaded_file is not None:
        # Load and prepare data
        df_test = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df_test)} transactions for simulation")
        
        # Prepare models if not already loaded
        if not st.session_state.models_loaded:
            with st.spinner("Preparing models..."):
                # Load scaler explicitly
                model_path = 'trained_models'
                scaler_path = os.path.join(model_path, 'scaler.pkl')
                
                if os.path.exists(scaler_path):
                    st.info("Loading saved scaler...")
                    scaler = joblib.load(scaler_path)
                    st.session_state.scaler = scaler
                    st.success("✓ Scaler loaded successfully!")
                    
                    # Load pre-trained models
                    hybrid_model, _ = load_or_train_models(None)
                    st.session_state.hybrid_model = hybrid_model
                else:
                    # Use a sample for training if models don't exist
                    st.info("Training new scaler and models...")
                    df_train = df_test.sample(min(10000, len(df_test)), random_state=42)
                    X_train, scaler, _ = preprocess_data(df_train.copy())
                    st.session_state.scaler = scaler
                    
                    # Train new models
                    hybrid_model, _ = load_or_train_models(X_train)
                    st.session_state.hybrid_model = hybrid_model
                
                st.session_state.models_loaded = True
        
        # Simulation controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            num_transactions = st.number_input("Transactions to simulate", min_value=1, max_value=100, value=20)
        
        with col2:
            threshold = st.slider("Block Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        
        # Check if scaler is available before simulation
        if st.session_state.scaler is None:
            st.error("Scaler not loaded! Please ensure models are loaded first or run 'generate_sample_data.py' to train models and generate the scaler.")
            st.stop()
        
        # Start simulation button
        if st.button("▶ Start Simulation", type="primary"):
            st.session_state.anomaly_scores_history = []
            st.session_state.flagged_transactions = []
            
            # Create placeholders for live updates
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Sample transactions
            transactions_to_process = df_test.sample(n=min(num_transactions, len(df_test)))
            
            total_processed = 0
            total_blocked = 0
            
            for idx, (_, transaction) in enumerate(transactions_to_process.iterrows()):
                # Step 1: Feature engineering and selection
                transaction_df = pd.DataFrame([transaction])
                df_featured = engineer_features(transaction_df.copy())
                df_selected = select_features(df_featured)
                
                # Step 2: Extract raw feature row
                X_row = df_selected.values
                
                # Step 3: Apply scaler transform to get scaled features
                X_scaled = st.session_state.scaler.transform(X_row)
                
                # Step 4: Get predictions using scaled data
                predictions, hybrid_scores, individual_scores = st.session_state.hybrid_model.predict(
                    X_scaled, 
                    threshold=threshold
                )
                
                total_processed += 1
                hybrid_score = hybrid_scores[0]
                is_blocked = predictions[0] == 1
                
                if is_blocked:
                    total_blocked += 1
                
                # Store anomaly score
                st.session_state.anomaly_scores_history.append({
                    'transaction_id': idx,
                    'score': hybrid_score,
                    'blocked': is_blocked
                })
                
                # Display status
                with status_placeholder.container():
                    if is_blocked:
                        st.error(f"**TRANSACTION BLOCKED** - Hybrid Score: {hybrid_score:.3f}")
                        st.markdown(f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        - Isolation Forest Score: {individual_scores['isolation_forest'][0]:.3f}
                        - Autoencoder Score: {individual_scores['autoencoder'][0]:.3f}
                        - DBSCAN Score: {individual_scores['dbscan'][0]:.3f}
                        """)
                        
                        # Store flagged transaction
                        flagged_data = transaction.to_dict()
                        flagged_data['hybrid_score'] = hybrid_score
                        flagged_data['fraud_score'] = hybrid_score
                        flagged_data['risk_level'] = get_risk_level(hybrid_score)[0]
                        flagged_data['timestamp'] = datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')
                        flagged_data['reason'] = f"High hybrid score ({hybrid_score:.3f})"
                        if individual_scores['autoencoder'][0] > 0.7:
                            flagged_data['reason'] += ", High reconstruction error"
                        if individual_scores['isolation_forest'][0] > 0.7:
                            flagged_data['reason'] += ", Isolation Forest anomaly"
                        st.session_state.flagged_transactions.append(flagged_data)
                        
                        # Update active alerts count
                        st.session_state.active_alerts = len(st.session_state.flagged_transactions)
                        st.session_state.fraudulent_count += 1
                    else:
                        st.success(f"**TRANSACTION APPROVED** - Hybrid Score: {hybrid_score:.3f}")
                        st.markdown(f"""
                        **Transaction Details:**
                        - Type: {transaction.get('type', 'N/A')}
                        - Amount: ${transaction.get('amount', 0):,.2f}
                        """)
                    
                    # Add to recent transactions
                    recent_txn = {
                        'transaction_id': f"TXN{2000 + idx}",
                        'customer_id': transaction.get('nameOrig', f'CUST{300 + idx}')[:10],
                        'amount': transaction.get('amount', 0),
                        'is_fraud': is_blocked,
                        'timestamp': datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')
                    }
                    st.session_state.recent_transactions.insert(0, recent_txn)
                    st.session_state.recent_transactions = st.session_state.recent_transactions[:20]
                    st.session_state.total_transactions += 1
                
                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Processed", total_processed)
                    m2.metric("Blocked", total_blocked)
                    m3.metric("Approved", total_processed - total_blocked)
                    m4.metric("Block Rate", f"{(total_blocked/total_processed*100):.1f}%")
                
                # Update chart
                if len(st.session_state.anomaly_scores_history) > 0:
                    with chart_placeholder.container():
                        scores_df = pd.DataFrame(st.session_state.anomaly_scores_history)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        colors = ['red' if x else 'green' for x in scores_df['blocked']]
                        ax.scatter(scores_df['transaction_id'], scores_df['score'], c=colors, alpha=0.6)
                        ax.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
                        ax.set_xlabel('Transaction Number')
                        ax.set_ylabel('Hybrid Anomaly Score')
                        ax.set_title('Real-Time Anomaly Scores')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                        plt.clf()
                
                # Simulate delay
                time.sleep(1)
            
            st.success(f"Simulation complete! Processed {total_processed} transactions, blocked {total_blocked}.")


def batch_analysis():
    """Page 3: Batch Analysis"""
    st.title("Batch Analysis")
    st.markdown("Upload a CSV file for bulk fraud detection analysis.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], key="batch_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} transactions")
        
        # Show preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
        
        threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key="batch_threshold")
        
        if st.button("🔍 Analyze Transactions", type="primary"):
            # Prepare models if needed
            if not st.session_state.models_loaded:
                with st.spinner("Training models..."):
                    df_train = df.sample(min(10000, len(df)), random_state=42)
                    X_train, scaler, _ = preprocess_data(df_train.copy())
                    
                    hybrid_model, _ = load_or_train_models(X_train)
                    st.session_state.hybrid_model = hybrid_model
                    st.session_state.scaler = scaler
                    st.session_state.models_loaded = True
            
            # Process all transactions
            with st.spinner("Processing transactions..."):
                X_processed, _, df_processed = preprocess_data(df.copy(), scaler=st.session_state.scaler)
                predictions, hybrid_scores, individual_scores = st.session_state.hybrid_model.predict(
                    X_processed, 
                    threshold=threshold
                )
            
            # Calculate statistics
            num_anomalies = predictions.sum()
            num_normal = len(predictions) - num_anomalies
            
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            colors = ['red' if p == 1 else 'blue' for p in predictions]
            ax1.scatter(range(len(hybrid_scores)), hybrid_scores, c=colors, alpha=0.5, s=20)
            ax1.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
            ax1.set_xlabel('Transaction Index')
            ax1.set_ylabel('Hybrid Anomaly Score')
            ax1.set_title('Anomaly Scores: Normal (Blue) vs Anomalies (Red)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histogram
            ax2.hist(hybrid_scores[predictions == 0], bins=50, alpha=0.7, label='Normal', color='blue')
            ax2.hist(hybrid_scores[predictions == 1], bins=50, alpha=0.7, label='Anomalies', color='red')
            ax2.axvline(x=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
            ax2.set_xlabel('Hybrid Anomaly Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
            # Show anomalous transactions
            if num_anomalies > 0:
                st.markdown("### Detected Anomalies")
                anomaly_indices = np.where(predictions == 1)[0]
                df_anomalies = df.iloc[anomaly_indices].copy()
                df_anomalies['hybrid_score'] = hybrid_scores[anomaly_indices]
                df_anomalies['iso_score'] = individual_scores['isolation_forest'][anomaly_indices]
                df_anomalies['ae_score'] = individual_scores['autoencoder'][anomaly_indices]
                df_anomalies['dbscan_score'] = individual_scores['dbscan'][anomaly_indices]
                
                st.dataframe(df_anomalies)
                
                # Download button
                csv = df_anomalies.to_csv(index=False)
                st.download_button(
                    label="Download Anomalies CSV",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def investigation_reports():
    """Page 4: Investigation & Reports"""
    st.title("Investigation & Reports")
    st.markdown("View flagged transactions and download reports.")
    
    if len(st.session_state.flagged_transactions) == 0:
        st.info("No flagged transactions yet. Run the Real-Time Monitor or Batch Analysis first.")
    else:
        st.success(f"Found {len(st.session_state.flagged_transactions)} flagged transactions")
        
        # Convert to DataFrame
        df_flagged = pd.DataFrame(st.session_state.flagged_transactions)
        
        # Display table
        st.markdown("### Flagged Transactions")
        
        # Select columns to display
        display_cols = ['type', 'amount', 'nameOrig', 'nameDest', 'hybrid_score', 'reason']
        available_cols = [col for col in display_cols if col in df_flagged.columns]
        
        st.dataframe(df_flagged[available_cols], use_container_width=True)
        
        # Statistics
        st.markdown("### Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Flagged", len(df_flagged))
        
        with col2:
            if 'amount' in df_flagged.columns:
                avg_amount = df_flagged['amount'].mean()
                st.metric("Avg Amount", f"${avg_amount:,.2f}")
        
        with col3:
            if 'hybrid_score' in df_flagged.columns:
                avg_score = df_flagged['hybrid_score'].mean()
                st.metric("Avg Score", f"{avg_score:.3f}")
        
        # Visualization
        if 'type' in df_flagged.columns:
            st.markdown("### Transaction Type Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            type_counts = df_flagged['type'].value_counts()
            type_counts.plot(kind='bar', ax=ax, color='coral')
            ax.set_xlabel('Transaction Type')
            ax.set_ylabel('Count')
            ax.set_title('Flagged Transactions by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        
        # Download report
        st.markdown("### Download Report")
        csv = df_flagged.to_csv(index=False)
        st.download_button(
            label="Download Full Report (CSV)",
            data=csv,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Clear button
        if st.button("Clear Flagged Transactions"):
            st.session_state.flagged_transactions = []
            st.rerun()


def main():
    """Main application logic with navigation"""
    
    # Check login status
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛡️ Fraud Detection System")
        st.markdown("---")
        
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Real-Time Monitor", "Batch Analysis", "Investigation & Reports"],
            icons=["speedometer2", "activity", "folder", "search"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("### User Info")
        st.info("👤 **User:** admin")
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.flagged_transactions = []
            st.session_state.recent_transactions = []
            st.session_state.anomaly_scores_history = []
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
