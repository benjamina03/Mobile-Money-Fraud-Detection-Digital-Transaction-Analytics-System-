# Quick Start Guide - Updated Dashboard

## 🚀 Running Your Application

### Start the App
```powershell
streamlit run app.py
```

### Login Credentials
- **Username:** `admin`
- **Password:** `admin123`

## 📊 Using the New Dashboard

### 1. Dashboard (Home Page)
When you first login, you'll see the dashboard with:
- **Metrics Cards**: All starting at 0 until you process transactions
- **Empty Charts**: Will populate once you run analysis
- **No Alerts**: Clean state

**To populate the dashboard**, proceed to step 2 or 3.

### 2. Real-Time Monitor
This is the best way to see the dashboard come alive!

1. Click "**Real-Time Monitor**" in sidebar
2. Upload your `test_data.csv` file
3. Set number of transactions (e.g., 20)
4. Set block threshold (e.g., 0.75)
5. Click "**▶ Start Simulation**"

**What happens:**
- Transactions process one by one
- Fraudulent ones appear in dashboard alerts
- Metrics update in real-time
- Charts populate automatically
- Recent transactions show in dashboard

### 3. Batch Analysis
For bulk processing:

1. Click "**Batch Analysis**" in sidebar
2. Upload CSV file
3. Set anomaly threshold
4. Click "**🔍 Analyze Transactions**"

**What happens:**
- All transactions analyzed at once
- Fraudulent transactions added to dashboard
- Charts and metrics update
- Can download anomaly report

### 4. View Results on Dashboard
Navigate back to "**Dashboard**":

**You'll now see:**
- ✅ Updated metrics with real numbers
- 📊 Transaction volume chart with actual data
- 📊 Risk distribution pie chart
- 🚨 Active fraud alerts with:
  - Fraud scores (color-coded bars)
  - Risk levels (badges)
  - **✅ Resolve button** - Click to mark as resolved
  - **🔍 Investigate button** - Click to see full details
- 📝 Recent transactions grid (last 10)

### 5. Manage Alerts

**Resolve an Alert:**
1. Find the alert in the table
2. Click the **✅ button**
3. Alert disappears from active list
4. Active Alerts count decreases

**Investigate an Alert:**
1. Click the **🔍 button**
2. Expandable panel shows full JSON data
3. Review all transaction details

### 6. Investigation & Reports
View all flagged transactions and download reports:

1. Click "**Investigation & Reports**"
2. See complete list of flagged transactions
3. Download CSV reports
4. Clear flagged transactions if needed

## 🎨 Theme Features

### Light White & Blue Design
- Clean, professional appearance
- Easy on the eyes
- Blue accents (#3498db) throughout
- White cards with subtle shadows

### Color Coding
- 🟢 **GREEN**: Clean transactions, low risk
- 🟡 **YELLOW**: Medium risk
- 🟠 **ORANGE**: High risk  
- 🔴 **RED**: Critical risk, fraud detected

### Interactive Elements
- Buttons change color on hover
- Cards lift on hover
- Smooth transitions
- Responsive layout

## 💡 Tips

1. **Start with Real-Time Monitor** to see live updates
2. **Use smaller batches** (10-20 transactions) for demo
3. **Threshold 0.75** catches most fraud without false positives
4. **Resolve alerts** to keep dashboard clean
5. **Charts update automatically** - no need to refresh

## 🔧 Troubleshooting

**Models not loaded?**
- Run `generate_sample_data.py` first to create trained models
- Check if `trained_models/` folder exists

**No data showing?**
- Process some transactions first
- Upload test_data.csv in Real-Time Monitor

**Dashboard empty?**
- This is normal on first login
- Run Real-Time Monitor or Batch Analysis

## 📁 Files You Need
- `app.py` - Main application (updated)
- `test_data.csv` - Test transactions
- `trained_models/` - Pre-trained models (from generate_sample_data.py)
- `preprocessing.py` - Feature engineering
- `models.py` - ML models

## 🎯 Key Differences from Before

| Feature | Before | Now |
|---------|--------|-----|
| Currency | Naira (₦) | USD ($) |
| Theme | Dark/Default | Light White & Blue |
| Buttons | Non-functional HTML | Fully Functional |
| Data | Sample/Static | Real-time from Model |
| Alerts | Can't resolve | Clickable Resolve/Investigate |
| Charts | Sample data | Your actual transaction data |
| Metrics | Hardcoded | Dynamic, updates live |

Enjoy your professional fraud detection dashboard! 🛡️
