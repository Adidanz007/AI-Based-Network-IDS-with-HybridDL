# 🛡️ AI Network IDS Dashboard - Setup Guide

## 📋 Overview

This guide provides complete instructions for setting up and running the AI-Based Network Intrusion Detection Dashboard that connects your Jupyter notebook training results with a real-time Streamlit dashboard.

## 🔧 Prerequisites

- Python 3.8+ installed
- Git (for cloning if needed)
- Web browser for dashboard access

## 📦 Installation

### Step 1: Install Required Dependencies

Run this single command to install all necessary packages:

```bash
pip install streamlit plotly pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter
```

### Step 2: Verify Installation

```bash
# Check Streamlit installation
streamlit --version

# Check Python version  
python --version
```

## 🚀 Running the Project

### Phase 1: Train the Models (Generate Results)

1. **Open the training notebook:**
   ```bash
   jupyter notebook HybridDL.ipynb
   ```

2. **Execute all cells in order** (this will take some time):
   - The notebook trains 11 different ML/DL models
   - Each model's performance is automatically stored
   - At the end, `results.csv` is generated with all metrics

3. **Verify the export:**
   - Check that `results.csv` appears in your project folder
   - File should contain columns: Model, Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Phase 2: Launch the Dashboard

1. **Start the Streamlit dashboard:**
   ```bash
   streamlit run cyber_dashboard.py
   ```

2. **Access the dashboard:**
   - Automatically opens in your browser at: `http://localhost:8501`
   - If it doesn't open automatically, navigate to that URL manually

## 🎯 Dashboard Features

### 🏠 Overview Page
- **System metrics** showing best model and performance summary
- **Interactive bar chart** comparing all model accuracies
- **Top 3 leaderboard** with medal rankings
- **Dataset information** and training statistics

### 🏆 Model Leaderboard  
- **Sortable table** by any metric (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- **Category filtering** (ML Baseline, Deep Learning, Hybrid DL)
- **Performance distribution** charts and statistics
- **Gold/Silver/Bronze** highlighting for top performers

### 📈 Metrics Visualization
- **Interactive charts**: Bar charts, radar plots, heatmaps, box plots
- **Metric selection** to focus on specific performance aspects
- **Multi-model comparison** using radar charts for top 5 models
- **Correlation analysis** showing relationships between metrics

### 🚨 Real-time IDS Demo
- **Traffic parameter sliders**:
  - Connection duration (0-300 seconds)
  - Source bytes sent (0-100K)
  - Destination bytes received (0-100K)
  - Protocol type (TCP/UDP/ICMP)  
  - Network service (HTTP/FTP/SSH/SMTP/DNS/Telnet)
- **AI-powered threat analysis** with risk scoring
- **Visual alert system** for threats vs normal traffic
- **Real-time statistics** and monitoring metrics

## 🎨 Design Features

- **Dark cybersecurity theme** with neon green/blue accents
- **Responsive layout** that works on desktop and mobile
- **Interactive visualizations** using Plotly charts
- **Professional UI** with metric cards and progress indicators
- **Real-time updates** when new results.csv is generated

## 🔄 Workflow Summary

```
1. HybridDL.ipynb → Train Models → results.csv
2. cyber_dashboard.py → Load results.csv → Interactive Dashboard
3. Use dashboard to analyze model performance and demo IDS
```

## 🛠️ Troubleshooting

### "results.csv not found"
- **Solution**: Run the HybridDL.ipynb notebook completely
- **Alternative**: Dashboard shows demo data if CSV is missing
- **Check**: Ensure the CSV export cell executed successfully

### "Module not found" errors
- **Solution**: Install missing packages:
  ```bash
  pip install [missing-package-name]
  ```
- **Common packages**: streamlit, plotly, pandas, numpy

### Dashboard won't start
- **Check port**: Ensure port 8501 isn't already in use
- **Alternative port**: `streamlit run cyber_dashboard.py --server.port 8502`
- **Firewall**: Make sure firewall allows local connections

### Slow loading
- **Cause**: Large dataset processing
- **Solution**: Dashboard caches data automatically after first load
- **Tip**: Demo mode loads instantly if no CSV file exists

## 📊 Expected Results

After running the complete workflow, you should see:

### In the Dashboard:
- 11 trained models with performance metrics
- Best performing model typically: Ensemble (Weighted Voting) or Hybrid CNN-LSTM
- Accuracy range: 85-97% across different model types
- Interactive charts showing model comparisons
- Working real-time intrusion detection simulation

### Model Categories:
- **🔵 ML Baseline** (5 models): Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest
- **🟢 Deep Learning** (3 models): CNN Conv1D, LSTM Stacked, GRU Stacked  
- **🟠 Hybrid DL ★** (3 models): Hybrid CNN-LSTM, Ensemble Average, Ensemble Weighted

## 🔍 File Structure

After setup, your project should look like:

```
AI-Based-Network-IDS_ML-DL-main/
├── HybridDL.ipynb           # Training notebook
├── cyber_dashboard.py       # Streamlit dashboard
├── results.csv              # Generated model results  
├── DASHBOARD_SETUP.md       # This guide
├── README.md                # Original project info
└── nsl-kdd/                 # Dataset folder
    ├── KDDTrain+.txt
    └── KDDTest+.txt
```

## 🎯 Next Steps

1. **Explore the dashboard**: Navigate through all four main pages
2. **Experiment with parameters**: Try different settings in the IDS demo
3. **Analyze results**: Compare model performance across different metrics
4. **Customize**: Modify the dashboard code to add new features
5. **Deploy**: Consider deploying to Streamlit Cloud for sharing

## 🤝 Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Verify installation** of all required packages
3. **Ensure CSV generation** by re-running notebook cells
4. **Try demo mode** to test dashboard functionality

## ✅ Success Checklist

- [ ] All packages installed successfully
- [ ] HybridDL.ipynb notebook runs completely  
- [ ] results.csv file generated with model data
- [ ] Dashboard starts without errors
- [ ] All four dashboard pages load and function
- [ ] Real-time IDS demo responds to parameter changes
- [ ] Charts and visualizations display correctly

---

🛡️ **Your AI-powered cybersecurity dashboard is ready to monitor network threats!**