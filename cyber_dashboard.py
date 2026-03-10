"""
🛡️ AI-Based Network Intrusion Detection System Dashboard
=====================================================

A comprehensive cybersecurity dashboard for monitoring ML/DL model performance
and real-time network intrusion detection using the NSL-KDD dataset.

Author: AI Cybersecurity Team
Last Updated: March 2026
Dataset: NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 PAGE CONFIGURATION & THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🛡️ Network IDS Dashboard", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cybersecurity theme
st.markdown("""
<style>
/* Dark cybersecurity theme */
.stApp {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
}

/* Metric cards styling */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border: 1px solid #00ff41;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(0, 255, 65, 0.2);
}

/* Headers */
.main-header {
    font-size: 2.5rem;
    color: #00ff41;
    text-align: center;
    text-shadow: 0 0 10px #00ff41;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 1.8rem;
    color: #00ccff;
    text-shadow: 0 0 8px #00ccff;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

/* Alert boxes */
.alert-success {
    background: linear-gradient(90deg, #28a745, #20c997);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 5px solid #00ff41;
}

.alert-danger {
    background: linear-gradient(90deg, #dc3545, #fd7e14);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 5px solid #ff073a;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #ff073a 30%, #ff6b35 90%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255, 7, 58, 0.4);
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: #16213e;
    color: #00ff41;
    border: 1px solid #00ccff;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DATA LOADING & DEMO SETUP  
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_results_data():
    """Load ML model results from CSV file or return demo data if not available."""
    results_file = Path("results.csv")
    
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            # Ensure required columns exist
            required_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns in results.csv: {missing_cols}. Using demo data.")
                return get_demo_data()
            return df
        except Exception as e:
            st.error(f"Error loading results.csv: {e}")
            return get_demo_data()
    else:
        st.info("📁 results.csv not found. Displaying demo results. Run the HybridDL.ipynb notebook to generate real data.")
        return get_demo_data()

def get_demo_data():
    """Generate realistic demo data for dashboard preview."""
    models_data = {
        'Model': [
            'Logistic Regression', 'Naive Bayes', 'SVM (RBF)', 
            'Decision Tree', 'Random Forest', 'CNN (Conv1D)',
            'LSTM (Stacked)', 'GRU (Stacked)', 'Hybrid CNN-LSTM',
            'Ensemble (Avg Voting)', 'Ensemble (Weighted Voting)'
        ],
        'Accuracy': [85.2, 81.7, 87.9, 89.1, 92.4, 94.2, 93.8, 93.5, 95.7, 96.1, 96.8],
        'Precision': [84.8, 80.2, 88.1, 88.7, 91.9, 93.6, 92.9, 92.8, 95.2, 95.8, 96.4],
        'Recall': [85.6, 83.1, 87.7, 89.5, 92.8, 94.8, 94.7, 94.2, 96.2, 96.4, 97.1],
        'F1-Score': [85.2, 81.6, 87.9, 89.1, 92.3, 94.2, 93.8, 93.5, 95.7, 96.1, 96.7],
        'AUC-ROC': [88.4, 85.3, 90.1, 91.2, 94.6, 96.8, 96.2, 95.9, 97.8, 98.1, 98.5]
    }
    return pd.DataFrame(models_data)

def create_model_category_mapping(df):
    """Create model category mapping for visualization."""
    categories = {}
    for model in df['Model']:
        if any(ml_model in model.lower() for ml_model in ['logistic', 'naive', 'svm', 'decision', 'random']):
            categories[model] = 'ML Baseline'
        elif any(dl_model in model.lower() for dl_model in ['cnn', 'lstm', 'gru']) and 'hybrid' not in model.lower():
            categories[model] = 'Deep Learning'
        elif any(hybrid_model in model.lower() for hybrid_model in ['hybrid', 'ensemble']):
            categories[model] = 'Hybrid DL ★'
        else:
            categories[model] = 'Other'
    return categories

# ══════════════════════════════════════════════════════════════════════════════
# 🎯 MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    results_df = load_results_data()
    model_categories = create_model_category_mapping(results_df)
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ AI-Based Network Intrusion Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["🏠 Overview", "🏆 Model Leaderboard", "📈 Metrics Visualization", "🚨 Real-time IDS Demo"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Information")
    st.sidebar.info("**Dataset:** NSL-KDD\n\n**Task:** Binary Classification\n\n**Classes:** Normal vs Attack")
    
    st.sidebar.markdown("### 🔧 Model Categories")
    st.sidebar.markdown("""
    - **🔵 ML Baseline:** Traditional ML models
    - **🟢 Deep Learning:** CNN, LSTM, GRU
    - **🟠 Hybrid DL ★:** Advanced ensemble methods
    """)
    
    # Page routing
    if page == "🏠 Overview":
        show_overview_page(results_df, model_categories)
    elif page == "🏆 Model Leaderboard":
        show_leaderboard_page(results_df, model_categories)
    elif page == "📈 Metrics Visualization":
        show_metrics_visualization(results_df, model_categories)
    elif page == "🚨 Real-time IDS Demo":
        show_realtime_demo()

# ══════════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_overview_page(df, categories):
    st.markdown('<h2 class="section-header">🏠 System Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = df.loc[df['Accuracy'].idxmax(), 'Model']
        st.metric(
            label="🏆 Best Model", 
            value=best_model[:15] + "..." if len(best_model) > 15 else best_model,
            delta="Top Performer"
        )
    
    with col2:
        highest_acc = df['Accuracy'].max()
        st.metric(
            label="📊 Highest Accuracy", 
            value=f"{highest_acc:.2f}%",
            delta=f"+{highest_acc - df['Accuracy'].mean():.1f}% vs avg"
        )
    
    with col3:
        total_models = len(df)
        st.metric(
            label="🔢 Models Trained", 
            value=str(total_models),
            delta="Complete evaluation"
        )
    
    with col4:
        dataset_name = "NSL-KDD"
        st.metric(
            label="💾 Dataset", 
            value=dataset_name,
            delta="Binary classification"
        )
    
    st.markdown("---")
    
    # Performance comparison chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Model Accuracy Comparison")
        
        # Create color mapping
        color_map = {
            'ML Baseline': '#3498db',      # Blue
            'Deep Learning': '#2ecc71',    # Green  
            'Hybrid DL ★': '#f39c12',      # Orange
            'Other': '#95a5a6'             # Gray
        }
        
        df_sorted = df.sort_values('Accuracy', ascending=True)
        colors = [color_map.get(categories.get(model, 'Other'), '#95a5a6') for model in df_sorted['Model']]
        
        fig = px.bar(
            df_sorted, 
            x='Accuracy', 
            y='Model',
            orientation='h',
            title="Model Performance (Accuracy %)",
            color_discrete_sequence=colors
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500,
            showlegend=False
        )
        
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Performance Summary")
        
        # Category performance
        category_stats = {}
        for category in ['ML Baseline', 'Deep Learning', 'Hybrid DL ★']:
            cat_models = [m for m in df['Model'] if categories.get(m) == category]
            if cat_models:
                cat_df = df[df['Model'].isin(cat_models)]
                avg_acc = cat_df['Accuracy'].mean()
                category_stats[category] = avg_acc
        
        for category, avg_acc in category_stats.items():
            color = {'ML Baseline': '🔵', 'Deep Learning': '🟢', 'Hybrid DL ★': '🟠'}[category]
            st.markdown(f"{color} **{category}:** {avg_acc:.1f}%")
        
        st.markdown("---")
        
        # Top 3 models
        st.markdown("### 🏅 Top 3 Models")
        top_3 = df.nlargest(3, 'Accuracy')
        medals = ['🥇', '🥈', '🥉']
        
        for i, (_, model_row) in enumerate(top_3.iterrows()):
            medal = medals[i] if i < 3 else '🏅'
            st.markdown(f"{medal} **{model_row['Model']}**")
            st.markdown(f"   Accuracy: {model_row['Accuracy']:.2f}%")
            st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# 🏆 MODEL LEADERBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_leaderboard_page(df, categories):
    st.markdown('<h2 class="section-header">🏆 Model Leaderboard</h2>', unsafe_allow_html=True)
    
    # Sorting options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sort_metric = st.selectbox(
            "📊 Sort by metric:",
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            index=0
        )
    
    with col2:
        st.markdown("### 🎯 Leaderboard Controls")
        show_categories = st.multiselect(
            "Filter by category:",
            list(set(categories.values())),
            default=list(set(categories.values()))
        )
    
    # Filter and sort data
    filtered_models = [m for m in df['Model'] if categories.get(m) in show_categories]
    df_filtered = df[df['Model'].isin(filtered_models)].copy()
    df_sorted = df_filtered.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    
    # Add ranking and category columns
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    df_sorted['Category'] = df_sorted['Model'].map(categories)
    
    # Display leaderboard table
    st.markdown(f"### 📋 Models ranked by {sort_metric}")
    
    # Highlight top performer
    if not df_sorted.empty:
        top_model = df_sorted.iloc[0]
        st.success(f"🏆 **Champion:** {top_model['Model']} with {sort_metric}: {top_model[sort_metric]:.2f}%")
    
    # Custom table styling
    def highlight_ranks(val):
        if val == 1:
            return 'background-color: #FFD700; color: black; font-weight: bold'  # Gold
        elif val == 2:
            return 'background-color: #C0C0C0; color: black; font-weight: bold'  # Silver
        elif val == 3:
            return 'background-color: #CD7F32; color: white; font-weight: bold'  # Bronze
        return ''
    
    # Display styled dataframe
    styled_df = df_sorted[['Rank', 'Model', 'Category', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].style.applymap(
        highlight_ranks, subset=['Rank']
    ).format({
        'Accuracy': '{:.2f}%',
        'Precision': '{:.2f}%', 
        'Recall': '{:.2f}%',
        'F1-Score': '{:.2f}%',
        'AUC-ROC': '{:.2f}%'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Performance distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Score Distribution")
        fig = px.histogram(
            df_sorted, 
            x=sort_metric,
            title=f"{sort_metric} Distribution",
            nbins=10,
            color_discrete_sequence=['#00ff41']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Category Performance")
        category_avg = df_sorted.groupby('Category')[sort_metric].mean().reset_index()
        
        fig = px.pie(
            category_avg,
            values=sort_metric,
            names='Category',
            title=f"Average {sort_metric} by Category",
            color_discrete_sequence=['#3498db', '#2ecc71', '#f39c12']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📈 METRICS VISUALIZATION PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_metrics_visualization(df, categories):
    st.markdown('<h2 class="section-header">📈 Advanced Metrics Visualization</h2>', unsafe_allow_html=True)
    
    # Metric selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_metric = st.selectbox(
            "📊 Select Metric:",
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            index=3
        )
        
        chart_type = st.radio(
            "📈 Visualization Type:",
            ['Bar Chart', 'Radar Chart', 'Heatmap', 'Box Plot']
        )
    
    with col2:
        if chart_type == 'Bar Chart':
            show_bar_chart(df, selected_metric, categories)
        elif chart_type == 'Radar Chart':
            show_radar_chart(df)
        elif chart_type == 'Heatmap':
            show_heatmap(df)
        elif chart_type == 'Box Plot':
            show_box_plot(df, categories)

def show_bar_chart(df, metric, categories):
    """Interactive bar chart with category coloring."""
    st.markdown(f"### 📊 {metric} Comparison")
    
    # Color mapping
    df_viz = df.copy()
    df_viz['Category'] = df_viz['Model'].map(categories)
    df_viz = df_viz.sort_values(metric, ascending=False)
    
    fig = px.bar(
        df_viz,
        x='Model',
        y=metric,
        color='Category',
        title=f"Model Performance: {metric}",
        color_discrete_map={
            'ML Baseline': '#3498db',
            'Deep Learning': '#2ecc71', 
            'Hybrid DL ★': '#f39c12',
            'Other': '#95a5a6'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45,
        height=500
    )
    
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def show_radar_chart(df):
    """Multi-model radar chart comparison."""
    st.markdown("### 🎯 Multi-Metric Radar Comparison")
    
    # Select top 5 models for cleaner visualization
    df_top5 = df.nlargest(5, 'F1-Score')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    
    colors = ['#ff6b35', '#f7931e', '#ffcc02', '#9bc53d', '#5aa3c4']
    
    for i, (_, model) in enumerate(df_top5.iterrows()):
        values = [model[metric] for metric in metrics]
        values += [values[0]]  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model['Model'][:20],
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='white'
            ),
            angularaxis=dict(color='white')
        ),
        showlegend=True,
        title="Top 5 Models - Multi-Metric Comparison",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_heatmap(df):
    """Correlation heatmap of metrics."""
    st.markdown("### 🔥 Metrics Correlation Heatmap")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    corr_matrix = df[metrics].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Metrics Correlation Matrix",
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_box_plot(df, categories):
    """Box plot showing metric distribution by category.""" 
    st.markdown("### 📦 Performance Distribution by Category")
    
    df_viz = df.copy()
    df_viz['Category'] = df_viz['Model'].map(categories)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        horizontal_spacing=0.05
    )
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    for i, metric in enumerate(metrics):
        for j, category in enumerate(['ML Baseline', 'Deep Learning', 'Hybrid DL ★']):
            cat_data = df_viz[df_viz['Category'] == category][metric]
            if not cat_data.empty:
                fig.add_trace(
                    go.Box(
                        y=cat_data,
                        name=category,
                        showlegend=(i == 0),
                        marker_color=colors[j],
                        boxmean=True
                    ),
                    row=1, col=i+1
                )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_text="Performance Distribution by Model Category"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🚨 REAL-TIME IDS DEMO
# ══════════════════════════════════════════════════════════════════════════════

def show_realtime_demo():
    st.markdown('<h2 class="section-header">🚨 Real-time Network Intrusion Detection Demo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔍 Network Traffic Analyzer
    Simulate real-time network traffic analysis using trained ML models.
    Adjust the network parameters below and analyze for potential threats.
    """)
    
    # Traffic parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Network Traffic Parameters")
        
        duration = st.slider(
            "⏱️ Connection Duration (seconds):",
            min_value=0.0,
            max_value=300.0,
            value=15.5,
            step=0.1,
            help="Duration of network connection"
        )
        
        src_bytes = st.slider(
            "📤 Source Bytes Sent:",
            min_value=0,
            max_value=100000,
            value=2048,
            step=100,
            help="Number of bytes sent from source"
        )
        
        dst_bytes = st.slider(
            "📥 Destination Bytes Received:",
            min_value=0,
            max_value=100000,
            value=1024,
            step=100,
            help="Number of bytes received by destination"
        )
        
        # Additional parameters
        protocol = st.selectbox(
            "🌐 Protocol Type:",
            ['TCP', 'UDP', 'ICMP'],
            help="Network protocol used"
        )
        
        service = st.selectbox(
            "🔧 Network Service:",
            ['HTTP', 'FTP', 'SSH', 'SMTP', 'DNS', 'Telnet'],
            help="Network service being accessed"
        )
    
    with col2:
        st.markdown("#### 🎯 Analysis Results")
        
        # Analysis button
        if st.button("🔍 ANALYZE TRAFFIC", type="primary"):
            
            # Show analysis progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate analysis process
            for i in range(1, 101, 10):
                time.sleep(0.05)
                progress_bar.progress(i)
                if i <= 30:
                    status_text.text("🔍 Preprocessing traffic data...")
                elif i <= 60:
                    status_text.text("🧠 Running ML models...")
                elif i <= 90:
                    status_text.text("📊 Analyzing results...")
                else:
                    status_text.text("✅ Analysis complete!")
            
            # Simulate prediction based on parameters
            risk_score = calculate_risk_score(duration, src_bytes, dst_bytes, protocol, service)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if risk_score > 0.7:
                st.markdown("""
                <div class="alert-danger">
                    <h4>🚨 THREAT DETECTED!</h4>
                    <p><strong>Risk Level:</strong> HIGH ({:.1f}%)</p>
                    <p><strong>Classification:</strong> Potential Network Attack</p>
                    <p><strong>Recommended Action:</strong> Block connection and investigate</p>
                </div>
                """.format(risk_score * 100), unsafe_allow_html=True)
                
                # Show threat details
                st.error("⚠️ Suspicious Activity Detected")
                
                threat_metrics = {
                    "Anomaly Score": f"{risk_score:.3f}",
                    "Threat Type": random.choice(["DDoS", "Port Scan", "Data Exfiltration", "Brute Force"]),
                    "Confidence": f"{random.uniform(85, 98):.1f}%",
                    "Response Time": f"{random.uniform(0.05, 0.2):.3f}s"
                }
                
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                
                for i, (metric, value) in enumerate(threat_metrics.items()):
                    with cols[i]:
                        st.metric(label=metric, value=value)
                
            else:
                st.markdown("""
                <div class="alert-success">
                    <h4>✅ TRAFFIC NORMAL</h4>
                    <p><strong>Risk Level:</strong> LOW ({:.1f}%)</p>
                    <p><strong>Classification:</strong> Legitimate Network Traffic</p>
                    <p><strong>Status:</strong> Connection allowed</p>
                </div>
                """.format(risk_score * 100), unsafe_allow_html=True)
                
                st.success("🟢 Network traffic appears legitimate")
                
                normal_metrics = {
                    "Safety Score": f"{1 - risk_score:.3f}",
                    "Traffic Type": "Normal",
                    "Confidence": f"{random.uniform(90, 99):.1f}%",
                    "Response Time": f"{random.uniform(0.01, 0.08):.3f}s"
                }
                
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                
                for i, (metric, value) in enumerate(normal_metrics.items()):
                    with cols[i]:
                        st.metric(label=metric, value=value)
    
    # Model insights section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Active Models")
        models_status = [
            ("🟢 Hybrid CNN-LSTM", "Active", "97.8%"),
            ("🟢 Ensemble Weighted", "Active", "96.7%"),
            ("🟡 Random Forest", "Standby", "92.3%"),
            ("🟢 Deep LSTM", "Active", "93.8%"),
        ]
        
        for model, status, accuracy in models_status:
            st.markdown(f"**{model}**")
            st.markdown(f"Status: {status} | Accuracy: {accuracy}")
            st.markdown("")
    
    with col2:
        st.markdown("### 📊 Real-time Statistics")
        
        # Generate realistic stats
        current_time = time.strftime("%H:%M:%S")
        daily_scans = random.randint(1200, 2500)
        threats_blocked = random.randint(15, 45)
        
        stats = {
            "⏰ Current Time": current_time,
            "🔍 Daily Scans": f"{daily_scans:,}",
            "🛡️ Threats Blocked": f"{threats_blocked}",
            "📈 Success Rate": f"{random.uniform(97, 99.5):.1f}%"
        }
        
        for stat, value in stats.items():
            st.markdown(f"**{stat}:** {value}")

def calculate_risk_score(duration, src_bytes, dst_bytes, protocol, service):
    """Calculate risk score based on traffic parameters."""
    risk = 0.0
    
    # Duration-based risk
    if duration > 200:
        risk += 0.3
    elif duration < 1:
        risk += 0.2
    
    # Bytes-based risk
    if src_bytes > 50000 or dst_bytes > 50000:
        risk += 0.4
    
    # Unusual byte ratios
    if src_bytes > 0 and dst_bytes > 0:
        ratio = max(src_bytes, dst_bytes) / min(src_bytes, dst_bytes)
        if ratio > 100:
            risk += 0.3
    
    # Protocol and service risk
    high_risk_combinations = [
        ('TCP', 'Telnet'),
        ('UDP', 'DNS'),
        ('TCP', 'FTP')
    ]
    
    if (protocol, service) in high_risk_combinations:
        risk += 0.2
    
    # Add random variation to simulate real detection
    risk += random.uniform(-0.1, 0.1)
    
    return max(0.0, min(1.0, risk))

# ══════════════════════════════════════════════════════════════════════════════
# 🚀 APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #00ccff; margin-top: 2rem;">
        <p>🛡️ <strong>AI-Based Network Intrusion Detection System</strong></p>
        <p>Powered by Machine Learning & Deep Learning | NSL-KDD Dataset</p>
        <p>Real-time cybersecurity monitoring and threat detection</p>
    </div>
    """, unsafe_allow_html=True)