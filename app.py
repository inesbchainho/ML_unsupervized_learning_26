import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score

# Page configuration
st.set_page_config(
    page_title="Customer Clustering Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme from Main_Notebook
COLOR_PRIMARY = '#3ecae3'
COLOR_PALETTE = ['#e2fbff', '#b6f4ff', '#6ae8ff', '#3ecae3', '#1ab5d1', '#0097b2', '#007b91']

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #3ecae3;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #0097b2;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">💳 Customer Clustering & Segmentation Analysis</div>', unsafe_allow_html=True)

# Sidebar for file upload and parameters
st.sidebar.markdown("### 📁 Data & Parameters")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Data preprocessing
    st.sidebar.markdown("### ⚙️ Preprocessing Options")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.sidebar.warning(f"⚠️ Found {df.isnull().sum().sum()} missing values")
        fill_method = st.sidebar.radio("How to handle missing values?",
                                       ["Drop rows", "Fill with median", "Fill with mean"],
                                       index=1
        )
        if fill_method == "Drop rows":
            df = df.dropna()
        elif fill_method == "Fill with median":
            df = df.fillna(df.median(numeric_only=True))
        else:
            df = df.fillna(df.mean(numeric_only=True))
    
    # Select numeric columns for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove ID column if present
    numeric_cols = [col for col in numeric_cols if col.upper() not in ['CUST_ID', 'ID']]
    
    selected_features = st.sidebar.multiselect(
        "Select features for clustering",
        numeric_cols,
        default=numeric_cols
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering")
    else:
        # Prepare data for clustering
        X = df[selected_features].copy()
        X = X.fillna(X.median(numeric_only=True))
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering parameters
        st.sidebar.markdown("### 🎯 Clustering Parameters")
        n_clusters = st.sidebar.slider(
            "Number of clusters (k)",
            min_value=2,
            max_value=10,
            value=3
        )
        
        random_state = st.sidebar.number_input(
            "Random state",
            value=42
        )
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, clusters)
        
        # Add cluster labels to dataframe
        df['Cluster'] = clusters
        if n_clusters == 3:
            cluster_names = {
                0: "Low to Moderate Activity Customers",
                1: "High Value Active Purchasers",
                2: "High Balance / Potential Risk Customers"
            }
            df["Cluster_Name"] = df["Cluster"].map(cluster_names)
        else:
            df["Cluster_Name"] = df["Cluster"].apply(lambda x: f"Cluster {x}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Dataset Overview", "🎨 Cluster Visualization", "📈 Statistics", "🔍 Cluster Details"]
        )
        
        # Tab 1: Dataset Overview
        with tab1:
            st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features Used", len(selected_features))
            with col3:
                st.metric("Number of Clusters", n_clusters)
            with col4:
                st.metric("Inertia (WCSS)", round(kmeans.inertia_, 2))
            with col5:
                st.metric("Silhouette Score", round(sil_score, 3))
    
            
            st.markdown("### Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### Descriptive Statistics")
            st.dataframe(df[selected_features].describe(), use_container_width=True)
        
        # Tab 2: Cluster Visualization
        with tab2:
            st.markdown('<div class="sub-header">Cluster Visualization</div>', unsafe_allow_html=True)
            
            # 2D Scatter plot (PCA for visualization)
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create 2D scatter plot
            fig_2d = go.Figure()
            
            for cluster in range(n_clusters):
                mask = clusters == cluster
                fig_2d.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=8,
                        color=COLOR_PALETTE[cluster % len(COLOR_PALETTE)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    )
                ))
            
            # Add centroids
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            fig_2d.add_trace(go.Scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode='markers+text',
                name='Centroids',
                marker=dict(size=15, color='black', symbol='x'),
                text=[f'C{i}' for i in range(n_clusters)],
                textposition='top center'
            ))
            
            fig_2d.update_layout(
                title="Cluster Distribution (PCA Projection)",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                hovermode='closest',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Cluster distribution pie chart
            st.markdown("### Cluster Distribution")
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f'Cluster {i}' for i in cluster_counts.index],
                values=cluster_counts.values,
                marker=dict(colors=[COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in cluster_counts.index]),
                textposition='inside',
                textinfo='label+percent'
            )])
            
            fig_pie.update_layout(
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tab 3: Statistics
        with tab3:
            st.markdown('<div class="sub-header">Cluster Statistics</div>', unsafe_allow_html=True)
            
            # Cluster sizes
            st.markdown("### Cluster Sizes")
            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            fig_sizes = go.Figure(data=[
                go.Bar(
                    x=[f'Cluster {i}' for i in cluster_sizes.index],
                    y=cluster_sizes.values,
                    marker_color=[COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in cluster_sizes.index]
                )
            ])
            fig_sizes.update_layout(
                title="Number of Customers per Cluster",
                xaxis_title="Cluster",
                yaxis_title="Count",
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
            
            # Feature means by cluster
            st.markdown("### Average Feature Values by Cluster")
            cluster_stats = df.groupby('Cluster')[selected_features].mean()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=cluster_stats.T.values,
                x=[f'Cluster {i}' for i in cluster_stats.index],
                y=selected_features,
                colorscale='Blues',
                colorbar=dict(title="Mean Value")
            ))
            fig_heatmap.update_layout(
                title="Feature Heatmap by Cluster",
                height=max(400, len(selected_features) * 25),
                template='plotly_white'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Statistical summary table
            st.dataframe(cluster_stats.round(2), use_container_width=True)
        
        # Tab 4: Cluster Details
        with tab4:
            st.markdown('<div class="sub-header">Detailed Cluster Information</div>', unsafe_allow_html=True)
            
            selected_cluster = st.selectbox(
                "Select a cluster to explore",
                range(n_clusters),
                format_func=lambda x: f"Cluster {x} ({(clusters == x).sum()} customers)"
            )
            
            cluster_data = df[df['Cluster'] == selected_cluster]

            st.markdown("### Cluster Interpretation")

            if selected_cluster == 0 and n_clusters == 3:
                st.info("""
                This cluster represents low to moderate activity customers. 
                These customers have relatively low or moderate balances, purchases, credit limits and payments.
                """)
            elif selected_cluster == 1 and n_clusters == 3:
                st.success("""
                This cluster represents high value active purchasers. 
                These customers have the highest purchase amounts, high purchase frequency, many transactions and high payments.
                """)
            elif selected_cluster == 2 and n_clusters == 3:
                st.warning("""
                This cluster represents high balance and potential risk customers. 
                These customers have high balances, very high minimum payments and a very low full payment rate, which may suggest a potential risk profile.
                """)
            else:
                st.info("This cluster can be interpreted by comparing its average feature values with the other clusters.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Customers in this cluster", len(cluster_data))
            with col2:
                st.metric("% of Total", f"{len(cluster_data)/len(df)*100:.1f}%")
            with col3:
                st.metric("Features Average", f"{cluster_data[selected_features].mean().mean():.2f}")
            
            st.markdown("### Cluster Sample Data")
            st.dataframe(cluster_data.head(20), use_container_width=True)
            
            st.markdown("### Download Cluster Data")
            csv = cluster_data.to_csv(index=False)
            st.download_button(
                label=f"Download Cluster {selected_cluster} as CSV",
                data=csv,
                file_name=f"cluster_{selected_cluster}.csv",
                mime="text/csv"
            )

else:
    # Instructions when no file is uploaded
    st.info("👋 Welcome! To get started:")
    st.markdown("""
    1. **Upload a CSV file** using the file uploader in the sidebar
    2. **Select features** for clustering analysis
    3. **Adjust parameters** like the number of clusters
    4. **Explore the results** using the different tabs:
       - 📊 Dataset Overview: View your data statistics
       - 🎨 Cluster Visualization: See clusters in 2D space
       - 📈 Statistics: Analyze cluster characteristics
       - 🔍 Cluster Details: Dive deeper into individual clusters
    """)
    
    st.markdown("### Sample Dataset")
    st.markdown("""
    You can use the **CC GENERAL.csv** file from your project folder.
    This dataset contains credit card customer information with features like:
    - BALANCE
    - PURCHASES
    - CASH_ADVANCE
    - PAYMENTS
    - And more...
    """)
