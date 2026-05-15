import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Page configuration
st.set_page_config(
    page_title="Customer Clustering Analysis",
    page_icon="📊",
    layout="wide"
)

# Visual identity
PALETTE = ['#3ECAE3', '#416788', '#D0B8AC', '#F3D8C7', '#3F6634']
BACKGROUND_COLOR = '#0E1117'
CARD_COLOR = '#12171F'
TEXT_COLOR = '#E5E9F0'
HEADER_COLOR = '#3ECAE3'
ACCENT_COLOR = '#D0B8AC'

st.markdown(f"""
<style>
    body {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR}; }}
    .stApp {{ background-color: {BACKGROUND_COLOR}; }}
    .main {{ background-color: {BACKGROUND_COLOR}; }}
    .block-container {{ padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{ color: {HEADER_COLOR} !important; }}
    .stMetricValue, .stMetricLabel {{ color: {TEXT_COLOR} !important; }}
    div.stButton > button {{ background-color: {ACCENT_COLOR} !important; color: {BACKGROUND_COLOR} !important; }}
    .css-1d391kg, .css-1n76uvr, .css-1lcbmhc {{ background: {CARD_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<h1 style='color:{HEADER_COLOR}; margin-bottom:0.2rem;'>💳 Customer Clustering & Segmentation</h1>", unsafe_allow_html=True)

st.markdown("---")

try:
    df = pd.read_csv("data_droped.csv")
except FileNotFoundError:
    st.error("Could not find `data_droped.csv` in the repository root. Please place the file next to `app.py`.")
    st.stop()

all_features = df.select_dtypes(include=[np.number]).columns.tolist()
all_features = [col for col in all_features if col.upper() not in ['CUST_ID', 'ID', 'CUSTOMER_ID']]
if len(all_features) < 2:
    st.error("The dataset must contain at least two numeric features for clustering.")
    st.stop()

X = df[all_features].copy()
X = X.fillna(X.median(numeric_only=True))

card_col1, card_col2, card_col3 = st.columns([1, 1.8, 1])
with card_col1:
    st.markdown(
        f"<div style='background:{CARD_COLOR}; padding:1rem 1rem 0.9rem 1rem; border-radius:18px; min-height:130px;'>"
        f"<h3 style='color:{HEADER_COLOR}; margin-bottom:0.45rem;'>DATA</h3>"
        f"<p style='color:{TEXT_COLOR}; margin:0; font-size:0.98rem;'>data_droped.csv</p>"
        f"</div>",
        unsafe_allow_html=True
    )
with card_col2:
    st.markdown(
        f"<div class='cluster-card' style='background:{CARD_COLOR}; padding:1rem 1rem 0.9rem 1rem; border-radius:18px; min-height:170px;'>"
        f"<h3 style='color:{HEADER_COLOR}; margin-bottom:0.45rem;'>Number of Clusters (k)</h3>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<style>"
        f".cluster-card + div > div:first-child {{margin-top:-1.25rem; padding:0.4rem 0.75rem 0.65rem 0.75rem; background:{CARD_COLOR}; border-radius:0 0 18px 18px;}}"
        f".cluster-card + div .stSlider {{margin:0 !important; padding:0 !important;}}"
        f".cluster-card + div .stSlider label {{margin-bottom:0.15rem !important;}}"
        f".cluster-card + div .stSlider > div > div {{padding:0.25rem 0.25rem 0.25rem 0.25rem !important;}}"
        f"</style>",
        unsafe_allow_html=True
    )
    n_clusters = st.slider(
        "",
        min_value=2,
        max_value=30,
        value=3,
        help="Select how many clusters the K-Means model should produce."
    )
with card_col3:
    st.markdown(
        f"<div style='background:{CARD_COLOR}; padding:1rem 1rem 0.9rem 1rem; border-radius:18px; min-height:130px;'>"
        f"<h3 style='color:{HEADER_COLOR}; margin-bottom:0.45rem;'>RANDOM STATE</h3>"
        f"<p style='color:{TEXT_COLOR}; margin:0; font-size:0.98rem;'>42</p>"
        f"</div>",
        unsafe_allow_html=True
    )

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, clusters)

df['Cluster'] = clusters
if n_clusters == 3:
    cluster_names = {
        0: "Low to Moderate Activity Customers",
        1: "High Value Active Purchasers",
        2: "High Balance / Potential Risk Customers"
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
else:
    df['Cluster_Name'] = df['Cluster'].apply(lambda x: f"Cluster {x}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🎨 Cluster Visualization",
    "📈 Statistics",
    "🔍 Cluster Details"
])

with tab1:
    st.markdown(f"<h2 style='color:{HEADER_COLOR};'>Dataset Information</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", len(df))
    col2.metric("Features Used", len(all_features))
    col3.metric("Number of Clusters", n_clusters)
    col4.metric("Inertia (WCSS)", round(kmeans.inertia_, 2))
    col5.metric("Silhouette Score", round(sil_score, 3))

    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Descriptive Statistics")
    st.dataframe(df[all_features].describe(), use_container_width=True)

with tab2:
    st.markdown(f"<h2 style='color:{HEADER_COLOR};'>Cluster Visualization</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_COLOR};'>This view uses <strong>data_droped.csv</strong>, all numeric variables, and a fixed random state of <strong>42</strong>. Only the number of clusters <strong>k</strong> is adjustable.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<ul style='color:{TEXT_COLOR};'>"
        f"<li><strong>Dataset:</strong> data_droped.csv</li>"
        f"<li><strong>Features:</strong> all numeric variables</li>"
        f"<li><strong>Random state:</strong> 42</li>"
        f"<li><strong>Adjustable:</strong> number of clusters (k) only</li>"
        f"</ul>",
        unsafe_allow_html=True
    )

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

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
                color=PALETTE[cluster % len(PALETTE)],
                opacity=0.85,
                line=dict(width=1, color=BACKGROUND_COLOR)
            )
        ))

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    fig_2d.add_trace(go.Scatter(
        x=centroids_pca[:, 0],
        y=centroids_pca[:, 1],
        mode='markers+text',
        name='Centroids',
        marker=dict(size=16, color=TEXT_COLOR, symbol='x'),
        text=[f'C{i}' for i in range(n_clusters)],
        textposition='top center',
        textfont=dict(color=TEXT_COLOR)
    ))

    fig_2d.update_layout(
        title='Cluster Distribution (PCA Projection)',
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
        hovermode='closest',
        height=620,
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    fig_2d.update_xaxes(showgrid=True, gridcolor='#22303b')
    fig_2d.update_yaxes(showgrid=True, gridcolor='#22303b')

    st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown("### Cluster Distribution")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    fig_pie = go.Figure(data=[go.Pie(
        labels=[f'Cluster {i}' for i in cluster_counts.index],
        values=cluster_counts.values,
        marker=dict(colors=[PALETTE[i % len(PALETTE)] for i in cluster_counts.index]),
        textposition='inside',
        textinfo='label+percent',
        hole=0.3
    )])
    fig_pie.update_layout(
        height=520,
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.markdown(f"<h2 style='color:{HEADER_COLOR};'>Cluster Statistics</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_COLOR};'>Updated color scheme applied to this page.</p>", unsafe_allow_html=True)

    st.markdown("### Cluster Sizes")
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    fig_sizes = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_sizes.index],
            y=cluster_sizes.values,
            marker_color=[PALETTE[i % len(PALETTE)] for i in cluster_sizes.index]
        )
    ])
    fig_sizes.update_layout(
        title='Number of Customers per Cluster',
        xaxis_title='Cluster',
        yaxis_title='Count',
        height=420,
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        showlegend=False
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

    st.markdown("### Average Feature Values by Cluster")
    cluster_stats = df.groupby('Cluster')[all_features].mean()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=cluster_stats.T.values,
        x=[f'Cluster {i}' for i in cluster_stats.index],
        y=all_features,
        colorscale=[
            [0.0, PALETTE[1]],
            [0.5, PALETTE[2]],
            [1.0, PALETTE[3]]
        ],
        colorbar=dict(title='Mean Value', tickfont=dict(color=TEXT_COLOR), titlefont=dict(color=TEXT_COLOR))
    ))
    fig_heatmap.update_layout(
        title='Feature Heatmap by Cluster',
        height=max(420, len(all_features) * 28),
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    fig_heatmap.update_xaxes(showgrid=False)
    fig_heatmap.update_yaxes(showgrid=False)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.dataframe(cluster_stats.round(2), use_container_width=True)

with tab4:
    st.markdown(f"<h2 style='color:{HEADER_COLOR};'>Detailed Cluster Information</h2>", unsafe_allow_html=True)
    selected_cluster = st.selectbox(
        "Select a cluster to explore",
        range(n_clusters),
        format_func=lambda x: f"Cluster {x} ({(clusters == x).sum()} customers)"
    )
    cluster_data = df[df['Cluster'] == selected_cluster]

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers in this cluster", len(cluster_data))
    col2.metric("% of Total", f"{len(cluster_data) / len(df) * 100:.1f}%")
    col3.metric("Average Feature Value", f"{cluster_data[all_features].mean().mean():.2f}")

    st.markdown("### Cluster Sample Data")
    st.dataframe(cluster_data.head(20), use_container_width=True)

    st.markdown("### Download Cluster Data")
    csv = cluster_data.to_csv(index=False)
    st.download_button(
        label=f"Download Cluster {selected_cluster} as CSV",
        data=csv,
        file_name=f"cluster_{selected_cluster}.csv",
        mime='text/csv'
    )
