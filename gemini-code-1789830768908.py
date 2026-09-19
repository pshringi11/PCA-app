import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="ISLR Ch 12.2: Principal Component Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ISLR Chapter 12.2: Principal Component Analysis (PCA)")
st.markdown("*An Introduction to Statistical Learning with Applications in Python (2023)* - **Ch 12.2 Interactive Guide**")

# Sidebar - Mode Selection
st.sidebar.header("Navigation")
mode = st.sidebar.radio(
    "Choose Module:",
    [
        "1. Interactive 2D PCA & Rotation",
        "2. Feature Scaling & Standardization",
        "3. Scree Plot & PVE Analysis",
        "4. Mathematical Formulas & Concepts",
        "5. Self-Assessment Quiz"
    ]
)

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# MODULE 1: INTERACTIVE 2D PCA & ROTATION
# ==========================================
if mode == "1. Interactive 2D PCA & Rotation":
    st.header("1. Interactive Projection & Vector Rotation")
    st.markdown("""
    PCA finds a direction (loading vector $\phi_1 = (\phi_{11}, \phi_{21})^T$) that **maximizes variance** 
    of projected points or equivalently **minimizes squared errors** (reconstruction error).
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        num_points = st.slider("Number of Points", 20, 200, 50, step=10)
        corr = st.slider("Correlation Strength", -0.99, 0.99, 0.80, step=0.05)
        angle_deg = st.slider("Rotation Angle (Degrees)", 0, 180, 30, step=1)
        
        # Generate correlated bivariate normal data
        mean = [0, 0]
        cov = [[1, corr], [corr, 1]]
        X_raw = np.random.multivariate_normal(mean, cov, num_points)
        X_centered = X_raw - np.mean(X_raw, axis=0)
        
        # Calculate angle vector
        theta = np.radians(angle_deg)
        v = np.array([np.cos(theta), np.sin(theta)]) # Loading vector
        
        # Projections
        scores = X_centered @ v
        projections = np.outer(scores, v)
        variance_proj = np.var(scores, ddof=1)
        
        # Reconstruction errors
        reconstruction_errors = np.sum((X_centered - projections)**2) / num_points

        # Calculate actual PCA First Component for comparison
        pca = PCA(n_components=1)
        pca.fit(X_centered)
        pca_angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])) % 180

        st.metric("Projected Variance", f"{variance_proj:.4f}")
        st.metric("Mean Reconstruction Error (MSE)", f"{reconstruction_errors:.4f}")
        st.info(f"💡 **Optimal PCA Angle:** `{pca_angle:.1f}°` (Angle that maximizes variance)")

    with col2:
        fig = go.Figure()

        # Original Points
        fig.add_trace(go.Scatter(
            x=X_centered[:, 0], y=X_centered[:, 1],
            mode='markers', name='Data Points',
            marker=dict(color='#3b82f6', size=8, opacity=0.7)
        ))

        # Projected Points
        fig.add_trace(go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers', name='Projections',
            marker=dict(color='#ef4444', size=6)
        ))

        # Connecting orthogonal lines (Errors)
        for i in range(num_points):
            fig.add_trace(go.Scatter(
                x=[X_centered[i, 0], projections[i, 0]],
                y=[X_centered[i, 1], projections[i, 1]],
                mode='lines', showlegend=False,
                line=dict(color='gray', width=1, dash='dot')
            ))

        # Direction vector line
        line_range = np.array([-3, 3])
        fig.add_trace(go.Scatter(
            x=line_range * v[0], y=line_range * v[1],
            mode='lines', name=f'Vector Direction ({angle_deg}°)',
            line=dict(color='#10b981', width=3)
        ))

        fig.update_layout(
            title="Orthogonal Projections onto Direction Vector",
            xaxis_title="X1 (Centered)", yaxis_title="X2 (Centered)",
            xaxis=dict(range=[-3.5, 3.5]), yaxis=dict(range=[-3.5, 3.5]),
            height=550, showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 2: FEATURE SCALING & STANDARDIZATION
# ==========================================
elif mode == "2. Feature Scaling & Standardization":
    st.header("2. Impact of Variable Scaling (ISLR Section 12.2.4)")
    st.markdown("""
    **Crucial ISLR Takeaway:** Unstandardized variables with large variance will disproportionately dominate 
    the first principal component loadings. Standardizing variables ($s_x = 1$) is usually recommended.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dataset Settings")
        scale_x1 = st.slider("Scale Multiplier for Feature X1", 1.0, 50.0, 10.0, step=1.0)
        noise = st.slider("Noise Level", 0.1, 5.0, 1.0, step=0.1)
        standardize = st.checkbox("Standardize Variables (Zero Mean & Unit Variance)", value=False)

        # Generate data with high scale difference
        x1 = np.random.normal(0, scale_x1, 100)
        x2 = 0.5 * x1 / scale_x1 + np.random.normal(0, noise, 100)
        
        df = pd.DataFrame({'X1': x1, 'X2': x2})

        if standardize:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(df)
            title_prefix = "Standardized Data"
        else:
            X_proc = df.values - df.mean().values
            title_prefix = "Unstandardized (Raw) Data"

        pca = PCA()
        pca.fit(X_proc)

        st.subheader("Results")
        st.write("**Loading Vectors ($\phi_1$ & $\phi_2$):**")
        loadings_df = pd.DataFrame(
            pca.components_.T, 
            columns=['PC1', 'PC2'], 
            index=['X1', 'X2']
        )
        st.dataframe(loadings_df.style.format("{:.4f}"))

        pve = pca.explained_variance_ratio_
        st.write(f"**PC1 PVE:** `{pve[0]*100:.2f}%`")
        st.write(f"**PC2 PVE:** `{pve[1]*100:.2f}%`")

    with col2:
        fig = px.scatter(
            x=X_proc[:, 0], y=X_proc[:, 1],
            labels={'x': 'X1 ' + ('(Scaled)' if standardize else '(Raw)'),
                    'y': 'X2 ' + ('(Scaled)' if standardize else '(Raw)')},
            title=f"{title_prefix} with PC Axes"
        )

        # Add PC direction vectors
        origin = [0, 0]
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
            length = np.sqrt(var) * 2
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=comp[0] * length, y1=comp[1] * length,
                line=dict(color="Red" if i == 0 else "Green", width=3)
            )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 3: SCREE PLOT & PVE ANALYSIS
# ==========================================
elif mode == "3. Scree Plot & PVE Analysis":
    st.header("3. Proportion of Variance Explained (PVE) & Scree Plot")
    st.markdown("""
    The **Scree Plot** shows the proportion of total variance captured by each principal component:
    $$PVE_m = \\frac{\\sum_{i=1}^n z_{im}^2}{\\sum_{j=1}^p \\sum_{i=1}^n x_{ij}^2}$$
    """)

    n_features = st.slider("Select Number of Synthetic Features (p)", 3, 15, 8)
    n_samples = st.slider("Select Sample Size (n)", 50, 500, 200)

    # Generate synthetic multi-feature data
    X_synthetic = np.random.randn(n_samples, n_features)
    # Add correlated structure
    for j in range(1, n_features):
        X_synthetic[:, j] += X_synthetic[:, 0] * (n_features - j) / n_features

    X_std = StandardScaler().fit_transform(X_synthetic)
    pca_multi = PCA()
    pca_multi.fit(X_std)

    pve = pca_multi.explained_variance_ratio_
    cumulative_pve = np.cumsum(pve)

    col1, col2 = st.columns(2)

    with col1:
        # Scree plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(range(1, n_features + 1)), y=pve,
            mode='lines+markers', name='PVE',
            marker=dict(size=10, color='#3b82f6')
        ))
        fig1.update_layout(
            title="Scree Plot (PVE per Component)",
            xaxis_title="Principal Component",
            yaxis_title="Proportion of Variance Explained",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Cumulative Scree Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(1, n_features + 1)), y=cumulative_pve,
            mode='lines+markers', name='Cumulative PVE',
            marker=dict(size=10, color='#10b981')
        ))
        fig2.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% Threshold")
        fig2.update_layout(
            title="Cumulative Scree Plot",
            xaxis_title="Number of Components",
            yaxis_title="Cumulative PVE",
            yaxis=dict(range=[0, 1.05])
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# MODULE 4: MATHEMATICAL FORMULAS & CONCEPTS
# ==========================================
elif mode == "4. Mathematical Formulas & Concepts":
    st.header("4. Core Mathematical Concepts (ISLR Chapter 12.2)")

    st.subheader("1. First Principal Component Optimization")
    st.latex(r"""
    \max_{\phi_{11},\dots,\phi_{p1}} \left\{ \frac{1}{n}\sum_{i=1}^n \left( \sum_{j=1}^p \phi_{j1} x_{ij} \right)^2 \right\} 
    \quad \text{subject to} \quad \sum_{j=1}^p \phi_{j1}^2 = 1
    """)

    st.subheader("2. Principal Component Scores")
    st.latex(r"""
    z_{i1} = \phi_{11} x_{i1} + \phi_{21} x_{i2} + \dots + \phi_{p1} x_{ip}
    """)

    st.subheader("3. Equivalence of Max Variance and Min Error")
    st.latex(r"""
    \sum_{j=1}^p \sum_{i=1}^n x_{ij}^2 = \sum_{i=1}^n z_{i1}^2 + \text{Reconstruction Error}
    """)
    st.markdown("""
    Since the total variance $\sum\sum x_{ij}^2$ is constant, **maximizing the component score variance $\sum z_{i1}^2$ is mathematically identical to minimizing the perpendicular projection errors.**
    """)

# ==========================================
# MODULE 5: SELF-ASSESSMENT QUIZ
# ==========================================
elif mode == "5. Self-Assessment Quiz":
    st.header("5. ISLR 12.2 Knowledge Check")

    q1 = st.radio(
        "Q1. What is the maximum number of distinct principal components possible for a dataset with n observations and p variables?",
        [
            "A) Always p",
            "B) Always n",
            "C) min(n - 1, p)",
            "D) max(n, p)"
        ]
    )

    if st.button("Check Q1 Answer"):
        if q1 == "C) min(n - 1, p)":
            st.success("Correct! ISLR specifies that we can calculate at most min(n-1, p) principal components.")
        else:
            st.error("Incorrect! The correct answer is min(n-1, p).")

    st.divider()

    q2 = st.radio(
        "Q2. Why is variable scaling/standardization important before applying PCA?",
        [
            "A) PCA cannot run on unscaled data.",
            "B) Variables with higher variance will dominate the principal components if not scaled.",
            "C) Standardization increases the number of components.",
            "D) It converts non-linear patterns to linear patterns."
        ]
    )

    if st.button("Check Q2 Answer"):
        if q2 == "B) Variables with higher variance will dominate the principal components if not scaled.":
            st.success("Correct! Unscaled variables measured in large units will artificially carry high loadings.")
        else:
            st.error("Incorrect! Read ISLR Section 12.2.4 on Scaling Variables.")