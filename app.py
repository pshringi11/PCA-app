import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Page Config
st.set_page_config(
    page_title="ISLR PCA & PCR Learning Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ISLR Chapter 12.2 & 6.3.1: PCA & Principal Components Regression")
st.markdown("*An Introduction to Statistical Learning with Applications in Python (2023)* - **Ch 12.2 (PCA) & Ch 6.3.1 (PCR)**")

# Sidebar - Mode Selection
st.sidebar.header("Navigation")
mode = st.sidebar.radio(
    "Choose Module:",
    [
        "1. Upload CSV & Run Real-time PCA",
        "2. Principal Components Regression (Ch 6.3.1)",
        "3. Interactive 2D PCA & Rotation",
        "4. Feature Scaling & Standardization",
        "5. Scree Plot & PVE Analysis",
        "6. Mathematical Formulas & Concepts",
        "7. Self-Assessment Quiz"
    ]
)

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# MODULE 1: UPLOAD CSV & REAL-TIME PCA
# ==========================================
if mode == "1. Upload CSV & Run Real-time PCA":
    st.header("1. Upload Your CSV Dataset & Real-time PCA Analysis")
    st.markdown("""
    Apna custom dataset (CSV format) upload karein, numerical variables select karein aur real-time me PCA scores, loadings, biplot aur variance breakdown analysis dekhein.
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset successfully loaded: `{df_raw.shape[0]}` rows and `{df_raw.shape[1]}` columns.")

            # Show Data Preview
            with st.expander("🔍 Dataset Preview & Summary", expanded=False):
                st.dataframe(df_raw.head())
                st.write("**Data Types & Null Counts:**")
                st.dataframe(pd.DataFrame({"Data Type": df_raw.dtypes, "Null Values": df_raw.isnull().sum()}))

            # Separate Numerical & Categorical Columns
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df_raw.columns.tolist()

            if len(num_cols) < 2:
                st.error("⚠️ PCA run karne ke liye dataset me kam se kam 2 Numerical Columns hona zaroori hain.")
            else:
                st.subheader("⚙️ PCA Settings")
                col_s1, col_s2, col_s3 = st.columns(3)

                with col_s1:
                    selected_features = st.multiselect(
                        "Select Numerical Features for PCA:",
                        options=num_cols,
                        default=num_cols
                    )

                with col_s2:
                    color_col = st.selectbox(
                        "Select Categorical/Label Column for Grouping (Optional):",
                        options=["None"] + all_cols
                    )

                with col_s3:
                    scale_data = st.checkbox("Standardize Features (StandardScaler)", value=True, help="PCA se pehle mean=0 aur variance=1 karna highly recommended hai.")

                if len(selected_features) >= 2:
                    X = df_raw[selected_features].dropna()

                    if scale_data:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                    else:
                        X_scaled = X.values - X.mean().values

                    # PCA Calculation
                    n_comps = min(len(selected_features), X_scaled.shape[0])
                    pca = PCA(n_components=n_comps)
                    X_pca = pca.fit_transform(X_scaled)

                    # Create DataFrame for PCA Scores
                    pca_cols = [f"PC{i+1}" for i in range(n_comps)]
                    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

                    if color_col != "None" and color_col in df_raw.columns:
                        df_pca[color_col] = df_raw.loc[X.index, color_col].astype(str)

                    # Variance Summary Metrics
                    pve = pca.explained_variance_ratio_
                    cum_pve = np.cumsum(pve)

                    st.markdown("---")
                    st.subheader("📈 PCA Variance & Loadings Overview")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("PC1 Variance Explained (PVE)", f"{pve[0]*100:.2f}%")
                    m2.metric("PC2 Variance Explained (PVE)", f"{pve[1]*100:.2f}%" if n_comps > 1 else "N/A")
                    m3.metric("Cumulative PVE (PC1 + PC2)", f"{cum_pve[1]*100:.2f}%" if n_comps > 1 else f"{pve[0]*100:.2f}%")

                    # Loadings Table
                    loadings_df = pd.DataFrame(
                        pca.components_.T,
                        columns=pca_cols,
                        index=selected_features
                    )

                    col_vis1, col_vis2 = st.columns([1.2, 1])

                    with col_vis1:
                        st.write("**Feature Loadings ($\phi_{ji}$ Vectors):**")
                        st.dataframe(loadings_df.style.background_gradient(cmap="coolwarm", axis=None).format("{:.4f}"))

                    with col_vis2:
                        # Loadings Heatmap
                        fig_hm = px.imshow(
                            loadings_df,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Loadings Heatmap"
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)

                    # Visualizations (2D & 3D Biplot)
                    st.subheader("🎨 Principal Component Scatter & Biplot Visualizations")

                    tab1, tab2, tab3 = st.tabs(["2D Score Scatter Plot", "3D Score Scatter Plot", "Scree Plot"])

                    with tab1:
                        if n_comps >= 2:
                            color_kw = color_col if color_col != "None" else None
                            fig_2d = px.scatter(
                                df_pca, x="PC1", y="PC2",
                                color=color_kw,
                                title=f"PC1 vs PC2 Scatter Plot (Total Variance Captured: {cum_pve[1]*100:.1f}%)",
                                labels={
                                    "PC1": f"PC1 ({pve[0]*100:.1f}% Variance)",
                                    "PC2": f"PC2 ({pve[1]*100:.1f}% Variance)"
                                },
                                hover_data=[color_col] if color_col != "None" else None
                            )

                            # Overlay loadings vectors (Biplot arrows)
                            scale_factor = np.max(np.abs(X_pca[:, :2])) * 0.8
                            for feature in selected_features:
                                fig_2d.add_annotation(
                                    x=loadings_df.loc[feature, "PC1"] * scale_factor,
                                    y=loadings_df.loc[feature, "PC2"] * scale_factor,
                                    ax=0, ay=0,
                                    xanchor="center", yanchor="center",
                                    text=feature,
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="red"
                                )

                            st.plotly_chart(fig_2d, use_container_width=True)

                    with tab2:
                        if n_comps >= 3:
                            color_kw = color_col if color_col != "None" else None
                            fig_3d = px.scatter_3d(
                                df_pca, x="PC1", y="PC2", z="PC3",
                                color=color_kw,
                                title=f"3D PCA Scatter Plot (PC1 + PC2 + PC3 Variance: {cum_pve[2]*100:.1f}%)",
                                labels={
                                    "PC1": f"PC1 ({pve[0]*100:.1f}%)",
                                    "PC2": f"PC2 ({pve[1]*100:.1f}%)",
                                    "PC3": f"PC3 ({pve[2]*100:.1f}%)"
                                }
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.info("3D Visualization ke liye kam se kam 3 features required hain.")

                    with tab3:
                        fig_scree = go.Figure()
                        fig_scree.add_trace(go.Bar(
                            x=pca_cols, y=pve, name="Individual PVE",
                            marker_color="#3b82f6"
                        ))
                        fig_scree.add_trace(go.Scatter(
                            x=pca_cols, y=cum_pve, name="Cumulative PVE",
                            mode="lines+markers", marker_color="#10b981"
                        ))
                        fig_scree.update_layout(
                            title="Scree Plot & Cumulative Variance Explained",
                            xaxis_title="Principal Components",
                            yaxis_title="Proportion of Variance",
                            yaxis=dict(range=[0, 1.05])
                        )
                        st.plotly_chart(fig_scree, use_container_width=True)

                    # Export Transformed PCA Dataset
                    st.subheader("📥 Download Transformed Dataset")
                    csv_export = pd.concat([df_raw.loc[X.index], df_pca], axis=1).to_csv(index=False)
                    st.download_button(
                        label="Download Dataset with PCA Scores (CSV)",
                        data=csv_export,
                        file_name="pca_transformed_dataset.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("PCA run karne ke liye kam se kam 2 Features select karein.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("👆 Kripya ek CSV file upload karein. Testing ke liye aap koi bhi standard dataset use kar sakte hain.")

# ==========================================
# MODULE 2: PRINCIPAL COMPONENTS REGRESSION (CH 6.3.1)
# ==========================================
elif mode == "2. Principal Components Regression (Ch 6.3.1)":
    st.header("2. Chapter 6.3.1 - Principal Components Regression (PCR)")
    st.markdown("""
    **ISLR Chapter 6.3.1 Key Insight:** 
    PCR constructs the first $M$ principal components $Z_1, Z_2, \dots, Z_M$ from $p$ predictors ($M \le p$) 
    and then fits a standard linear regression model using these $M$ components as predictors:
    $$y_i = \\theta_0 + \\sum_{m=1}^M \\theta_m z_{im} + \\epsilon_i$$
    
    * **Why use PCR?** Multicollinearity reduces, variance drops, and high-dimensional data ($p > n$) becomes solvable.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Controls")
        n_obs = st.slider("Number of Observations (n)", 50, 500, 150)
        p_vars = st.slider("Number of Correlated Predictors (p)", 5, 30, 12)
        noise_sd = st.slider("Response Noise (std dev)", 0.5, 5.0, 1.5)
        
        # Generate correlated predictors
        X_base = np.random.randn(n_obs, 2)
        weights = np.random.uniform(0.5, 2.0, (2, p_vars))
        X = X_base @ weights + np.random.randn(n_obs, p_vars) * 0.5
        
        # Response Y is driven by first component primarily
        true_beta = np.random.uniform(1.0, 3.0, p_vars)
        y = X @ true_beta + np.random.normal(0, noise_sd, n_obs)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Component Selector
        m_components = st.slider("Select Number of Components (M)", 1, p_vars, min(3, p_vars))

        # Fit PCR for chosen M
        pca_pcr = PCA(n_components=m_components)
        Z_train = pca_pcr.fit_transform(X_train_scaled)
        Z_test = pca_pcr.transform(X_test_scaled)

        lr = LinearRegression()
        lr.fit(Z_train, y_train)

        pred_train = lr.predict(Z_train)
        pred_test = lr.predict(Z_test)

        mse_train = mean_squared_error(y_train, pred_train)
        mse_test = mean_squared_error(y_test, pred_test)

        # Standard OLS Baseline
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        ols_test_pred = ols.predict(X_test_scaled)
        ols_mse = mean_squared_error(y_test, ols_test_pred)

        st.metric(f"PCR Test MSE (M={m_components})", f"{mse_test:.3f}")
        st.metric("Standard OLS Test MSE (Full M=p)", f"{ols_mse:.3f}")
        pve_m = sum(pca_pcr.explained_variance_ratio_) * 100
        st.info(f"Total Predictor Variance Captured by $M={m_components}$: `{pve_m:.2f}%`")

    with col2:
        # Cross-validation curve simulator across all M
        ms = list(range(1, p_vars + 1))
        train_mses = []
        test_mses = []

        pca_full = PCA(n_components=p_vars)
        Z_tr_full = pca_full.fit_transform(X_train_scaled)
        Z_te_full = pca_full.transform(X_test_scaled)

        for m in ms:
            reg = LinearRegression()
            reg.fit(Z_tr_full[:, :m], y_train)
            train_mses.append(mean_squared_error(y_train, reg.predict(Z_tr_full[:, :m])))
            test_mses.append(mean_squared_error(y_test, reg.predict(Z_te_full[:, :m])))

        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Scatter(x=ms, y=train_mses, mode='lines+markers', name='Train MSE', line=dict(color='#3b82f6')))
        fig_pcr.add_trace(go.Scatter(x=ms, y=test_mses, mode='lines+markers', name='Test MSE', line=dict(color='#ef4444')))
        
        # Selected M highlight
        fig_pcr.add_vline(x=m_components, line_dash="dash", line_color="green", annotation_text=f"Selected M={m_components}")

        fig_pcr.update_layout(
            title="PCR MSE vs Number of Components (M)",
            xaxis_title="Number of Principal Components (M)",
            yaxis_title="Mean Squared Error (MSE)",
            height=500
        )
        st.plotly_chart(fig_pcr, use_container_width=True)

# ==========================================
# MODULE 3: INTERACTIVE 2D PCA & ROTATION
# ==========================================
elif mode == "3. Interactive 2D PCA & Rotation":
    st.header("3. Interactive Projection & Vector Rotation")
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
# MODULE 4: FEATURE SCALING & STANDARDIZATION
# ==========================================
elif mode == "4. Feature Scaling & Standardization":
    st.header("4. Impact of Variable Scaling (ISLR Section 12.2.4)")
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
# MODULE 5: SCREE PLOT & PVE ANALYSIS
# ==========================================
elif mode == "5. Scree Plot & PVE Analysis":
    st.header("5. Proportion of Variance Explained (PVE) & Scree Plot")
    st.markdown("""
    The **Scree Plot** shows the proportion of total variance captured by each principal component:
    $$PVE_m = \\frac{\\sum_{i=1}^n z_{im}^2}{\\sum_{j=1}^p \\sum_{i=1}^n x_{ij}^2}$$
    """)

    n_features = st.slider("Select Number of Synthetic Features (p)", 3, 15, 8)
    n_samples = st.slider("Select Sample Size (n)", 50, 500, 200)

    # Generate synthetic multi-feature data
    X_synthetic = np.random.randn(n_samples, n_features)
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
# MODULE 6: MATHEMATICAL FORMULAS & CONCEPTS
# ==========================================
elif mode == "6. Mathematical Formulas & Concepts":
    st.header("6. Core Mathematical Concepts (ISLR Ch 12.2 & Ch 6.3.1)")

    st.subheader("1. First Principal Component Optimization (Ch 12.2)")
    st.latex(r"""
    \max_{\phi_{11},\dots,\phi_{p1}} \left\{ \frac{1}{n}\sum_{i=1}^n \left( \sum_{j=1}^p \phi_{j1} x_{ij} \right)^2 \right\} 
    \quad \text{subject to} \quad \sum_{j=1}^p \phi_{j1}^2 = 1
    """)

    st.subheader("2. Principal Component Scores")
    st.latex(r"""
    z_{i1} = \phi_{11} x_{i1} + \phi_{21} x_{i2} + \dots + \phi_{p1} x_{ip}
    """)

    st.subheader("3. Principal Components Regression Equation (Ch 6.3.1)")
    st.latex(r"""
    y_i = \theta_0 + \sum_{m=1}^M \theta_m z_{im} + \epsilon_i \quad \text{where } M \le p
    """)
    st.markdown("""
    When expanded back in terms of original predictors $X_1, \dots, X_p$:
    $$y_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij} + \epsilon_i \quad \text{where } \beta_j = \sum_{m=1}^M \theta_m \phi_{jm}$$
    """)

# ==========================================
# MODULE 7: SELF-ASSESSMENT QUIZ
# ==========================================
elif mode == "7. Self-Assessment Quiz":
    st.header("7. ISLR Knowledge Check (Ch 12.2 & Ch 6.3.1)")

    q1 = st.radio(
        "Q1. Is PCR a supervised or unsupervised feature selection method?",
        [
            "A) Fully Supervised, because it predicts Y.",
            "B) Unsupervised dimension reduction followed by supervised regression.",
            "C) Semi-supervised classification algorithm.",
            "D) Fully Unsupervised, because it ignores Y."
        ]
    )

    if st.button("Check Q1 Answer"):
        if q1 == "B) Unsupervised dimension reduction followed by supervised regression.":
            st.success("Correct! PCR generates components strictly based on X (unsupervised), then uses Y to fit OLS regression.")
        else:
            st.error("Incorrect! ISLR Chapter 6.3 emphasizes that PCA/PCR generates directions without using Y.")

    st.divider()

    q2 = st.radio(
        "Q2. What is the maximum number of distinct principal components possible for a dataset with n observations and p variables?",
        [
            "A) Always p",
            "B) Always n",
            "C) min(n - 1, p)",
            "D) max(n, p)"
        ]
    )

    if st.button("Check Q2 Answer"):
        if q2 == "C) min(n - 1, p)":
            st.success("Correct! ISLR specifies that we can calculate at most min(n-1, p) principal components.")
        else:
            st.error("Incorrect! The correct answer is min(n-1, p).")
