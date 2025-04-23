# app.py

import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import requests
from io import BytesIO
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a custom root_mean_squared_error function if it's not available
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

# Set Streamlit page configuration
st.set_page_config(
    page_title="🦪 Abalone Age Predictor",
    page_icon="🦪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #4B0082;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
    }
    .subtitle {
        color: #483D8B;
        text-align: center;
        font-size: 24px;
    }
    .info {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown("<h1 class='title'>🦪 Abalone Age Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Predict the Age of Abalone Using K-Nearest Neighbors Regression</h3>", unsafe_allow_html=True)

# Display Abalone Image with Reduced Size
image_url = "https://seahistory.org/wp-content/uploads/Abalone-Underwater.jpg"
response = requests.get(image_url)
if response.status_code == 200:
    abalone_image = Image.open(BytesIO(response.content))
    new_width = int(abalone_image.width * 0.35)
    new_height = int(abalone_image.height * 0.35)
    resized_image = abalone_image.resize((new_width, new_height))
    
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Center the image in the middle column
    with col2:
        st.image(resized_image, caption="Abalone Underwater")
else:
    st.error("Failed to load abalone image.")

# Add Description About Abalone and Project
st.markdown(
    """
    <div class="info">
    <h3>About Abalones</h3>
    <p>
    Abalones are marine snails known for their colorful, ear-shaped shells. They are harvested for their meat, which is considered a delicacy in many cultures. The age of an abalone is determined by counting the number of rings on its shell, a process that is both time-consuming and labor-intensive.
    </p>
    <h3>Project Overview</h3>
    <p>
    This project utilizes the K-Nearest Neighbors (K-NN) regression technique to predict the age of abalones based on various physical measurements. By inputting parameters such as length, diameter, height, and weights, the model provides an estimated age, streamlining the age determination process.
    </p>
    <p><b>
    Developed by Shravani Raykar
    </p></b>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user inputs and information
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    sex = st.sidebar.selectbox('Sex', ('M', 'F', 'I'))
    length = st.sidebar.slider('Length (mm)', 0.075, 0.815, 0.5, 0.01)
    diameter = st.sidebar.slider('Diameter (mm)', 0.055, 0.650, 0.5, 0.01)
    height = st.sidebar.slider('Height (mm)', 0.0, 1.130, 0.3, 0.01)
    whole_weight = st.sidebar.slider('Whole Weight (g)', 0.002, 2.826, 0.5, 0.01)
    shucked_weight = st.sidebar.slider('Shucked Weight (g)', 0.001, 1.488, 0.5, 0.01)
    viscera_weight = st.sidebar.slider('Viscera Weight (g)', 0.001, 0.760, 0.2, 0.01)
    shell_weight = st.sidebar.slider('Shell Weight (g)', 0.002, 1.005, 0.2, 0.01)
    data = {
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole weight': whole_weight,
        'Shucked weight': shucked_weight,
        'Viscera weight': viscera_weight,
        'Shell weight': shell_weight
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load the dataset using ucimlrepo
@st.cache_data
def load_data():
    abalone = fetch_ucirepo(id=1)  # ID 1 corresponds to the Abalone dataset
    X = abalone.data.features
    y = abalone.data.targets
    df = X.copy()
    df['Rings'] = y
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    return df

df = load_data()

# Sidebar: Show Dataset Information
st.sidebar.subheader("📚 Dataset Information")
dataset_info = """
**Title:** Abalone Data Set

**Sources:**
- **Original Owners:** Marine Resources Division, Marine Research Laboratories - Taroona, Department of Primary Industry and Fisheries, Tasmania.
- **Donor:** Sam Waugh, Department of Computer Science, University of Tasmania.
- **Date Received:** December 1995

**Number of Instances:** 4177  
**Number of Attributes:** 8  
**Class Distribution:** Rings range from 1 to 29.

**Description:**  
Predicting the age of abalone from physical measurements. The age of abalone is determined by counting the number of rings, which is a time-consuming task. This dataset includes various physical measurements to predict the age more efficiently.
"""
st.sidebar.info(dataset_info)

# Main Content Area with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Overview", "📈 Visualizations", "🤖 Model Training", "📊 Prediction"])

with tab1:
    st.subheader("🔍 Data Overview")
    if st.checkbox('Show Raw Data'):
        st.write(df.head())
    
    st.markdown("### 🧮 Statistical Summary")
    st.write(df.describe())
    
    st.markdown("### 📊 Class Distribution")
    fig_dist = px.histogram(df, x='Rings', nbins=30, title='Distribution of Rings', 
                            labels={'Rings':'Number of Rings'}, color_discrete_sequence=['#4B0082'])
    st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.subheader("📈 Exploratory Data Analysis")
    
    # Encode 'Sex' for numerical computations
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['Sex'] = le.fit_transform(df_encoded['Sex'])
    
    # Correlation Heatmap
    st.markdown("#### 🔗 Correlation Heatmap")
    corr = df_encoded.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    # Scatter Plot: Length vs. Whole Weight
    st.markdown("#### 📏 Length vs. Whole Weight")
    fig_scatter = px.scatter(df, x='Length', y='Whole weight', color='Sex',
                             title='Whole Weight vs. Length by Sex',
                             labels={'Length':'Length (mm)', 'Whole weight':'Whole Weight (g)'})
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Box Plot: Height by Sex
    st.markdown("#### 📦 Height Distribution by Sex")
    fig_box = px.box(df, x='Sex', y='Height', title='Height Distribution by Sex',
                    labels={'Height':'Height (mm)'}, color='Sex', color_discrete_sequence=['#483D8B', '#6A5ACD', '#7B68EE'])
    st.plotly_chart(fig_box, use_container_width=True)
    
    

with tab3:
    st.subheader("🤖 Model Training and Evaluation")
    
    # Data Preprocessing
    X = df_encoded.drop('Rings', axis=1)
    y = df_encoded['Rings']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Update tooltips to display only once and make them bold
    TOOLTIPS = {
        "Optimal K": "**Optimal K**: The value of K in K-Nearest Neighbors that minimizes the Root Mean Squared Error (RMSE).",
        "RMSE": "**RMSE**: Root Mean Squared Error, a measure of how well the model's predictions match the actual values. Lower RMSE indicates better performance.",
        "MAE": "**MAE**: Mean Absolute Error, the average of the absolute differences between predicted and actual values. It measures prediction accuracy.",
        "R²": "**R²**: R-squared, a statistical measure that indicates how well the regression predictions approximate the actual data points. Higher values are better.",
        "Training Data": "**Training Data**: The portion of the dataset used to train the machine learning model.",
        "Testing Data": "**Testing Data**: The portion of the dataset used to evaluate the model's performance on unseen data.",
        "Scaling": "**Scaling**: A preprocessing step to normalize data to a specific range, often [0, 1], to improve model performance.",
        "K-Nearest Neighbors": "**K-Nearest Neighbors**: A machine learning algorithm that predicts values based on the K closest data points in the feature space.",
        "Correlation Heatmap": "**Correlation Heatmap**: A graphical representation of the correlation matrix, showing the strength and direction of relationships between variables.",
        "Actual vs Predicted": "**Actual vs Predicted**: A comparison of the model's predictions against the actual values to evaluate its accuracy."
    }

    # Function to add tooltips to terms
    @st.cache_data
    def add_tooltip(term):
        return f'<span title="{TOOLTIPS.get(term, term)}" style="text-decoration: underline; cursor: help; font-weight: bold;">{term}</span>'

    # Update markdowns to display tooltips only once
    st.markdown(f"**Training Data and Testing Data:** {add_tooltip('Training Data')} is used to train the model, while {add_tooltip('Testing Data')} evaluates its performance.", unsafe_allow_html=True)
    st.markdown(f"**Scaling:** {add_tooltip('Scaling')} is applied to normalize the data.", unsafe_allow_html=True)
    st.markdown(f"**K-Nearest Neighbors:** {add_tooltip('K-Nearest Neighbors')} is the algorithm used for regression.", unsafe_allow_html=True)
    st.markdown(f"**Correlation Heatmap:** {add_tooltip('Correlation Heatmap')} visualizes relationships between variables.", unsafe_allow_html=True)
    st.markdown(f"**Actual vs Predicted:** {add_tooltip('Actual vs Predicted')} evaluates model accuracy.", unsafe_allow_html=True)
    st.markdown(f"**Optimal K:** {add_tooltip('Optimal K')} is determined to minimize RMSE.", unsafe_allow_html=True)
    st.markdown(f"**RMSE, MAE, and R²:** {add_tooltip('RMSE')}, {add_tooltip('MAE')}, and {add_tooltip('R²')} are metrics used to evaluate model performance.", unsafe_allow_html=True)

    # Finding the optimal k
    st.markdown("##### 🔍 Finding the Optimal K")
    rmse_val = []
    mae_val = []
    r2_val = []
    k_list = list(range(1, 51))
    for k in k_list:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        # Use root_mean_squared_error if available, else use custom function
        rmse = root_mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rmse_val.append(rmse)
        mae_val.append(mae)
        r2_val.append(r2)
    
    # Identifying Best K based on RMSE
    best_rmse = min(rmse_val)
    best_k = k_list[rmse_val.index(best_rmse)]
    best_mae = mae_val[rmse_val.index(best_rmse)]
    best_r2 = r2_val[rmse_val.index(best_rmse)]
    
    st.markdown(f"{add_tooltip('Optimal K')} with{add_tooltip('RMSE')} = {best_rmse:.2f},{add_tooltip('MAE')} = {best_mae:.2f},{add_tooltip('R²')} = {best_r2:.2f}", unsafe_allow_html=True)
    
    # Plotting Evaluation Metrics vs K
    st.markdown("##### 📉 Evaluation Metrics vs Number of Neighbors (K)")
    fig_metrics = make_subplots(
        rows=3, cols=1,
        subplot_titles=("RMSE vs K", "MAE vs K", "R² Score vs K"),
        shared_xaxes=True
    )
    
    # RMSE Plot
    fig_metrics.add_trace(
        px.line(x=k_list, y=rmse_val, markers=True, labels={'x':'K', 'y':'RMSE'}).data[0],
        row=1, col=1
    )
    fig_metrics.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f'Optimal K = {best_k}', row=1, col=1)
    
    # MAE Plot
    fig_metrics.add_trace(
        px.line(x=k_list, y=mae_val, markers=True, labels={'x':'K', 'y':'MAE'}).data[0],
        row=2, col=1
    )
    fig_metrics.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f'Optimal K = {best_k}', row=2, col=1)
    
    # R² Score Plot
    fig_metrics.add_trace(
        px.line(x=k_list, y=r2_val, markers=True, labels={'x':'K', 'y':'R² Score'}).data[0],
        row=3, col=1
    )
    fig_metrics.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f'Optimal K = {best_k}', row=3, col=1)
    
    fig_metrics.update_layout(height=900, showlegend=False, title_text="Model Evaluation Metrics")
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Train the final model with optimal k
    final_model = KNeighborsRegressor(n_neighbors=best_k)
    final_model.fit(X_train_scaled, y_train)
    
    # Display model performance
    y_pred = final_model.predict(X_test_scaled)
    rmse_final = root_mean_squared_error(y_test, y_pred)
    mae_final = mean_absolute_error(y_test, y_pred)
    r2_final = r2_score(y_test, y_pred)
    
    st.markdown("### 📊 Final Model Performance on Test Set")
    st.markdown(f"- **RMSE:** {rmse_final:.2f}")
    st.markdown(f"- **MAE:** {mae_final:.2f}")
    st.markdown(f"- **R² Score:** {r2_final:.2f}")
    
    # Additional Visualization: Actual vs Predicted
    st.markdown("#### 🔄 Actual vs Predicted Rings")
    fig_actual_pred = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Rings', 'y':'Predicted Rings'},
                                 title='Actual vs Predicted Rings',
                                 trendline="ols")
    st.plotly_chart(fig_actual_pred, use_container_width=True)
    
    # Download Model Metrics
    st.markdown("### 📥 Download Model Metrics")
    metrics_df = pd.DataFrame({
        'K': k_list,
        'RMSE': rmse_val,
        'MAE': mae_val,
        'R² Score': r2_val
    })
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Metrics as CSV",
        data=csv,
        file_name='model_metrics.csv',
        mime='text/csv',
    )

with tab4:
    st.subheader("📊 Prediction")
    st.markdown("Enter the abalone's characteristics in the sidebar to predict its age (number of rings).")
    
    # Encode and scale user input
    input_encoded = input_df.copy()
    input_encoded['Sex'] = le.transform(input_encoded['Sex'])
    input_scaled = scaler.transform(input_encoded)
    
    # Prediction
    prediction = final_model.predict(input_scaled)[0]
    age = prediction + 1.5  # As per dataset description
    
    # Display Prediction
    st.markdown("### 🎯 Predicted Age")
    st.success(f"The predicted age of the abalone is **{round(age):.0f} years** (Number of Rings: {prediction:.2f})")
    # Additional Visualizations: Predicted Rings
    st.markdown("#### 📈 Prediction Details")
    fig_pred = px.bar(x=['Predicted Rings'], y=[prediction], 
                      labels={'x':'', 'y':'Number of Rings'},
                      title='Predicted Number of Rings',
                      text=[f"{prediction:.2f}"],
                      color=['Predicted Rings'],
                      color_discrete_sequence=['#4B0082'])
    fig_pred.update_traces(textposition='auto')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Download Prediction
    st.markdown("### 📥 Download Prediction")
    prediction_df = pd.DataFrame({
        'Sex': input_df['Sex'],
        'Length (mm)': input_df['Length'],
        'Diameter (mm)': input_df['Diameter'],
        'Height (mm)': input_df['Height'],
        'Whole Weight (g)': input_df['Whole weight'],
        'Shucked Weight (g)': input_df['Shucked weight'],
        'Viscera Weight (g)': input_df['Viscera weight'],
        'Shell Weight (g)': input_df['Shell weight'],
        'Predicted Rings': [prediction],
        'Predicted Age (years)': [age]
    })
    csv_pred = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Prediction as CSV",
        data=csv_pred,
        file_name='abalone_prediction.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
    <b>Developed by Shravani Raykar</b><br> 
    Contact: [shravaniraykar@gmail.com]
    </div>
    """, unsafe_allow_html=True)
