import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), \"..\"))
from app.data_preprocessing import load_data, preprocess_data
from models.model_training import train_model, evaluate_model

st.set_page_config(layout=\"wide\")

st.title(\"Enterprise AI Decision Support Platform\")

# Sidebar for navigation
st.sidebar.title(\"Navigation\")
selection = st.sidebar.radio(\"Go to\", [\"Dashboard Overview\", \"Data Upload & Preprocessing\", \"Model Training & Evaluation\", \"Prediction\"])

if selection == \"Dashboard Overview\":
    st.header(\"Executive KPI Dashboard\")
    st.write(\"This section provides an overview of key performance indicators.\")

    # Dummy data for demonstration
    kpi_data = {
        \"KPI\": [\"Revenue\", \"Profit Margin\", \"Customer Acquisition\", \"Sales Growth\"],
        \"Value\": [\"$1.2M\", \"25%\", \"1500\", \"10%\"],
        \"Trend\": [\"Up\", \"Stable\", \"Up\", \"Up\"]
    }
    st.table(kpi_data)

    st.subheader(\"Revenue Trends\")
    revenue_df = pd.DataFrame({
        \"Month\": [\"Jan\", \"Feb\", \"Mar\", \"Apr\", \"May\", \"Jun\"],
        \"Revenue\": [100, 120, 150, 130, 160, 180]
    })
    fig_revenue = px.line(revenue_df, x=\"Month\", y=\"Revenue\", title=\"Monthly Revenue Trend\")
    st.plotly_chart(fig_revenue, use_container_width=True)

    st.subheader(\"Customer Analytics\")
    customer_df = pd.DataFrame({
        \"Segment\": [\"New\", \"Existing\", \"Churned\"],
        \"Count\": [500, 2000, 100]
    })
    fig_customer = px.pie(customer_df, values=\"Count\", names=\"Segment\", title=\"Customer Segmentation\")
    st.plotly_chart(fig_customer, use_container_width=True)


elif selection == \"Data Upload & Preprocessing\":
    st.header(\"Upload Your Data\")
    uploaded_file = st.file_uploader(\"Choose a CSV file\", type=\"csv\")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write(\"Original Data Preview:\")
            st.dataframe(df.head())

            st.subheader(\"Preprocessing Steps\")
            if st.button(\"Perform Preprocessing\"):
                X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)
                if X_train is not None:
                    st.success(\"Data preprocessed successfully!\")
                    st.write(\"Preprocessed Data (X_train) Preview:\")
                    st.dataframe(X_train.head())
                    st.session_state["X_train"] = X_train
                    st.session_state["X_test"] = X_test
                    st.session_state["y_train"] = y_train
                    st.session_state["y_test"] = y_test
                    st.session_state["scaler"] = scaler
                    st.session_state["feature_columns"] = feature_columns
                else:
                    st.error(\"Error during data preprocessing.\")

elif selection == \"Model Training & Evaluation\":
    st.header(\"Train and Evaluate Machine Learning Models\")

    if \"X_train\" in st.session_state:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        model_choice = st.selectbox(
            \"Choose a model to train:\",
            (\"LogisticRegression\", \"RandomForestClassifier\", \"XGBClassifier\", \"LGBMClassifier\")
        )

        if st.button(\"Train Model\"):
            with st.spinner(f\"Training {model_choice}..."):
                model = train_model(X_train, y_train, model_name=model_choice)
                metrics = evaluate_model(model, X_test, y_test)
                st.success(f\"{model_choice} trained and evaluated!\")
                st.write(\"Model Metrics:\", metrics)
                st.session_state["trained_model"] = model
    else:
        st.warning(\"Please upload and preprocess data first in the \"Data Upload & Preprocessing\" section.\")

elif selection == \"Prediction\":
    st.header(\"Make Predictions\")
    st.write(\"Enter features to get a prediction.\")

    if \"trained_model\" in st.session_state and \"feature_columns\" in st.session_state:
        model = st.session_state["trained_model"]
        feature_columns = st.session_state["feature_columns"]
        scaler = st.session_state["scaler"]

        input_data = {}
        for col in feature_columns:
            # Simplified input for demonstration; in real app, handle different data types
            input_data[col] = st.number_input(f\"Enter value for {col}\", value=0.0)

        if st.button(\"Predict\"):
            input_df = pd.DataFrame([input_data])
            # Apply the same scaling as during training
            # Note: This assumes all feature_columns are numerical and were scaled.
            # A more robust solution would handle categorical features and ensure correct order.
            try:
                input_df_scaled = scaler.transform(input_df[feature_columns])
                prediction = model.predict(input_df_scaled)[0]
                st.success(f\"Prediction: {prediction}\")
            except Exception as e:
                st.error(f\"Error during prediction: {e}. Make sure input features match training data.\")
    else:
        st.warning(\"Please train a model first in the \"Model Training & Evaluation\" section.\")
