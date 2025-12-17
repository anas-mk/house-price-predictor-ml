import streamlit as st
import joblib
import numpy as np 

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------ Load Model ------------------
model = joblib.load("model.pkl")

# ------------------ Header ------------------
st.markdown(
    """
    <h1 style="text-align:center;">🏠 House Price Prediction</h1>
    <p style="text-align:center; color:gray;">
        Enter house details and get an instant price prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------ Layout ------------------
left, right = st.columns([1, 2])

# ------------------ Sidebar Inputs ------------------
with left:
    st.markdown("### House Features")

    bedrooms = st.number_input(" Bedrooms", min_value=0, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2)
    living_area = st.number_input("Living Area (sqft)", min_value=0, value=2000)
    condition = st.slider("Condition", 1, 5, 3)
    schools = st.number_input("Nearby Schools", min_value=0, value=3)

    predict_button = st.button(" Predict Price", use_container_width=True)

# ------------------ Prediction Section ------------------
with right:
    st.markdown("### 📊 Prediction Result")

    if predict_button:
        X = np.array([[bedrooms, bathrooms, living_area, condition, schools]])
        prediction = model.predict(X)[0]

        st.success("✅ Prediction Complete")
        st.metric(
            label="Estimated House Price",
            value=f"${prediction:,.2f}"
        )

        st.balloons()
    else:
        st.info("👈 Enter the values and click **Predict Price**")

# ------------------ Footer ------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with Streamlit ❤️</p>",
    unsafe_allow_html=True
)



