import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="EV Battery Failure Risk",
    page_icon="🔋",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "ev_battery_pipeline.pkl"

st.title("🔋 EV Battery Failure Risk Prediction")
st.write("Enter battery details to estimate the failure-risk category.")

if not MODEL_PATH.exists():
    st.error("Model file not found: ev_battery_pipeline.pkl")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

if not hasattr(model, "predict"):
    st.error(
        "The saved file does not contain a prediction model. "
        "Save your fitted model or pipeline using joblib.dump()."
    )
    st.stop()


with st.form("battery_form"):
    st.subheader("Battery and usage details")

    c1, c2, c3 = st.columns(3)

    with c1:
        chemistry = st.selectbox(
            "Battery chemistry",
            ["LFP", "NMC", "NCA", "LCO", "Other"]
        )

        manufacturer = st.text_input(
            "Manufacturer", "Manufacturer_A"
        )

        bms_firmware_version = st.selectbox(
            "BMS Firmware Version",
            ["BMS_v2.1", "BMS_v3.1"]
        )

        climate_zone = st.selectbox(
            "Climate zone",
            ["Temperate", "Hot", "Cold", "Humid", "Arid", "Other"]
        )

        degradation_category = st.selectbox(
            "Degradation category",
            ["Low", "Moderate", "High", "Severe", "Unknown"]
        )

        pack_capacity_kwh = st.number_input(
            "Pack capacity (kWh)", min_value=0.1, value=60.0
        )

        battery_age_days = st.number_input(
            "Battery age (days)", min_value=0, value=500
        )

        cycle_count = st.number_input(
            "Cycle count", min_value=0, value=500
        )

        avg_daily_distance_km = st.number_input(
            "Average daily distance (km)", min_value=0.0, value=40.0
        )

        avg_charging_power_kw = st.number_input(
            "Average charging power (kW)", min_value=0.0, value=7.0
        )

    with c2:
        fast_charge_ratio = st.slider(
            "Fast-charge ratio", 0.0, 1.0, 0.25
        )

        avg_depth_of_discharge_pct = st.slider(
            "Average depth of discharge (%)", 0.0, 100.0, 60.0
        )

        avg_state_of_charge_pct = st.slider(
            "Average state of charge (%)", 0.0, 100.0, 70.0
        )

        charge_cycles_per_month = st.number_input(
            "Charge cycles per month", min_value=0.0, value=25.0
        )

        total_energy_throughput_kwh = st.number_input(
            "Total energy throughput (kWh)",
            min_value=0.0,
            value=30000.0
        )

        avg_ambient_temp_c = st.number_input(
            "Average ambient temperature (°C)", value=25.0
        )

        humidity_pct = st.slider(
            "Humidity (%)", 0.0, 100.0, 50.0
        )

        altitude_m = st.number_input(
            "Altitude (m)", value=200.0
        )

        internal_resistance_mohm = st.number_input(
            "Internal resistance (mΩ)", min_value=0.0, value=2.5
        )

    with c3:
        voltage_avg_v = st.number_input(
            "Average voltage (V)", min_value=0.0, value=350.0
        )

        current_avg_a = st.number_input(
            "Average current (A)", value=80.0
        )

        charge_time_avg_min = st.number_input(
            "Average charge time (min)", min_value=0.0, value=45.0
        )

        peak_temp_during_fast_charge_c = st.number_input(
            "Peak fast-charge temperature (°C)", value=42.0
        )

        capacity_fade_pct = st.slider(
            "Capacity fade (%)", 0.0, 100.0, 8.0
        )

        service_count = st.number_input(
            "Service count", min_value=0, value=1
        )

        recall_flag = st.selectbox("Recall flag", [0, 1])

        state_of_health_pct = st.slider(
            "State of health (%)", 0.0, 100.0, 90.0
        )

        remaining_useful_life_cycles = st.number_input(
            "Remaining useful-life cycles", min_value=0, value=800
        )

    st.subheader("Dates")

    d1, d2, d3 = st.columns(3)

    with d1:
        manufacture_date = st.date_input(
            "Manufacture date",
            value=pd.Timestamp("2023-01-01")
        )

    with d2:
        observation_date = st.date_input(
            "Observation date",
            value=pd.Timestamp.today().date()
        )

    with d3:
        last_service_date = st.date_input(
            "Last service date",
            value=pd.Timestamp("2025-01-01")
        )

    submitted = st.form_submit_button(
        "Predict Failure Risk",
        type="primary"
    )


if submitted:
    input_data = {
        "chemistry": chemistry,
        "manufacturer": manufacturer,
        "bms_firmware_version": bms_firmware_version,
        "climate_zone": climate_zone,
        "degradation_category": degradation_category,
        "pack_capacity_kwh": pack_capacity_kwh,
        "battery_age_days": battery_age_days,
        "cycle_count": cycle_count,
        "avg_daily_distance_km": avg_daily_distance_km,
        "avg_charging_power_kw": avg_charging_power_kw,
        "fast_charge_ratio": fast_charge_ratio,
        "avg_depth_of_discharge_pct": avg_depth_of_discharge_pct,
        "avg_state_of_charge_pct": avg_state_of_charge_pct,
        "charge_cycles_per_month": charge_cycles_per_month,
        "total_energy_throughput_kwh": total_energy_throughput_kwh,
        "avg_ambient_temp_c": avg_ambient_temp_c,
        "humidity_pct": humidity_pct,
        "altitude_m": altitude_m,
        "internal_resistance_mohm": internal_resistance_mohm,
        "voltage_avg_v": voltage_avg_v,
        "current_avg_a": current_avg_a,
        "charge_time_avg_min": charge_time_avg_min,
        "peak_temp_during_fast_charge_c": peak_temp_during_fast_charge_c,
        "capacity_fade_pct": capacity_fade_pct,
        "service_count": service_count,
        "recall_flag": recall_flag,
        "state_of_health_pct": state_of_health_pct,
        "remaining_useful_life_cycles": remaining_useful_life_cycles,
    }

    dates = {
        "manufacture_date": manufacture_date,
        "observation_date": observation_date,
        "last_service_date": last_service_date,
    }

    for name, date_value in dates.items():
        input_data[f"{name}_Year"] = date_value.year
        input_data[f"{name}_Month"] = date_value.month
        input_data[f"{name}_Day"] = date_value.day

    input_df = pd.DataFrame([input_data])

    try:
        # Find the original training feature names.
        expected_columns = getattr(model, "feature_names_in_", None)

        # Some fitted pipelines store feature names in their preprocessor.
        if expected_columns is None and hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected_columns = step.feature_names_in_
                    break

        if expected_columns is not None:
            expected_columns = list(expected_columns)

            missing = [
                col for col in expected_columns
                if col not in input_df.columns
            ]

            if missing:
                st.error("Missing features required by your trained model:")
                st.code("\n".join(missing))
                st.info(
                    "Add these exact columns to the input form and input_data. "
                    "Do not create arbitrary values for unknown features."
                )
                st.stop()

            input_df = input_df[expected_columns]

        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Failure Risk: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            classes = getattr(model, "classes_", None)

            if classes is None and hasattr(model, "named_steps"):
                for step in reversed(list(model.named_steps.values())):
                    if hasattr(step, "classes_"):
                        classes = step.classes_
                        break

            if classes is not None:
                probability_df = pd.DataFrame({
                    "Risk Category": classes,
                    "Probability (%)": (probabilities * 100).round(2)
                }).sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.subheader("Prediction Probabilities")
                st.bar_chart(
                    probability_df.set_index("Risk Category")
                )
                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True
                )

        st.caption(
            "This is a model prediction, not a confirmed battery diagnosis."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Verify that the saved model is fitted and that all input "
            "features match the training dataset."
        )