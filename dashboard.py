import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(page_title="EOT Crane Predictive Maintenance", layout="wide")

# ------------------------------------------------
# DARK INDUSTRIAL THEME
# ------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #2c2f36;
}

[data-testid="stMetricLabel"] {
    color: #9aa0a6;
}

[data-testid="stMetricValue"] {
    color: #00E5FF;
    font-size: 28px;
}

h1, h2, h3 {
    color: #EAEAEA;
}

</style>
""", unsafe_allow_html=True)

st.title("EOT Crane Predictive Maintenance Dashboard")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_excel("cleaned_wheel_data.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

# ------------------------------------------------
# WEAR TYPE DETECTION
# ------------------------------------------------

def detect_wear(desc):

    desc = str(desc).lower()

    if "broken" in desc:
        return "Collar Broken"
    elif "reduced" in desc:
        return "Collar Wear"
    else:
        return "Other"

df["Wear_Type"] = df["Job Description"].apply(detect_wear)

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.title("Filters")

crane_filter = st.sidebar.multiselect(
    "Crane",
    df["Crane"].unique(),
    default=df["Crane"].unique()
)

equipment_filter = st.sidebar.multiselect(
    "Wheel Position",
    df["Equipment"].unique(),
    default=df["Equipment"].unique()
)

filtered_df = df[
    (df["Crane"].isin(crane_filter)) &
    (df["Equipment"].isin(equipment_filter))
]

if filtered_df.empty:
    st.error("No data for selected filters")
    st.stop()

# ------------------------------------------------
# WHEEL LIFE CALCULATION
# ------------------------------------------------

sorted_df = filtered_df.sort_values("Date")
sorted_df["Wheel_Life"] = sorted_df["Date"].diff().dt.days

wheel_life = sorted_df["Wheel_Life"].dropna()
wheel_life = wheel_life[wheel_life > 0]

MTBF = wheel_life.mean() if len(wheel_life) > 0 else 0
failure_rate = 1 / MTBF if MTBF > 0 else 0

# ------------------------------------------------
# FAILURE PREDICTION DATE
# ------------------------------------------------

last_date = sorted_df["Date"].max()
predicted_failure = last_date + pd.Timedelta(days=MTBF)

days_remaining = (predicted_failure - pd.Timestamp.today()).days

# ------------------------------------------------
# KPI PANELS
# ------------------------------------------------

st.subheader("System Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Replacements", len(filtered_df))

col2.metric(
    "MTBF (Days)",
    round(MTBF,1),
    delta=f"{round(MTBF-60,1)} vs baseline"
)

col3.metric(
    "Failure Rate",
    round(failure_rate,4)
)

col4.metric(
    "Next Predicted Failure",
    predicted_failure.strftime("%d %b %Y"),
    delta=f"{days_remaining} days remaining"
)

# ------------------------------------------------
# CRANE HEALTH SCORE
# ------------------------------------------------

TARGET_MTBF = 120
FAILURE_THRESHOLD = 20

mtbf_score = min((MTBF / TARGET_MTBF) * 100, 100)
failure_score = max(100 - (len(filtered_df) / FAILURE_THRESHOLD) * 100, 0)

wear_count = filtered_df["Wear_Type"].value_counts()
wear_ratio = wear_count.get("Collar Wear", 0) / len(filtered_df)

wear_score = (1 - wear_ratio) * 100

health_score = round(
    0.5 * mtbf_score +
    0.3 * failure_score +
    0.2 * wear_score
)

st.metric("Crane Health Score", f"{health_score}%")

# ------------------------------------------------
# HEALTH GAUGE
# ------------------------------------------------

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "Crane Health Index"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "#00E5FF"},
        'steps': [
            {'range':[0,50],'color':'#ff4b4b'},
            {'range':[50,75],'color':'#ffa500'},
            {'range':[75,100],'color':'#00ff88'}
        ],
    }
))

fig_gauge.update_layout(template="plotly_dark")

st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------
# REPLACEMENT TREND
# ------------------------------------------------

filtered_df["Month"] = filtered_df["Date"].dt.to_period("M").astype(str)

monthly = filtered_df.groupby("Month").size().reset_index(name="Replacements")

fig1 = px.bar(
    monthly,
    x="Month",
    y="Replacements",
    template="plotly_dark",
    color="Replacements",
    title="Wheel Replacement Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# WHEEL LIFE TREND
# ------------------------------------------------

fig2 = px.line(
    sorted_df,
    x="Date",
    y="Wheel_Life",
    markers=True,
    template="plotly_dark",
    title="Wheel Life Trend"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# RELIABILITY CURVE
# ------------------------------------------------

if MTBF > 0:

    time = np.linspace(0,150,100)
    reliability = np.exp(-failure_rate*time)

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=time,
        y=reliability,
        mode="lines"
    ))

    fig3.update_layout(
        template="plotly_dark",
        title="Wheel Reliability Curve",
        xaxis_title="Operating Time (Days)",
        yaxis_title="Reliability"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# FAILURE TYPE
# ------------------------------------------------

wear_counts = filtered_df["Wear_Type"].value_counts().reset_index()
wear_counts.columns = ["Wear_Type","Count"]

fig4 = px.pie(
    wear_counts,
    values="Count",
    names="Wear_Type",
    template="plotly_dark",
    title="Failure Type Distribution"
)

st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# WHEEL POSITION FAILURE
# ------------------------------------------------

equip_counts = filtered_df["Equipment"].value_counts().reset_index()
equip_counts.columns = ["Equipment","Failures"]

fig5 = px.bar(
    equip_counts,
    x="Equipment",
    y="Failures",
    template="plotly_dark",
    color="Failures",
    title="Wheel Position Failure Frequency"
)

st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------
# HARDNESS CORRELATION MODEL
# ------------------------------------------------

if "Rail_Hardness" in filtered_df.columns:

    hardness_df = sorted_df.dropna(subset=["Rail_Hardness","Wheel_Life"])

    if len(hardness_df) > 2:

        X = hardness_df[["Rail_Hardness"]]
        y = hardness_df["Wheel_Life"]

        model = LinearRegression()
        model.fit(X,y)

        fig6 = px.scatter(
            hardness_df,
            x="Rail_Hardness",
            y="Wheel_Life",
            trendline="ols",
            template="plotly_dark",
            title="Rail Hardness vs Wheel Life"
        )

        st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------
# FAILURE FORECAST (NEXT 6 MONTHS)
# ------------------------------------------------

st.subheader("Failure Forecast (Next 6 Months)")

if MTBF > 0:

    monthly_fail_rate = 30 / MTBF

    future_months = pd.date_range(start=last_date, periods=7, freq="M")[1:]

    forecast = []

    for i, m in enumerate(future_months):
        expected = round((i+1) * monthly_fail_rate)
        forecast.append(expected)

    forecast_df = pd.DataFrame({
        "Month": future_months.strftime("%Y-%m"),
        "Predicted Failures": forecast
    })

    fig7 = px.bar(
        forecast_df,
        x="Month",
        y="Predicted Failures",
        template="plotly_dark",
        color="Predicted Failures",
        title="Predicted Wheel Failures"
    )

    st.plotly_chart(fig7, use_container_width=True)

# ------------------------------------------------
# ROOT CAUSE INSIGHT
# ------------------------------------------------

st.subheader("Automated Engineering Insight")

most_failed_wheel = equip_counts.iloc[0]["Equipment"]
most_crane = filtered_df["Crane"].value_counts().idxmax()

st.markdown(f"""
Most affected wheel position: **{most_failed_wheel}**

Crane with highest wear: **{most_crane}**

Predicted next wheel replacement around **{predicted_failure.strftime("%d %b %Y")}**

Wear pattern suggests influence of rail hardness,
rail wear profile, or wheel-rail misalignment.
""")
