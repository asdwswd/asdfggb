import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="기온 예측기", layout="centered")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 이용해 연도별 평균기온의 추세를 분석하고, 특정 연도의 예상 기온을 예측합니다.")

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year
    return df

df = load_data()

# ---------------------------
# 2. 연도별 평균기온 및 관측일수 계산
# ---------------------------
yearly = df.groupby("연도").agg(
    평균기온=("평균기온", "mean"),
    관측일수=("평균기온", "count")
).reset_index()

# 조건: 2025년까지, 관측일수 300일 이상
yearly_filtered = yearly[
    (yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)
].copy()

yearly_filtered = yearly_filtered.sort_values("연도").reset_index(drop=True)

n_years = len(yearly_filtered)
start_year = int(yearly_filtered["연도"].min())
end_year = int(yearly_filtered["연도"].max())

# ---------------------------
# 3. 회귀 분석 (선형회귀)
# ---------------------------
x = yearly_filtered["연도"].values
y = yearly_filtered["평균기온"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

def predict_temp(year):
    return slope * year + intercept

# ---------------------------
# 4. 화면에 회귀 정보 표시
# ---------------------------
st.subheader("📊 회귀 분석에 사용된 데이터 정보")
col1, col2, col3 = st.columns(3)
col1.metric("사용된 연도 수", f"{n_years} 개")
col2.metric("시작 연도", f"{start_year} 년")
col3.metric("끝 연도", f"{end_year} 년")

st.write(f"**상관계수 (r)**: {r_value:.4f}  |  **결정계수 (R²)**: {r_squared:.4f}")
st.write(f"**회귀식**: 평균기온 = {slope:.5f} × 연도 + ({intercept:.2f})")

# ---------------------------
# 5. 산점도 + 회귀직선 (plotly)
# ---------------------------
st.subheader("📈 연도별 평균기온 산점도 및 회귀직선")

fig = go.Figure()

# 산점도
fig.add_trace(go.Scatter(
    x=yearly_filtered["연도"],
    y=yearly_filtered["평균기온"],
    mode="markers",
    name="연평균 기온 (실제)",
    marker=dict(color="royalblue", size=7)
))

# 회귀직선
x_line = np.linspace(start_year, end_year, 100)
y_line = predict_temp(x_line)

fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    name="회귀직선",
    line=dict(color="red", width=2)
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 6. 슬라이더로 연도 선택 후 예측
# ---------------------------
st.subheader("🔮 연도별 예상 기온 예측")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = predict_temp(selected_year)

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; border-radius: 15px; background-color:#f0f8ff; margin-top:20px;">
        <h2 style="color:#333;">{selected_year}년 예상 평균기온</h2>
        <h1 style="color:#d62728; font-size:60px;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 참고: 회귀 범위 밖 연도는 외삽(extrapolation)임을 안내
if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"⚠️ 선택한 연도({selected_year})는 실제 데이터 범위({start_year}~{end_year}) 밖입니다. "
        "이 예측값은 추세선을 이용한 외삽(extrapolation) 결과이므로 참고용으로만 활용하세요."
    )

# ---------------------------
# 7. 원본 데이터 확인 (선택 사항)
# ---------------------------
with st.expander("📄 필터링된 연도별 데이터 확인하기"):
    st.dataframe(yearly_filtered)
