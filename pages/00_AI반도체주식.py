import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="AI 반도체 전문 분석",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# AI 반도체 종목 정보
# 종목 목록과 설명은 앱 내부에서 직접 관리합니다.
# =========================================================
AI_SEMICONDUCTOR_STOCKS = {
    "NVIDIA": {
        "ticker": "NVDA",
        "country": "미국",
        "category": "AI GPU",
        "description": "데이터센터용 AI GPU 및 가속 컴퓨팅 플랫폼",
    },
    "AMD": {
        "ticker": "AMD",
        "country": "미국",
        "category": "AI GPU·CPU",
        "description": "데이터센터 CPU와 AI 가속기 사업",
    },
    "Broadcom": {
        "ticker": "AVGO",
        "country": "미국",
        "category": "AI ASIC·네트워크",
        "description": "맞춤형 AI 가속기, 네트워크 및 통신 반도체",
    },
    "Marvell": {
        "ticker": "MRVL",
        "country": "미국",
        "category": "AI 네트워크·ASIC",
        "description": "데이터센터 연결 및 맞춤형 AI 반도체",
    },
    "Micron": {
        "ticker": "MU",
        "country": "미국",
        "category": "AI 메모리",
        "description": "HBM과 데이터센터용 DRAM·낸드",
    },
    "TSMC": {
        "ticker": "TSM",
        "country": "대만",
        "category": "AI 파운드리",
        "description": "첨단 AI 반도체 위탁생산",
    },
    "ASML": {
        "ticker": "ASML",
        "country": "네덜란드",
        "category": "반도체 장비",
        "description": "첨단 반도체 생산용 EUV 노광 장비",
    },
    "Arm Holdings": {
        "ticker": "ARM",
        "country": "영국",
        "category": "AI 설계자산",
        "description": "CPU 설계자산과 데이터센터 컴퓨팅 아키텍처",
    },
    "Applied Materials": {
        "ticker": "AMAT",
        "country": "미국",
        "category": "반도체 장비",
        "description": "반도체 증착·식각·공정 장비",
    },
    "Lam Research": {
        "ticker": "LRCX",
        "country": "미국",
        "category": "반도체 장비",
        "description": "메모리 및 첨단 반도체 식각·증착 장비",
    },
    "KLA": {
        "ticker": "KLAC",
        "country": "미국",
        "category": "반도체 장비",
        "description": "반도체 공정 검사 및 계측 장비",
    },
    "Synopsys": {
        "ticker": "SNPS",
        "country": "미국",
        "category": "EDA",
        "description": "AI 반도체 설계 자동화 소프트웨어",
    },
    "Cadence": {
        "ticker": "CDNS",
        "country": "미국",
        "category": "EDA",
        "description": "반도체 및 시스템 설계 자동화 소프트웨어",
    },
    "삼성전자": {
        "ticker": "005930.KS",
        "country": "한국",
        "category": "AI 메모리·파운드리",
        "description": "HBM·메모리·파운드리 사업",
    },
    "SK하이닉스": {
        "ticker": "000660.KS",
        "country": "한국",
        "category": "AI 메모리",
        "description": "AI 가속기용 HBM과 서버 DRAM",
    },
    "한미반도체": {
        "ticker": "042700.KS",
        "country": "한국",
        "category": "HBM 장비",
        "description": "HBM 패키징 관련 반도체 장비",
    },
}


BENCHMARKS = {
    "필라델피아 반도체지수": "^SOX",
    "나스닥 100": "^NDX",
    "S&P 500": "^GSPC",
}


PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
}


INTERVAL_OPTIONS = {
    "일봉": "1d",
    "주봉": "1wk",
    "월봉": "1mo",
}


# =========================================================
# 데이터 조회 함수
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def download_single_stock(ticker, period, interval):
    """
    한 종목의 OHLCV 데이터를 조회합니다.
    조회 결과는 30분 동안 캐시됩니다.
    """

    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )

            if data is None or data.empty:
                raise ValueError("Yahoo Finance에서 빈 데이터가 반환되었습니다.")

            data = normalize_single_ticker_data(data, ticker)

            if data.empty:
                raise ValueError("유효한 주가 데이터가 없습니다.")

            return data

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(str(last_error))


@st.cache_data(ttl=1800, show_spinner=False)
def download_multiple_stocks(ticker_tuple, period):
    """
    여러 종목의 수정 종가를 한 번의 요청으로 조회합니다.
    """

    tickers = list(ticker_tuple)
    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                timeout=30,
            )

            if data is None or data.empty:
                raise ValueError("비교 데이터가 비어 있습니다.")

            close_data = extract_close_data(data, tickers)
            close_data = clean_datetime_index(close_data)

            if close_data.empty:
                raise ValueError("유효한 종가 데이터가 없습니다.")

            return close_data

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(str(last_error))


# =========================================================
# 데이터 정리 함수
# =========================================================
def normalize_single_ticker_data(data, ticker):
    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        ticker_levels = result.columns.get_level_values(-1)

        if ticker in ticker_levels:
            try:
                result = result.xs(
                    ticker,
                    axis=1,
                    level=-1,
                    drop_level=True,
                )
            except Exception:
                result.columns = result.columns.get_level_values(0)
        else:
            result.columns = result.columns.get_level_values(0)

    result = result.loc[:, ~result.columns.duplicated()]
    result = clean_datetime_index(result)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in result.columns:
            result[column] = 0

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(subset=["Close"])

    return result[required_columns]


def extract_close_data(data, tickers):
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Close" in first_level:
            close_data = data["Close"].copy()
        else:
            second_level = data.columns.get_level_values(1)

            if "Close" in second_level:
                close_data = data.xs(
                    "Close",
                    axis=1,
                    level=1,
                ).copy()
            else:
                raise ValueError("종가 열을 찾을 수 없습니다.")

        if isinstance(close_data, pd.Series):
            close_data = close_data.to_frame(name=tickers[0])

        return close_data

    if "Close" not in data.columns:
        raise ValueError("종가 열을 찾을 수 없습니다.")

    close_data = data[["Close"]].copy()
    close_data.columns = [tickers[0]]

    return close_data


def clean_datetime_index(data):
    result = data.copy()
    result.index = pd.to_datetime(result.index)

    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    return result.sort_index()


# =========================================================
# 기술적 지표 계산
# =========================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))

    return 100 - (100 / (1 + relative_strength))


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    return macd, signal, histogram


def calculate_bollinger_bands(close, period=20):
    middle = close.rolling(period).mean()
    standard_deviation = close.rolling(period).std()

    upper = middle + (standard_deviation * 2)
    lower = middle - (standard_deviation * 2)

    return middle, upper, lower


def calculate_max_drawdown(close):
    if close.empty:
        return float("nan")

    cumulative_max = close.cummax()
    drawdown = (close / cumulative_max) - 1

    return float(drawdown.min() * 100)


def calculate_annualized_volatility(close):
    returns = close.pct_change().dropna()

    if returns.empty:
        return float("nan")

    return float(returns.std() * (252 ** 0.5) * 100)


def calculate_period_return(series):
    valid = series.dropna()

    if len(valid) < 2 or valid.iloc[0] == 0:
        return float("nan")

    return float((valid.iloc[-1] / valid.iloc[0] - 1) * 100)


def add_indicators(data):
    result = data.copy()

    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA60"] = result["Close"].rolling(60).mean()
    result["MA120"] = result["Close"].rolling(120).mean()
    result["MA200"] = result["Close"].rolling(200).mean()

    result["RSI14"] = calculate_rsi(result["Close"])

    macd, signal, histogram = calculate_macd(result["Close"])
    result["MACD"] = macd
    result["MACD_SIGNAL"] = signal
    result["MACD_HIST"] = histogram

    middle, upper, lower = calculate_bollinger_bands(result["Close"])
    result["BB_MIDDLE"] = middle
    result["BB_UPPER"] = upper
    result["BB_LOWER"] = lower

    return result


# =========================================================
# 분석 함수
# =========================================================
def create_signal_analysis(data):
    """
    단순 기술적 상태를 점수화합니다.
    매수·매도 추천이 아니라 현재 추세를 요약하는 용도입니다.
    """

    latest = data.iloc[-1]
    score = 0
    reasons = []

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma60 = latest["MA60"]
    ma120 = latest["MA120"]
    rsi = latest["RSI14"]
    macd = latest["MACD"]
    macd_signal = latest["MACD_SIGNAL"]

    if pd.notna(ma20):
        if close > ma20:
            score += 1
            reasons.append("주가가 20일 이동평균 위")
        else:
            score -= 1
            reasons.append("주가가 20일 이동평균 아래")

    if pd.notna(ma60):
        if close > ma60:
            score += 1
            reasons.append("주가가 60일 이동평균 위")
        else:
            score -= 1
            reasons.append("주가가 60일 이동평균 아래")

    if pd.notna(ma20) and pd.notna(ma60):
        if ma20 > ma60:
            score += 1
            reasons.append("단기 이동평균이 중기 이동평균 상회")
        else:
            score -= 1
            reasons.append("단기 이동평균이 중기 이동평균 하회")

    if pd.notna(ma60) and pd.notna(ma120):
        if ma60 > ma120:
            score += 1
            reasons.append("중기 상승 추세")
        else:
            score -= 1
            reasons.append("중기 약세 추세")

    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal:
            score += 1
            reasons.append("MACD가 시그널선 상회")
        else:
            score -= 1
            reasons.append("MACD가 시그널선 하회")

    if pd.notna(rsi):
        if 50 <= rsi < 70:
            score += 1
            reasons.append("RSI가 긍정적 모멘텀 구간")
        elif rsi >= 70:
            reasons.append("RSI 과매수 주의 구간")
        elif rsi < 30:
            reasons.append("RSI 과매도 가능 구간")
        else:
            score -= 1
            reasons.append("RSI가 50 미만")

    if score >= 4:
        state = "강한 상승 추세"
    elif score >= 2:
        state = "상승 우위"
    elif score >= -1:
        state = "중립·혼조"
    elif score >= -3:
        state = "약세 우위"
    else:
        state = "강한 하락 추세"

    return {
        "score": score,
        "state": state,
        "reasons": reasons,
    }


def build_market_scanner(close_data, ticker_name_map):
    rows = []

    for ticker in close_data.columns:
        series = close_data[ticker].dropna()

        if len(series) < 2:
            continue

        daily_return = (
            (series.iloc[-1] / series.iloc[-2] - 1) * 100
            if series.iloc[-2] != 0
            else float("nan")
        )

        weekly_return = (
            (series.iloc[-1] / series.iloc[-6] - 1) * 100
            if len(series) >= 6 and series.iloc[-6] != 0
            else float("nan")
        )

        monthly_return = (
            (series.iloc[-1] / series.iloc[-22] - 1) * 100
            if len(series) >= 22 and series.iloc[-22] != 0
            else float("nan")
        )

        three_month_return = (
            (series.iloc[-1] / series.iloc[-64] - 1) * 100
            if len(series) >= 64 and series.iloc[-64] != 0
            else float("nan")
        )

        ma20 = series.rolling(20).mean().iloc[-1]
        ma60 = series.rolling(60).mean().iloc[-1]
        rsi = calculate_rsi(series).iloc[-1]

        trend = "중립"

        if pd.notna(ma20) and pd.notna(ma60):
            if series.iloc[-1] > ma20 > ma60:
                trend = "상승"
            elif series.iloc[-1] < ma20 < ma60:
                trend = "하락"

        rows.append(
            {
                "종목": ticker_name_map.get(ticker, ticker),
                "티커": ticker,
                "현재가": series.iloc[-1],
                "1일 수익률": daily_return,
                "1주 수익률": weekly_return,
                "1개월 수익률": monthly_return,
                "3개월 수익률": three_month_return,
                "RSI": rsi,
                "추세": trend,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 표시 형식 함수
# =========================================================
def format_price(value, ticker):
    if value is None or pd.isna(value):
        return "-"

    if ticker.endswith(".KS"):
        return f"₩{value:,.0f}"

    return f"${value:,.2f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


def format_volume(value):
    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.0f}"


# =========================================================
# 차트 생성 함수
# =========================================================
def create_candlestick_chart(data, ticker, show_bollinger):
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
            increasing_line_color="#ef4444",
            decreasing_line_color="#2563eb",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            mode="lines",
            name="MA 20",
            line=dict(width=1.4),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA60"],
            mode="lines",
            name="MA 60",
            line=dict(width=1.4),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA120"],
            mode="lines",
            name="MA 120",
            line=dict(width=1.4),
        )
    )

    if show_bollinger:
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_UPPER"],
                mode="lines",
                name="볼린저 상단",
                line=dict(width=1, dash="dot"),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_LOWER"],
                mode="lines",
                name="볼린저 하단",
                line=dict(width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.08)",
            )
        )

    figure.update_layout(
        title=f"{ticker} 주가 및 이동평균",
        height=620,
        xaxis_title="날짜",
        yaxis_title="가격",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return figure


def create_volume_chart(data):
    bar_colors = [
        "#ef4444" if close >= open_price else "#2563eb"
        for open_price, close in zip(data["Open"], data["Close"])
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            marker_color=bar_colors,
            name="거래량",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Volume"].rolling(20).mean(),
            mode="lines",
            name="20일 평균 거래량",
            yaxis="y",
        )
    )

    figure.update_layout(
        title="거래량",
        height=350,
        xaxis_title="날짜",
        yaxis_title="거래량",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )

    return figure


def create_rsi_chart(data):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI14"],
            mode="lines",
            name="RSI 14",
            line=dict(width=2),
        )
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="과매수 70",
    )

    figure.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="중립 50",
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="과매도 30",
    )

    figure.update_layout(
        title="RSI 상대강도지수",
        height=350,
        xaxis_title="날짜",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return figure


def create_macd_chart(data):
    histogram_colors = [
        "#ef4444" if value >= 0 else "#2563eb"
        for value in data["MACD_HIST"].fillna(0)
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD_HIST"],
            name="히스토그램",
            marker_color=histogram_colors,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            mode="lines",
            name="MACD",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_SIGNAL"],
            mode="lines",
            name="Signal",
        )
    )

    figure.add_hline(y=0, line_dash="dot")

    figure.update_layout(
        title="MACD",
        height=350,
        xaxis_title="날짜",
        yaxis_title="MACD",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )

    return figure


def create_relative_performance_chart(close_data, display_names):
    normalized = pd.DataFrame(index=close_data.index)

    for ticker in close_data.columns:
        valid = close_data[ticker].dropna()

        if valid.empty or valid.iloc[0] == 0:
            continue

        normalized[display_names.get(ticker, ticker)] = (
            close_data[ticker] / valid.iloc[0]
        ) * 100

    figure = go.Figure()

    for column in normalized.columns:
        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=column,
                line=dict(width=2),
            )
        )

    figure.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="시작점",
    )

    figure.update_layout(
        title="AI 반도체 종목 상대 수익률 · 시작점 100",
        height=600,
        xaxis_title="날짜",
        yaxis_title="상대 지수",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return figure


def create_return_bar_chart(scanner, return_column):
    chart_data = scanner.dropna(subset=[return_column]).copy()
    chart_data = chart_data.sort_values(return_column)

    colors = [
        "#ef4444" if value >= 0 else "#2563eb"
        for value in chart_data[return_column]
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data[return_column],
            y=chart_data["종목"],
            orientation="h",
            marker_color=colors,
            text=chart_data[return_column].map(
                lambda value: f"{value:+.2f}%"
            ),
            textposition="outside",
        )
    )

    figure.update_layout(
        title=f"{return_column} 순위",
        height=max(450, len(chart_data) * 34),
        xaxis_title="수익률",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=20, r=80, t=55, b=20),
    )

    return figure


# =========================================================
# 세션 상태
# =========================================================
if "ai_stock_data" not in st.session_state:
    st.session_state.ai_stock_data = None

if "ai_loaded_ticker" not in st.session_state:
    st.session_state.ai_loaded_ticker = None

if "ai_loaded_name" not in st.session_state:
    st.session_state.ai_loaded_name = None

if "ai_comparison_data" not in st.session_state:
    st.session_state.ai_comparison_data = None

if "ai_scanner_data" not in st.session_state:
    st.session_state.ai_scanner_data = None


# =========================================================
# 상단 제목
# =========================================================
st.title("🧠 AI 반도체 주식 전문 분석")
st.caption(
    "AI GPU, HBM, 파운드리, 네트워크, EDA 및 반도체 장비 기업을 "
    "기술적 지표와 상대 수익률로 분석합니다."
)

st.warning(
    "이 페이지의 추세 점수와 지표는 투자 추천이 아닙니다. "
    "Yahoo Finance 데이터는 지연되거나 일시적으로 조회가 제한될 수 있습니다."
)


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.header("분석 설정")

    selected_name = st.selectbox(
        "AI 반도체 종목",
        options=list(AI_SEMICONDUCTOR_STOCKS.keys()),
        index=0,
    )

    selected_info = AI_SEMICONDUCTOR_STOCKS[selected_name]
    selected_ticker = selected_info["ticker"]

    custom_ticker = st.text_input(
        "직접 티커 입력",
        placeholder="예: INTC, QCOM, 009150.KS",
        help="입력하면 위에서 선택한 종목 대신 해당 티커를 분석합니다.",
    ).strip().upper()

    ticker = custom_ticker if custom_ticker else selected_ticker
    display_name = custom_ticker if custom_ticker else selected_name

    period_label = st.selectbox(
        "조회 기간",
        options=list(PERIOD_OPTIONS.keys()),
        index=3,
    )

    interval_label = st.selectbox(
        "차트 간격",
        options=list(INTERVAL_OPTIONS.keys()),
        index=0,
    )

    show_bollinger = st.checkbox(
        "볼린저 밴드 표시",
        value=True,
    )

    load_button = st.button(
        "선택 종목 분석",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    if not custom_ticker:
        st.write(f"**분류:** {selected_info['category']}")
        st.write(f"**국가:** {selected_info['country']}")
        st.caption(selected_info["description"])

    st.divider()

    if st.button(
        "데이터 캐시 초기화",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.session_state.ai_stock_data = None
        st.session_state.ai_comparison_data = None
        st.session_state.ai_scanner_data = None
        st.success("캐시를 초기화했습니다.")


# =========================================================
# 단일 종목 조회
# =========================================================
if load_button:
    with st.spinner(f"{ticker} 데이터를 분석하고 있습니다..."):
        try:
            downloaded_data = download_single_stock(
                ticker=ticker,
                period=PERIOD_OPTIONS[period_label],
                interval=INTERVAL_OPTIONS[interval_label],
            )

            analyzed_data = add_indicators(downloaded_data)

            st.session_state.ai_stock_data = analyzed_data
            st.session_state.ai_loaded_ticker = ticker
            st.session_state.ai_loaded_name = display_name

        except Exception as error:
            error_text = str(error)

            if (
                "Too Many Requests" in error_text
                or "Rate limited" in error_text
                or "429" in error_text
            ):
                st.error(
                    "Yahoo Finance 요청 제한에 걸렸습니다. "
                    "캐시를 초기화하지 말고 잠시 후 다시 조회해 주세요."
                )
            else:
                st.error(f"데이터 조회에 실패했습니다: {error_text}")


# =========================================================
# 종목 상세 분석
# =========================================================
if st.session_state.ai_stock_data is None:
    st.info(
        "왼쪽에서 AI 반도체 종목을 선택한 후 "
        "‘선택 종목 분석’ 버튼을 눌러주세요."
    )

else:
    data = st.session_state.ai_stock_data
    loaded_ticker = st.session_state.ai_loaded_ticker
    loaded_name = st.session_state.ai_loaded_name

    latest_close = float(data["Close"].iloc[-1])

    previous_close = (
        float(data["Close"].iloc[-2])
        if len(data) >= 2
        else latest_close
    )

    daily_return = (
        (latest_close / previous_close - 1) * 100
        if previous_close != 0
        else float("nan")
    )

    period_return = calculate_period_return(data["Close"])
    volatility = calculate_annualized_volatility(data["Close"])
    max_drawdown = calculate_max_drawdown(data["Close"])

    latest_rsi = data["RSI14"].iloc[-1]
    latest_volume = data["Volume"].iloc[-1]

    signal_result = create_signal_analysis(data)

    st.subheader(f"{loaded_name} · {loaded_ticker}")

    st.caption(
        f"데이터 기준일: {data.index[-1].strftime('%Y-%m-%d')} · "
        f"마지막 조회: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    metric_columns = st.columns(7)

    metric_columns[0].metric(
        "현재 가격",
        format_price(latest_close, loaded_ticker),
        format_percent(daily_return),
    )

    metric_columns[1].metric(
        "기간 수익률",
        format_percent(period_return),
    )

    metric_columns[2].metric(
        "기간 최고가",
        format_price(data["High"].max(), loaded_ticker),
    )

    metric_columns[3].metric(
        "기간 최저가",
        format_price(data["Low"].min(), loaded_ticker),
    )

    metric_columns[4].metric(
        "연환산 변동성",
        format_percent(volatility),
    )

    metric_columns[5].metric(
        "최대 낙폭",
        format_percent(max_drawdown),
    )

    metric_columns[6].metric(
        "RSI",
        f"{latest_rsi:.1f}" if pd.notna(latest_rsi) else "-",
    )

    st.divider()

    signal_col, detail_col = st.columns([1, 2])

    with signal_col:
        st.markdown("### 기술적 추세 요약")

        st.metric(
            "추세 상태",
            signal_result["state"],
            f"점수 {signal_result['score']:+d}",
        )

        st.write(f"**최근 거래량:** {format_volume(latest_volume)}")

        if pd.notna(latest_rsi):
            if latest_rsi >= 70:
                st.warning("RSI가 과매수 주의 구간에 있습니다.")
            elif latest_rsi <= 30:
                st.info("RSI가 과매도 가능 구간에 있습니다.")
            else:
                st.success("RSI가 일반 범위에 있습니다.")

    with detail_col:
        st.markdown("### 판단 근거")

        for reason in signal_result["reasons"]:
            st.write(f"• {reason}")

    chart_tab, indicator_tab, return_tab, data_tab = st.tabs(
        [
            "주가 차트",
            "기술적 지표",
            "수익률 분석",
            "원본 데이터",
        ]
    )

    with chart_tab:
        candlestick_chart = create_candlestick_chart(
            data=data,
            ticker=loaded_ticker,
            show_bollinger=show_bollinger,
        )

        st.plotly_chart(
            candlestick_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.plotly_chart(
            create_volume_chart(data),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with indicator_tab:
        left_chart, right_chart = st.columns(2)

        with left_chart:
            st.plotly_chart(
                create_rsi_chart(data),
                use_container_width=True,
                config={"displaylogo": False},
            )

        with right_chart:
            st.plotly_chart(
                create_macd_chart(data),
                use_container_width=True,
                config={"displaylogo": False},
            )

        indicator_columns = st.columns(5)

        indicator_columns[0].metric(
            "MA 20",
            format_price(data["MA20"].iloc[-1], loaded_ticker),
        )

        indicator_columns[1].metric(
            "MA 60",
            format_price(data["MA60"].iloc[-1], loaded_ticker),
        )

        indicator_columns[2].metric(
            "MA 120",
            format_price(data["MA120"].iloc[-1], loaded_ticker),
        )

        indicator_columns[3].metric(
            "MACD",
            (
                f"{data['MACD'].iloc[-1]:.2f}"
                if pd.notna(data["MACD"].iloc[-1])
                else "-"
            ),
        )

        indicator_columns[4].metric(
            "MACD Signal",
            (
                f"{data['MACD_SIGNAL'].iloc[-1]:.2f}"
                if pd.notna(data["MACD_SIGNAL"].iloc[-1])
                else "-"
            ),
        )

    with return_tab:
        return_periods = {
            "1일": 2,
            "1주": 6,
            "1개월": 22,
            "3개월": 64,
            "6개월": 127,
            "1년": 253,
        }

        return_rows = []

        for label, required_length in return_periods.items():
            if len(data) >= required_length:
                start_value = data["Close"].iloc[-required_length]
                end_value = data["Close"].iloc[-1]

                result = (
                    (end_value / start_value - 1) * 100
                    if start_value != 0
                    else float("nan")
                )
            else:
                result = float("nan")

            return_rows.append(
                {
                    "기간": label,
                    "수익률": result,
                }
            )

        return_frame = pd.DataFrame(return_rows)

        return_figure = go.Figure()

        return_figure.add_trace(
            go.Bar(
                x=return_frame["기간"],
                y=return_frame["수익률"],
                text=return_frame["수익률"].map(
                    lambda value: (
                        f"{value:+.2f}%"
                        if pd.notna(value)
                        else "-"
                    )
                ),
                textposition="outside",
                marker_color=[
                    "#ef4444"
                    if pd.notna(value) and value >= 0
                    else "#2563eb"
                    for value in return_frame["수익률"]
                ],
            )
        )

        return_figure.update_layout(
            title="기간별 수익률",
            height=450,
            xaxis_title="기간",
            yaxis_title="수익률",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(
            return_figure,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with data_tab:
        display_data = data.copy()
        display_data = display_data.sort_index(ascending=False)
        display_data.index = display_data.index.strftime("%Y-%m-%d")
        display_data.index.name = "날짜"

        selected_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA20",
            "MA60",
            "MA120",
            "RSI14",
            "MACD",
            "MACD_SIGNAL",
        ]

        st.dataframe(
            display_data[selected_columns].round(2),
            use_container_width=True,
        )

        csv_data = display_data.to_csv().encode("utf-8-sig")

        st.download_button(
            "CSV 데이터 다운로드",
            data=csv_data,
            file_name=f"{loaded_ticker}_ai_semiconductor_analysis.csv",
            mime="text/csv",
        )


# =========================================================
# AI 반도체 시장 스캐너
# =========================================================
st.divider()
st.header("🔎 AI 반도체 시장 스캐너")

st.write(
    "전체 AI 반도체 종목을 한 번에 조회해 단기·중기 수익률과 "
    "기술적 추세를 비교합니다."
)

scanner_period_label = st.selectbox(
    "스캐너 데이터 기간",
    options=["6개월", "1년", "2년"],
    index=1,
    key="scanner_period",
)

scanner_button = st.button(
    "전체 AI 반도체 종목 스캔",
    use_container_width=True,
)

if scanner_button:
    stock_tickers = tuple(
        item["ticker"]
        for item in AI_SEMICONDUCTOR_STOCKS.values()
    )

    ticker_name_map = {
        item["ticker"]: name
        for name, item in AI_SEMICONDUCTOR_STOCKS.items()
    }

    with st.spinner("AI 반도체 시장 전체를 조회하고 있습니다..."):
        try:
            scanner_close_data = download_multiple_stocks(
                ticker_tuple=stock_tickers,
                period=PERIOD_OPTIONS[scanner_period_label],
            )

            scanner_data = build_market_scanner(
                close_data=scanner_close_data,
                ticker_name_map=ticker_name_map,
            )

            st.session_state.ai_scanner_data = scanner_data

        except Exception as error:
            error_text = str(error)

            if (
                "Too Many Requests" in error_text
                or "Rate limited" in error_text
                or "429" in error_text
            ):
                st.error(
                    "Yahoo Finance 요청 제한에 걸렸습니다. "
                    "잠시 후 다시 실행해 주세요."
                )
            else:
                st.error(f"시장 스캔에 실패했습니다: {error_text}")


if st.session_state.ai_scanner_data is not None:
    scanner = st.session_state.ai_scanner_data.copy()

    if not scanner.empty:
        best_daily = scanner.loc[scanner["1일 수익률"].idxmax()]
        best_monthly = scanner.loc[scanner["1개월 수익률"].idxmax()]
        best_three_month = scanner.loc[
            scanner["3개월 수익률"].idxmax()
        ]

        rising_count = int((scanner["추세"] == "상승").sum())
        falling_count = int((scanner["추세"] == "하락").sum())

        scanner_metrics = st.columns(5)

        scanner_metrics[0].metric(
            "오늘 상승 1위",
            best_daily["종목"],
            format_percent(best_daily["1일 수익률"]),
        )

        scanner_metrics[1].metric(
            "1개월 상승 1위",
            best_monthly["종목"],
            format_percent(best_monthly["1개월 수익률"]),
        )

        scanner_metrics[2].metric(
            "3개월 상승 1위",
            best_three_month["종목"],
            format_percent(best_three_month["3개월 수익률"]),
        )

        scanner_metrics[3].metric(
            "상승 추세 종목",
            f"{rising_count}개",
        )

        scanner_metrics[4].metric(
            "하락 추세 종목",
            f"{falling_count}개",
        )

        scanner_tab, ranking_tab = st.tabs(
            ["스캐너 표", "수익률 순위"]
        )

        with scanner_tab:
            scanner_display = scanner.copy()

            percent_columns = [
                "1일 수익률",
                "1주 수익률",
                "1개월 수익률",
                "3개월 수익률",
            ]

            for column in percent_columns:
                scanner_display[column] = scanner_display[column].map(
                    format_percent
                )

            scanner_display["현재가"] = scanner.apply(
                lambda row: format_price(
                    row["현재가"],
                    row["티커"],
                ),
                axis=1,
            )

            scanner_display["RSI"] = scanner_display["RSI"].map(
                lambda value: (
                    f"{value:.1f}"
                    if pd.notna(value)
                    else "-"
                )
            )

            st.dataframe(
                scanner_display,
                use_container_width=True,
                hide_index=True,
            )

        with ranking_tab:
            ranking_period = st.radio(
                "순위 기준",
                options=[
                    "1일 수익률",
                    "1주 수익률",
                    "1개월 수익률",
                    "3개월 수익률",
                ],
                index=2,
                horizontal=True,
            )

            st.plotly_chart(
                create_return_bar_chart(
                    scanner=scanner,
                    return_column=ranking_period,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )


# =========================================================
# 종목 및 벤치마크 비교
# =========================================================
st.divider()
st.header("📊 종목·반도체지수 상대 수익률 비교")

comparison_names = st.multiselect(
    "비교할 AI 반도체 종목",
    options=list(AI_SEMICONDUCTOR_STOCKS.keys()),
    default=[
        "NVIDIA",
        "AMD",
        "Broadcom",
        "TSMC",
        "SK하이닉스",
    ],
    max_selections=8,
)

selected_benchmarks = st.multiselect(
    "비교 기준 지수",
    options=list(BENCHMARKS.keys()),
    default=["필라델피아 반도체지수"],
    max_selections=3,
)

comparison_period_label = st.selectbox(
    "비교 기간",
    options=list(PERIOD_OPTIONS.keys()),
    index=3,
    key="comparison_period",
)

comparison_button = st.button(
    "상대 수익률 비교 실행",
    use_container_width=True,
)

if comparison_button:
    selected_tickers = {
        AI_SEMICONDUCTOR_STOCKS[name]["ticker"]: name
        for name in comparison_names
    }

    selected_tickers.update(
        {
            BENCHMARKS[name]: name
            for name in selected_benchmarks
        }
    )

    if not selected_tickers:
        st.warning("비교할 종목이나 지수를 하나 이상 선택해 주세요.")

    else:
        with st.spinner("비교 데이터를 불러오고 있습니다..."):
            try:
                comparison_close_data = download_multiple_stocks(
                    ticker_tuple=tuple(selected_tickers.keys()),
                    period=PERIOD_OPTIONS[comparison_period_label],
                )

                st.session_state.ai_comparison_data = {
                    "data": comparison_close_data,
                    "names": selected_tickers,
                }

            except Exception as error:
                error_text = str(error)

                if (
                    "Too Many Requests" in error_text
                    or "Rate limited" in error_text
                    or "429" in error_text
                ):
                    st.error(
                        "Yahoo Finance 요청 제한에 걸렸습니다. "
                        "잠시 후 다시 실행해 주세요."
                    )
                else:
                    st.error(f"비교 데이터 조회에 실패했습니다: {error_text}")


if st.session_state.ai_comparison_data is not None:
    comparison_data = st.session_state.ai_comparison_data["data"]
    comparison_names_map = st.session_state.ai_comparison_data["names"]

    st.plotly_chart(
        create_relative_performance_chart(
            close_data=comparison_data,
            display_names=comparison_names_map,
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )

    comparison_summary = []

    for ticker_column in comparison_data.columns:
        series = comparison_data[ticker_column].dropna()

        if len(series) < 2:
            continue

        comparison_summary.append(
            {
                "종목": comparison_names_map.get(
                    ticker_column,
                    ticker_column,
                ),
                "티커": ticker_column,
                "기간 수익률": calculate_period_return(series),
                "연환산 변동성": calculate_annualized_volatility(series),
                "최대 낙폭": calculate_max_drawdown(series),
            }
        )

    summary_frame = pd.DataFrame(comparison_summary)

    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            "기간 수익률",
            ascending=False,
        )

        for column in [
            "기간 수익률",
            "연환산 변동성",
            "최대 낙폭",
        ]:
            summary_frame[column] = summary_frame[column].map(
                format_percent
            )

        st.dataframe(
            summary_frame,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 하단 안내
# =========================================================
st.divider()

st.caption(
    "데이터 출처: Yahoo Finance · 분석 목적: 교육 및 정보 제공 · "
    "기술적 지표는 과거 가격을 기반으로 하므로 미래 수익을 보장하지 않습니다."
)
