import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="글로벌 주식 대시보드",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------
# 주요 글로벌 종목
# --------------------------------------------------
GLOBAL_STOCKS = {
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "엔비디아": "NVDA",
    "아마존": "AMZN",
    "구글": "GOOGL",
    "메타": "META",
    "테슬라": "TSLA",
    "브로드컴": "AVGO",
    "JP모건": "JPM",
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대자동차": "005380.KS",
    "NAVER": "035420.KS",
    "도요타": "7203.T",
    "소니": "6758.T",
    "TSMC": "TSM",
    "알리바바": "BABA",
    "텐센트": "0700.HK",
    "ASML": "ASML",
    "SAP": "SAP",
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


# --------------------------------------------------
# 데이터 조회 함수
# --------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_stock_data(ticker, period, interval):
    """
    Yahoo Finance 주가 데이터를 조회합니다.
    요청 제한이 발생하면 잠시 기다린 뒤 재시도합니다.
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
                raise ValueError("주가 데이터가 비어 있습니다.")

            # yfinance 버전에 따라 다중 인덱스가 반환될 수 있습니다.
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs(ticker, axis=1, level=1)
                except Exception:
                    data.columns = data.columns.get_level_values(0)

            data = data.copy()
            data.index = pd.to_datetime(data.index)

            if getattr(data.index, "tz", None) is not None:
                data.index = data.index.tz_localize(None)

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            for column in required_columns:
                if column not in data.columns:
                    data[column] = 0

            data = data.dropna(subset=["Close"])

            if data.empty:
                raise ValueError("유효한 종가 데이터가 없습니다.")

            return data

        except Exception as error:
            last_error = error

            # 1차 실패 후 3초, 2차 실패 후 6초 기다립니다.
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    raise RuntimeError(str(last_error))


@st.cache_data(ttl=1800, show_spinner=False)
def load_comparison_data(ticker_items, period):
    """
    여러 종목을 한 번의 요청으로 조회합니다.
    """

    ticker_names = list(ticker_items.keys())
    ticker_symbols = list(ticker_items.values())

    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=ticker_symbols,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                timeout=20,
            )

            if data is None or data.empty:
                raise ValueError("비교 데이터가 비어 있습니다.")

            result = pd.DataFrame()

            if len(ticker_symbols) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    close_data = data["Close"]

                    if isinstance(close_data, pd.DataFrame):
                        close_series = close_data.iloc[:, 0]
                    else:
                        close_series = close_data
                else:
                    close_series = data["Close"]

                result[ticker_names[0]] = close_series

            else:
                close_data = data["Close"]

                for name, symbol in ticker_items.items():
                    if symbol in close_data.columns:
                        result[name] = close_data[symbol]

            result.index = pd.to_datetime(result.index)

            if getattr(result.index, "tz", None) is not None:
                result.index = result.index.tz_localize(None)

            return result.dropna(how="all")

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    raise RuntimeError(str(last_error))


# --------------------------------------------------
# 계산 함수
# --------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    relative_strength = average_gain / average_loss.replace(0, pd.NA)

    return 100 - (100 / (1 + relative_strength))


def format_price(value):
    if value is None or pd.isna(value):
        return "-"

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    return f"{value:,.2f}"


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


# --------------------------------------------------
# 차트 함수
# --------------------------------------------------
def create_price_chart(data, ticker, chart_type, show_ma):
    chart_data = data.copy()

    chart_data["MA20"] = chart_data["Close"].rolling(20).mean()
    chart_data["MA60"] = chart_data["Close"].rolling(60).mean()

    figure = go.Figure()

    if chart_type == "캔들 차트":
        figure.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name=ticker,
            )
        )

    else:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"],
                mode="lines",
                name="종가",
                line=dict(width=2),
            )
        )

    if show_ma:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["MA20"],
                mode="lines",
                name="20일 이동평균",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["MA60"],
                mode="lines",
                name="60일 이동평균",
            )
        )

    figure.update_layout(
        title=f"{ticker} 주가",
        height=560,
        xaxis_title="날짜",
        yaxis_title="가격",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=60, b=20),
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
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="거래량",
        )
    )

    figure.update_layout(
        title="거래량",
        height=320,
        xaxis_title="날짜",
        yaxis_title="거래량",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    return figure


def create_rsi_chart(data):
    rsi = calculate_rsi(data["Close"])

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi,
            mode="lines",
            name="RSI",
        )
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="과매수",
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="과매도",
    )

    figure.update_layout(
        title="RSI",
        height=320,
        xaxis_title="날짜",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    return figure


# --------------------------------------------------
# 세션 상태
# --------------------------------------------------
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "loaded_ticker" not in st.session_state:
    st.session_state.loaded_ticker = None

if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None


# --------------------------------------------------
# 화면 제목
# --------------------------------------------------
st.title("🌍 글로벌 주요 주식 대시보드")
st.caption("Yahoo Finance 데이터를 이용한 글로벌 주식 분석 대시보드")


# --------------------------------------------------
# 사이드바
# --------------------------------------------------
with st.sidebar:
    st.header("조회 설정")

    selected_name = st.selectbox(
        "종목 선택",
        list(GLOBAL_STOCKS.keys()),
    )

    selected_ticker = GLOBAL_STOCKS[selected_name]

    custom_ticker = st.text_input(
        "직접 티커 입력",
        placeholder="예: AMD, NFLX, 005930.KS",
    ).strip().upper()

    ticker = custom_ticker if custom_ticker else selected_ticker

    period_label = st.selectbox(
        "조회 기간",
        list(PERIOD_OPTIONS.keys()),
        index=3,
    )

    interval_label = st.selectbox(
        "차트 간격",
        list(INTERVAL_OPTIONS.keys()),
    )

    chart_type = st.radio(
        "차트 유형",
        ["캔들 차트", "라인 차트"],
    )

    show_ma = st.checkbox(
        "이동평균선 표시",
        value=True,
    )

    load_button = st.button(
        "주가 데이터 불러오기",
        type="primary",
        use_container_width=True,
    )

    if st.button(
        "캐시 초기화",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.session_state.stock_data = None
        st.session_state.comparison_data = None
        st.success("캐시가 초기화되었습니다.")


# --------------------------------------------------
# 선택 종목 조회
# --------------------------------------------------
if load_button:
    with st.spinner("주가 데이터를 불러오는 중입니다..."):
        try:
            stock_data = load_stock_data(
                ticker,
                PERIOD_OPTIONS[period_label],
                INTERVAL_OPTIONS[interval_label],
            )

            st.session_state.stock_data = stock_data
            st.session_state.loaded_ticker = ticker

        except Exception as error:
            error_text = str(error)

            if (
                "Too Many Requests" in error_text
                or "Rate limited" in error_text
                or "429" in error_text
            ):
                st.error(
                    "Yahoo Finance의 요청 제한에 걸렸습니다. "
                    "잠시 후 다시 시도하거나 캐시 초기화 없이 "
                    "기존 데이터를 이용해 주세요."
                )
            else:
                st.error(f"데이터를 불러오지 못했습니다: {error_text}")


# --------------------------------------------------
# 종목 데이터 출력
# --------------------------------------------------
if st.session_state.stock_data is None:
    st.info(
        "왼쪽에서 종목을 선택한 후 "
        "'주가 데이터 불러오기' 버튼을 눌러주세요."
    )

else:
    data = st.session_state.stock_data
    loaded_ticker = st.session_state.loaded_ticker

    latest_close = float(data["Close"].iloc[-1])
    previous_close = (
        float(data["Close"].iloc[-2])
        if len(data) >= 2
        else latest_close
    )

    change = latest_close - previous_close
    change_percent = (
        change / previous_close * 100
        if previous_close != 0
        else 0
    )

    period_start = float(data["Close"].iloc[0])
    period_return = (
        (latest_close - period_start) / period_start * 100
        if period_start != 0
        else 0
    )

    columns = st.columns(5)

    columns[0].metric(
        "현재 가격",
        format_price(latest_close),
        f"{change_percent:+.2f}%",
    )

    columns[1].metric(
        "기간 수익률",
        f"{period_return:+.2f}%",
    )

    columns[2].metric(
        "기간 최고가",
        format_price(data["High"].max()),
    )

    columns[3].metric(
        "기간 최저가",
        format_price(data["Low"].min()),
    )

    columns[4].metric(
        "최근 거래량",
        format_volume(data["Volume"].iloc[-1]),
    )

    price_tab, indicator_tab, data_tab = st.tabs(
        ["주가 차트", "기술적 지표", "원본 데이터"]
    )

    with price_tab:
        price_figure = create_price_chart(
            data,
            loaded_ticker,
            chart_type,
            show_ma,
        )

        st.plotly_chart(
            price_figure,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with indicator_tab:
        left_column, right_column = st.columns(2)

        with left_column:
            st.plotly_chart(
                create_volume_chart(data),
                use_container_width=True,
                config={"displaylogo": False},
            )

        with right_column:
            st.plotly_chart(
                create_rsi_chart(data),
                use_container_width=True,
                config={"displaylogo": False},
            )

    with data_tab:
        display_data = data.copy()
        display_data = display_data.sort_index(ascending=False)
        display_data.index = display_data.index.strftime("%Y-%m-%d")
        display_data.index.name = "날짜"

        st.dataframe(
            display_data,
            use_container_width=True,
        )

        csv_data = display_data.to_csv().encode("utf-8-sig")

        st.download_button(
            "CSV 다운로드",
            data=csv_data,
            file_name=f"{loaded_ticker}_stock_data.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# 종목 비교
# --------------------------------------------------
st.divider()
st.subheader("📊 종목 수익률 비교")

comparison_names = st.multiselect(
    "비교할 종목을 선택하세요",
    list(GLOBAL_STOCKS.keys()),
    default=["애플", "마이크로소프트", "엔비디아"],
    max_selections=5,
)

comparison_period_label = st.selectbox(
    "비교 기간",
    list(PERIOD_OPTIONS.keys()),
    index=3,
    key="comparison_period",
)

compare_button = st.button(
    "비교 데이터 불러오기",
    use_container_width=True,
)

if compare_button:
    if not comparison_names:
        st.warning("비교할 종목을 하나 이상 선택해 주세요.")

    else:
        ticker_items = {
            name: GLOBAL_STOCKS[name]
            for name in comparison_names
        }

        with st.spinner("비교 데이터를 불러오는 중입니다..."):
            try:
                comparison_data = load_comparison_data(
                    ticker_items,
                    PERIOD_OPTIONS[comparison_period_label],
                )

                st.session_state.comparison_data = comparison_data

            except Exception as error:
                error_text = str(error)

                if (
                    "Too Many Requests" in error_text
                    or "Rate limited" in error_text
                    or "429" in error_text
                ):
                    st.error(
                        "Yahoo Finance 요청 제한에 걸렸습니다. "
                        "잠시 후 다시 시도해 주세요."
                    )
                else:
                    st.error(
                        f"비교 데이터를 불러오지 못했습니다: {error_text}"
                    )


if st.session_state.comparison_data is not None:
    comparison_data = st.session_state.comparison_data.copy()

    normalized_data = pd.DataFrame(index=comparison_data.index)

    for column in comparison_data.columns:
        valid_data = comparison_data[column].dropna()

        if not valid_data.empty and valid_data.iloc[0] != 0:
            normalized_data[column] = (
                comparison_data[column] / valid_data.iloc[0]
            ) * 100

    comparison_figure = go.Figure()

    for column in normalized_data.columns:
        comparison_figure.add_trace(
            go.Scatter(
                x=normalized_data.index,
                y=normalized_data[column],
                mode="lines",
                name=column,
            )
        )

    comparison_figure.add_hline(
        y=100,
        line_dash="dash",
    )

    comparison_figure.update_layout(
        title="종목별 상대 수익률 · 시작점 100",
        height=520,
        xaxis_title="날짜",
        yaxis_title="상대 지수",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )


st.divider()
st.caption(
    "본 대시보드는 학습 및 정보 제공 목적입니다. "
    "Yahoo Finance 데이터는 지연되거나 일시적으로 조회가 제한될 수 있습니다."
)
