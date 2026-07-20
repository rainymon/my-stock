import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# --------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="글로벌 주식 대시보드",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------
# 글로벌 주요 종목
# --------------------------------------------------
GLOBAL_STOCKS = {
    # 미국
    "애플 (Apple)": "AAPL",
    "마이크로소프트 (Microsoft)": "MSFT",
    "엔비디아 (NVIDIA)": "NVDA",
    "아마존 (Amazon)": "AMZN",
    "알파벳 A (Google)": "GOOGL",
    "메타 (Meta)": "META",
    "테슬라 (Tesla)": "TSLA",
    "브로드컴 (Broadcom)": "AVGO",
    "버크셔 해서웨이 B": "BRK-B",
    "JP모건": "JPM",

    # 한국
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대자동차": "005380.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",

    # 일본
    "도요타": "7203.T",
    "소니": "6758.T",
    "소프트뱅크 그룹": "9984.T",
    "닌텐도": "7974.T",

    # 대만·중국·홍콩
    "TSMC": "TSM",
    "알리바바": "BABA",
    "텐센트": "0700.HK",
    "샤오미": "1810.HK",

    # 유럽
    "ASML": "ASML",
    "노보 노디스크": "NVO",
    "SAP": "SAP",
    "LVMH": "MC.PA",
    "페라리": "RACE",
}

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "나스닥 종합": "^IXIC",
    "다우존스": "^DJI",
    "러셀 2000": "^RUT",
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "일본 닛케이 225": "^N225",
    "홍콩 항셍": "^HSI",
    "중국 상하이 종합": "000001.SS",
    "영국 FTSE 100": "^FTSE",
    "독일 DAX": "^GDAXI",
    "프랑스 CAC 40": "^FCHI",
}

PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
    "10년": "10y",
    "전체": "max",
}

INTERVAL_OPTIONS = {
    "일봉": "1d",
    "주봉": "1wk",
    "월봉": "1mo",
}


# --------------------------------------------------
# 데이터 함수
# --------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker, period, interval):
    """
    Yahoo Finance에서 종목의 가격 데이터를 가져옵니다.
    데이터는 15분 동안 캐시됩니다.
    """
    stock = yf.Ticker(ticker)

    data = stock.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()
    data.index = pd.to_datetime(data.index)

    # 시간대 정보가 있으면 제거해 Plotly 호환성을 높입니다.
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in data.columns:
            data[column] = 0

    data = data.dropna(subset=["Close"])

    return data


@st.cache_data(ttl=900, show_spinner=False)
def load_company_info(ticker):
    """
    회사 기본 정보를 가져옵니다.
    일부 종목은 Yahoo Finance에서 정보가 제공되지 않을 수 있습니다.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not isinstance(info, dict):
            return {}

        return info

    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_comparison_data(tickers, period):
    """
    여러 종목의 종가 데이터를 가져옵니다.
    """
    comparison = pd.DataFrame()

    for name, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            if data is not None and not data.empty and "Close" in data.columns:
                close = data["Close"].dropna()
                close.index = pd.to_datetime(close.index)

                if getattr(close.index, "tz", None) is not None:
                    close.index = close.index.tz_localize(None)

                comparison[name] = close

        except Exception:
            continue

    return comparison.dropna(how="all")


# --------------------------------------------------
# 표시 관련 함수
# --------------------------------------------------
def format_number(value):
    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:,.2f}조"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}십억"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}백만"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.2f}천"

    return f"{value:,.2f}"


def format_price(value, currency):
    if value is None or pd.isna(value):
        return "-"

    symbols = {
        "USD": "$",
        "KRW": "₩",
        "JPY": "¥",
        "EUR": "€",
        "GBP": "£",
        "HKD": "HK$",
        "CNY": "¥",
        "TWD": "NT$",
    }

    symbol = symbols.get(currency, "")
    value = float(value)

    if currency in ["KRW", "JPY"]:
        return f"{symbol}{value:,.0f}"

    return f"{symbol}{value:,.2f}"


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def create_price_chart(data, ticker, chart_type, show_ma, log_scale):
    chart_data = data.copy()

    chart_data["MA20"] = chart_data["Close"].rolling(window=20).mean()
    chart_data["MA60"] = chart_data["Close"].rolling(window=60).mean()
    chart_data["MA120"] = chart_data["Close"].rolling(window=120).mean()

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
                increasing_line_color="#ef5350",
                decreasing_line_color="#2962ff",
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
                line=dict(width=1.4),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["MA60"],
                mode="lines",
                name="60일 이동평균",
                line=dict(width=1.4),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["MA120"],
                mode="lines",
                name="120일 이동평균",
                line=dict(width=1.4),
            )
        )

    figure.update_layout(
        title=f"{ticker} 주가 차트",
        height=580,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="가격",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis_rangeslider_visible=False,
    )

    if log_scale:
        figure.update_yaxes(type="log")

    return figure


def create_volume_chart(data):
    figure = go.Figure()

    bar_colors = [
        "#ef5350" if close >= open_price else "#2962ff"
        for close, open_price in zip(data["Close"], data["Open"])
    ]

    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            marker_color=bar_colors,
            name="거래량",
        )
    )

    figure.update_layout(
        title="거래량",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="날짜",
        yaxis_title="거래량",
        hovermode="x unified",
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
            line=dict(width=2),
        )
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="과매수 70",
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="과매도 30",
    )

    figure.update_layout(
        title="RSI 상대강도지수",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="날짜",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        showlegend=False,
    )

    return figure


# --------------------------------------------------
# 사이드바
# --------------------------------------------------
with st.sidebar:
    st.header("📊 대시보드 설정")

    selected_name = st.selectbox(
        "종목 선택",
        options=list(GLOBAL_STOCKS.keys()),
        index=0,
    )

    ticker = GLOBAL_STOCKS[selected_name]

    custom_ticker = st.text_input(
        "직접 티커 입력",
        placeholder="예: AMD, NFLX, 005930.KS",
        help="값을 입력하면 위에서 선택한 종목 대신 해당 티커를 사용합니다.",
    ).strip().upper()

    if custom_ticker:
        ticker = custom_ticker
        selected_name = custom_ticker

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

    chart_type = st.radio(
        "차트 종류",
        options=["캔들 차트", "라인 차트"],
        horizontal=True,
    )

    show_ma = st.checkbox(
        "이동평균선 표시",
        value=True,
    )

    log_scale = st.checkbox(
        "로그 스케일",
        value=False,
    )

    st.divider()
    st.caption("데이터 출처: Yahoo Finance")
    st.caption("시장 상황에 따라 데이터가 지연될 수 있습니다.")


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌍 글로벌 주요 주식 대시보드")
st.caption(
    "글로벌 주요 기업과 시장지수의 주가, 거래량, 기술적 지표를 확인할 수 있습니다."
)


# --------------------------------------------------
# 주가 데이터 불러오기
# --------------------------------------------------
period = PERIOD_OPTIONS[period_label]
interval = INTERVAL_OPTIONS[interval_label]

with st.spinner("Yahoo Finance에서 주식 데이터를 불러오는 중입니다..."):
    try:
        stock_data = load_stock_data(ticker, period, interval)
        company_info = load_company_info(ticker)
    except Exception as error:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {error}")
        st.stop()


if stock_data.empty:
    st.error(
        "해당 종목의 데이터를 찾을 수 없습니다. "
        "티커가 정확한지 확인하거나 다른 조회 기간을 선택해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 기본 정보 계산
# --------------------------------------------------
latest_close = float(stock_data["Close"].iloc[-1])
latest_open = float(stock_data["Open"].iloc[-1])
latest_high = float(stock_data["High"].iloc[-1])
latest_low = float(stock_data["Low"].iloc[-1])
latest_volume = float(stock_data["Volume"].iloc[-1])

if len(stock_data) >= 2:
    previous_close = float(stock_data["Close"].iloc[-2])
else:
    previous_close = latest_close

daily_change = latest_close - previous_close
daily_change_percent = (
    (daily_change / previous_close) * 100
    if previous_close != 0
    else 0
)

period_start = float(stock_data["Close"].iloc[0])
period_change_percent = (
    ((latest_close - period_start) / period_start) * 100
    if period_start != 0
    else 0
)

period_high = float(stock_data["High"].max())
period_low = float(stock_data["Low"].min())

currency = company_info.get("currency", "")
company_name = (
    company_info.get("longName")
    or company_info.get("shortName")
    or selected_name
)

market_cap = company_info.get("marketCap")
sector = company_info.get("sector", "-")
industry = company_info.get("industry", "-")
country = company_info.get("country", "-")
exchange = company_info.get("exchange", "-")
website = company_info.get("website")


# --------------------------------------------------
# 종목 제목
# --------------------------------------------------
st.subheader(f"{company_name} · {ticker}")

info_text = f"거래소: {exchange} · 통화: {currency or '-'} · 국가: {country}"

if sector != "-":
    info_text += f" · 섹터: {sector}"

st.caption(info_text)


# --------------------------------------------------
# 핵심 지표
# --------------------------------------------------
metric_columns = st.columns(6)

with metric_columns[0]:
    st.metric(
        "현재 가격",
        format_price(latest_close, currency),
        f"{daily_change_percent:+.2f}%",
    )

with metric_columns[1]:
    st.metric(
        "전일 대비",
        format_price(daily_change, currency),
    )

with metric_columns[2]:
    st.metric(
        "조회 기간 수익률",
        f"{period_change_percent:+.2f}%",
    )

with metric_columns[3]:
    st.metric(
        "기간 최고가",
        format_price(period_high, currency),
    )

with metric_columns[4]:
    st.metric(
        "기간 최저가",
        format_price(period_low, currency),
    )

with metric_columns[5]:
    st.metric(
        "최근 거래량",
        format_number(latest_volume),
    )


# --------------------------------------------------
# 탭 구성
# --------------------------------------------------
tab_price, tab_indicator, tab_compare, tab_market, tab_data = st.tabs(
    [
        "📈 주가 차트",
        "📊 기술적 지표",
        "🔄 종목 비교",
        "🌐 글로벌 지수",
        "📋 원본 데이터",
    ]
)


# --------------------------------------------------
# 주가 차트
# --------------------------------------------------
with tab_price:
    price_figure = create_price_chart(
        data=stock_data,
        ticker=ticker,
        chart_type=chart_type,
        show_ma=show_ma,
        log_scale=log_scale,
    )

    st.plotly_chart(
        price_figure,
        use_container_width=True,
        config={"displaylogo": False},
    )

    detail_columns = st.columns(5)

    detail_columns[0].metric(
        "최근 시가",
        format_price(latest_open, currency),
    )
    detail_columns[1].metric(
        "최근 고가",
        format_price(latest_high, currency),
    )
    detail_columns[2].metric(
        "최근 저가",
        format_price(latest_low, currency),
    )
    detail_columns[3].metric(
        "시가총액",
        format_number(market_cap),
    )
    detail_columns[4].metric(
        "데이터 기준일",
        stock_data.index[-1].strftime("%Y-%m-%d"),
    )

    with st.expander("기업 정보 보기"):
        st.write(f"**기업명:** {company_name}")
        st.write(f"**티커:** {ticker}")
        st.write(f"**거래소:** {exchange}")
        st.write(f"**국가:** {country}")
        st.write(f"**섹터:** {sector}")
        st.write(f"**산업:** {industry}")

        if website:
            st.write(f"**웹사이트:** {website}")


# --------------------------------------------------
# 기술적 지표
# --------------------------------------------------
with tab_indicator:
    left_chart, right_chart = st.columns(2)

    with left_chart:
        volume_figure = create_volume_chart(stock_data)
        st.plotly_chart(
            volume_figure,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with right_chart:
        rsi_figure = create_rsi_chart(stock_data)
        st.plotly_chart(
            rsi_figure,
            use_container_width=True,
            config={"displaylogo": False},
        )

    indicator_data = stock_data.copy()
    indicator_data["MA20"] = indicator_data["Close"].rolling(20).mean()
    indicator_data["MA60"] = indicator_data["Close"].rolling(60).mean()
    indicator_data["MA120"] = indicator_data["Close"].rolling(120).mean()
    indicator_data["RSI14"] = calculate_rsi(indicator_data["Close"])

    latest_ma20 = indicator_data["MA20"].iloc[-1]
    latest_ma60 = indicator_data["MA60"].iloc[-1]
    latest_ma120 = indicator_data["MA120"].iloc[-1]
    latest_rsi = indicator_data["RSI14"].iloc[-1]

    indicator_columns = st.columns(4)

    indicator_columns[0].metric(
        "20일 이동평균",
        format_price(latest_ma20, currency),
    )
    indicator_columns[1].metric(
        "60일 이동평균",
        format_price(latest_ma60, currency),
    )
    indicator_columns[2].metric(
        "120일 이동평균",
        format_price(latest_ma120, currency),
    )
    indicator_columns[3].metric(
        "RSI 14",
        f"{latest_rsi:.2f}" if not pd.isna(latest_rsi) else "-",
    )

    st.info(
        "RSI는 일반적으로 70 이상이면 과매수, "
        "30 이하이면 과매도 구간으로 해석되지만 "
        "단독 투자 판단 기준으로 사용해서는 안 됩니다."
    )


# --------------------------------------------------
# 종목 비교
# --------------------------------------------------
with tab_compare:
    st.subheader("종목별 수익률 비교")

    comparison_names = st.multiselect(
        "비교할 종목을 선택하세요",
        options=list(GLOBAL_STOCKS.keys()),
        default=[
            "애플 (Apple)",
            "마이크로소프트 (Microsoft)",
            "엔비디아 (NVIDIA)",
            "삼성전자",
        ],
        max_selections=8,
    )

    comparison_period_label = st.selectbox(
        "비교 기간",
        options=list(PERIOD_OPTIONS.keys())[:-1],
        index=3,
        key="comparison_period",
    )

    if comparison_names:
        comparison_tickers = {
            name: GLOBAL_STOCKS[name]
            for name in comparison_names
        }

        with st.spinner("비교 데이터를 불러오는 중입니다..."):
            comparison_data = load_comparison_data(
                comparison_tickers,
                PERIOD_OPTIONS[comparison_period_label],
            )

        if comparison_data.empty:
            st.warning("비교 데이터를 불러오지 못했습니다.")

        else:
            # 시작일을 100으로 맞춰 종목별 수익률을 비교합니다.
            normalized_data = comparison_data.copy()

            for column in normalized_data.columns:
                first_valid = normalized_data[column].dropna()

                if not first_valid.empty and first_valid.iloc[0] != 0:
                    normalized_data[column] = (
                        normalized_data[column] / first_valid.iloc[0]
                    ) * 100

            comparison_figure = go.Figure()

            for column in normalized_data.columns:
                comparison_figure.add_trace(
                    go.Scatter(
                        x=normalized_data.index,
                        y=normalized_data[column],
                        mode="lines",
                        name=column,
                        line=dict(width=2),
                    )
                )

            comparison_figure.add_hline(
                y=100,
                line_dash="dash",
                annotation_text="기준점",
            )

            comparison_figure.update_layout(
                title="종목별 상대 수익률 비교 · 시작점 100",
                height=560,
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis_title="날짜",
                yaxis_title="상대 지수",
                hovermode="x unified",
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

            returns = []

            for column in comparison_data.columns:
                valid_data = comparison_data[column].dropna()

                if len(valid_data) >= 2:
                    start_price = valid_data.iloc[0]
                    end_price = valid_data.iloc[-1]
                    return_percent = (
                        (end_price - start_price) / start_price
                    ) * 100

                    returns.append(
                        {
                            "종목": column,
                            "기간 수익률": return_percent,
                        }
                    )

            if returns:
                return_table = pd.DataFrame(returns)
                return_table = return_table.sort_values(
                    "기간 수익률",
                    ascending=False,
                )
                return_table["기간 수익률"] = return_table[
                    "기간 수익률"
                ].map(lambda value: f"{value:+.2f}%")

                st.dataframe(
                    return_table,
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        st.info("비교할 종목을 하나 이상 선택해 주세요.")


# --------------------------------------------------
# 글로벌 시장 지수
# --------------------------------------------------
with tab_market:
    st.subheader("글로벌 주요 시장지수")

    selected_indices = st.multiselect(
        "표시할 지수를 선택하세요",
        options=list(MARKET_INDICES.keys()),
        default=[
            "S&P 500",
            "나스닥 종합",
            "코스피",
            "일본 닛케이 225",
        ],
        max_selections=8,
    )

    market_period_label = st.selectbox(
        "지수 조회 기간",
        options=list(PERIOD_OPTIONS.keys())[:-1],
        index=3,
        key="market_period",
    )

    if selected_indices:
        selected_index_tickers = {
            name: MARKET_INDICES[name]
            for name in selected_indices
        }

        with st.spinner("글로벌 지수 데이터를 불러오는 중입니다..."):
            market_data = load_comparison_data(
                selected_index_tickers,
                PERIOD_OPTIONS[market_period_label],
            )

        if market_data.empty:
            st.warning("시장지수 데이터를 불러오지 못했습니다.")

        else:
            normalized_market = market_data.copy()

            for column in normalized_market.columns:
                first_valid = normalized_market[column].dropna()

                if not first_valid.empty and first_valid.iloc[0] != 0:
                    normalized_market[column] = (
                        normalized_market[column] / first_valid.iloc[0]
                    ) * 100

            market_figure = go.Figure()

            for column in normalized_market.columns:
                market_figure.add_trace(
                    go.Scatter(
                        x=normalized_market.index,
                        y=normalized_market[column],
                        mode="lines",
                        name=column,
                        line=dict(width=2),
                    )
                )

            market_figure.add_hline(
                y=100,
                line_dash="dash",
                annotation_text="기준점",
            )

            market_figure.update_layout(
                title="글로벌 시장지수 상대 수익률 · 시작점 100",
                height=580,
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis_title="날짜",
                yaxis_title="상대 지수",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(
                market_figure,
                use_container_width=True,
                config={"displaylogo": False},
            )

            market_summary = []

            for column in market_data.columns:
                valid_data = market_data[column].dropna()

                if len(valid_data) >= 2:
                    first_value = valid_data.iloc[0]
                    last_value = valid_data.iloc[-1]

                    change_percent = (
                        (last_value - first_value) / first_value
                    ) * 100

                    market_summary.append(
                        {
                            "시장지수": column,
                            "최근 값": f"{last_value:,.2f}",
                            "기간 수익률": f"{change_percent:+.2f}%",
                        }
                    )

            if market_summary:
                st.dataframe(
                    pd.DataFrame(market_summary),
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        st.info("표시할 시장지수를 하나 이상 선택해 주세요.")


# --------------------------------------------------
# 원본 데이터
# --------------------------------------------------
with tab_data:
    st.subheader("주가 원본 데이터")

    display_data = stock_data.copy().sort_index(ascending=False)
    display_data.index = display_data.index.strftime("%Y-%m-%d")
    display_data.index.name = "날짜"

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in numeric_columns:
        if column in display_data.columns:
            if column == "Volume":
                display_data[column] = display_data[column].round(0)
            else:
                display_data[column] = display_data[column].round(2)

    st.dataframe(
        display_data,
        use_container_width=True,
    )

    csv_data = display_data.to_csv().encode("utf-8-sig")

    st.download_button(
        label="📥 CSV 파일 다운로드",
        data=csv_data,
        file_name=f"{ticker}_{period}_{interval}.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# 하단 안내
# --------------------------------------------------
st.divider()

st.caption(
    "본 대시보드는 학습 및 정보 제공 목적으로 제작되었습니다. "
    "표시된 정보는 투자 권유가 아니며, 데이터의 정확성이나 실시간성을 "
    "보장하지 않습니다."
)
