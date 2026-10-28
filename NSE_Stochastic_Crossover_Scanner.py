import time
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NSE Stochastic Crossover Scanner", layout="wide")

# ============================================================
# SETTINGS
# ============================================================
CSV_FILE = Path(r"D:\Py\ind_nifty500list_1.csv")
OUTPUT_DIR = Path(r"D:\Py")

OUTPUT_ALL = OUTPUT_DIR / "NSE500_Stochastic_1433_All_{datetime.now():%Y-%m-%d}.csv"
OUTPUT_BUY = OUTPUT_DIR / f"NSE500_Stochastic_1433_{datetime.now():%Y-%m-%d}_BUY.csv"
OUTPUT_BUY_WEB = OUTPUT_DIR / f"NSE500_Stochastic_1433_{datetime.now():%Y-%m-%d}_BUY_WEB.csv"
OUTPUT_SELL = OUTPUT_DIR / "NSE500_Stochastic_1433_SELL.csv"
OUTPUT_CROSSOVERS = OUTPUT_DIR / "NSE500_Stochastic_1433_Last_Crossovers.csv"

K_PERIOD = 14
K_SMOOTH = 3
D_PERIOD = 3
OVERSOLD = 20
OVERBOUGHT = 80
DOWNLOAD_PERIOD = "6mo"
DOWNLOAD_INTERVAL = "1d"
REQUEST_DELAY = 0.05
NSE_MARKET_CLOSE = dt_time(15, 30)


def load_symbols(uploaded_file=None):
    """Load NSE symbols from an uploaded CSV or the configured local CSV."""
    if uploaded_file is not None:
        universe = pd.read_csv(uploaded_file)
    else:
        if not CSV_FILE.exists():
            raise FileNotFoundError(
                f"CSV file not found: {CSV_FILE}. Upload your NSE symbols CSV in the sidebar."
            )
        universe = pd.read_csv(CSV_FILE)

    possible_columns = ["Symbol", "SYMBOL", "symbol", "Ticker", "TICKER"]
    symbol_column = next((c for c in possible_columns if c in universe.columns), None)
    if symbol_column is None:
        raise ValueError(
            "Could not find a Symbol/Ticker column. "
            f"Available columns: {list(universe.columns)}"
        )

    symbols = (
        universe[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.NS$", "", regex=True)
    )
    symbols = symbols[symbols.ne("")].drop_duplicates().tolist()
    return [symbol + ".NS" for symbol in symbols]


def remove_incomplete_today_bar(df):
    """Exclude today's daily bar if the NSE session has not closed yet."""
    if df.empty:
        return df

    now_india = pd.Timestamp.now(tz="Asia/Kolkata")
    last_date = pd.Timestamp(df.index[-1]).date()
    if last_date == now_india.date() and now_india.time().replace(tzinfo=None) < NSE_MARKET_CLOSE:
        return df.iloc[:-1]
    return df


def calculate_stochastic(df):
    """Calculate Stochastic 14,3,3 and crossover flags."""
    df = df.copy()
    highest_high = df["High"].rolling(K_PERIOD, min_periods=K_PERIOD).max()
    lowest_low = df["Low"].rolling(K_PERIOD, min_periods=K_PERIOD).min()
    denominator = highest_high - lowest_low

    df["Raw_%K"] = np.where(
        denominator.ne(0),
        100 * (df["Close"] - lowest_low) / denominator,
        np.nan,
    )
    df["%K"] = df["Raw_%K"].rolling(K_SMOOTH, min_periods=K_SMOOTH).mean()
    df["%D"] = df["%K"].rolling(D_PERIOD, min_periods=D_PERIOD).mean()

    df["Bullish_Crossover"] = (df["%K"].shift(1) <= df["%D"].shift(1)) & (df["%K"] > df["%D"])
    df["Bearish_Crossover"] = (df["%K"].shift(1) >= df["%D"].shift(1)) & (df["%K"] < df["%D"])
    return df


def get_last_crossover_details(df):
    """Details of the most recent crossover of any type."""
    cross_mask = df["Bullish_Crossover"] | df["Bearish_Crossover"]
    crossover_rows = df.loc[cross_mask]

    if crossover_rows.empty:
        return {
            "Last_Crossover_Date": "",
            "Last_Crossover_Type": "No crossover in available data",
            "Crossover_Close": np.nan,
            "Pct_Change_Since_Crossover": np.nan,
            "Trading_Days_Since_Crossover": np.nan,
        }

    crossover_date = crossover_rows.index[-1]
    row = crossover_rows.iloc[-1]
    crossover_close = float(row["Close"])
    latest_close = float(df.iloc[-1]["Close"])
    pct_change = (latest_close / crossover_close - 1) * 100 if crossover_close else np.nan
    position = df.index.get_loc(crossover_date)
    days_since = len(df) - 1 - int(position)
    crossover_type = "Bullish" if bool(row["Bullish_Crossover"]) else "Bearish"

    return {
        "Last_Crossover_Date": pd.Timestamp(crossover_date).strftime("%Y-%m-%d"),
        "Last_Crossover_Type": crossover_type,
        "Crossover_Close": round(crossover_close, 2),
        "Pct_Change_Since_Crossover": round(pct_change, 2),
        "Trading_Days_Since_Crossover": days_since,
    }


def get_last_signal_details(df):
    """Details of the most recent BUY/SELL signal in the extreme zones."""
    buy_mask = df["Bullish_Crossover"] & (df["%K"] < OVERSOLD)
    sell_mask = df["Bearish_Crossover"] & (df["%K"] > OVERBOUGHT)
    signal_mask = buy_mask | sell_mask
    signal_rows = df.loc[signal_mask]

    if signal_rows.empty:
        return {
            "Last_Signal_Date": "",
            "Last_Signal": "",
            "Signal_Close": np.nan,
            "Pct_Change_Since_Last_Signal": np.nan,
            "Trading_Days_Since_Last_Signal": np.nan,
        }

    signal_date = signal_rows.index[-1]
    row = signal_rows.iloc[-1]
    signal_close = float(row["Close"])
    latest_close = float(df.iloc[-1]["Close"])
    pct_change = (latest_close / signal_close - 1) * 100 if signal_close else np.nan
    position = df.index.get_loc(signal_date)
    days_since = len(df) - 1 - int(position)
    signal = "BUY" if bool(buy_mask.loc[signal_date]) else "SELL"

    return {
        "Last_Signal_Date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        "Last_Signal": signal,
        "Signal_Close": round(signal_close, 2),
        "Pct_Change_Since_Last_Signal": round(pct_change, 2),
        "Trading_Days_Since_Last_Signal": days_since,
    }


def process_stock(symbol):
    """Download and process one NSE stock; return None when data is unavailable."""
    ticker = symbol.removesuffix(".NS")
    df = yf.download(
        symbol,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return None

    # yfinance may return MultiIndex columns for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return None

    df = df[required].copy().dropna()
    df = remove_incomplete_today_bar(df)
    if len(df) < K_PERIOD + K_SMOOTH + D_PERIOD:
        return None

    df = calculate_stochastic(df)
    df = df.dropna(subset=["Raw_%K", "%K", "%D"])
    if len(df) < 2:
        return None

    last = df.iloc[-1]
    previous = df.iloc[-2]
    k, d = float(last["%K"]), float(last["%D"])
    bullish = bool(last["Bullish_Crossover"])
    bearish = bool(last["Bearish_Crossover"])

    if k < OVERSOLD:
        zone = "Oversold"
    elif k > OVERBOUGHT:
        zone = "Overbought"
    else:
        zone = "Neutral"

    if bullish and k < OVERSOLD:
        signal = "BUY"
    elif bearish and k > OVERBOUGHT:
        signal = "SELL"
    elif bullish:
        signal = "Bullish Crossover"
    elif bearish:
        signal = "Bearish Crossover"
    else:
        signal = ""

    if k > d:
        kd_status = "K > D"
    elif k < d:
        kd_status = "K < D"
    else:
        kd_status = "K = D"

    result = {
        "Ticker": ticker,
        "Date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        "Close": round(float(last["Close"]), 2),
        "Raw_%K": round(float(last["Raw_%K"]), 2),
        "Stoch_%K": round(k, 2),
        "Stoch_%D": round(d, 2),
        "Previous_%K": round(float(previous["%K"]), 2),
        "Previous_%D": round(float(previous["%D"]), 2),
        "K_D_Difference": round(k - d, 2),
        "K_D_Status": kd_status,
        "Zone": zone,
        "Bullish_Crossover": bullish,
        "Bearish_Crossover": bearish,
        "Signal": signal,
        **get_last_crossover_details(df),
        **get_last_signal_details(df),
        "Chart": f"https://www.tradingview.com/chart/?symbol=NSE:{ticker}",
        "Yahoo Finance": f"https://finance.yahoo.com/quote/{symbol}",
        "News": f"https://www.google.com/search?q={ticker}+stock+news",
    }
    return result


def run_scanner(symbols, progress_bar, status):
    results = []
    failed_symbols = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        status.write(f"Processing {i}/{total}: {symbol}")
        try:
            result = process_stock(symbol)
            if result is not None:
                results.append(result)
            else:
                failed_symbols.append(symbol)
        except Exception as exc:
            failed_symbols.append(symbol)
            st.warning(f"Error processing {symbol}: {exc}")

        progress_bar.progress(i / total)
        time.sleep(REQUEST_DELAY)

    progress_bar.empty()
    status.empty()

    if not results:
        return pd.DataFrame(), failed_symbols

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["Trading_Days_Since_Last_Signal", "Last_Signal", "Ticker"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result_df.insert(0, "No", range(1, len(result_df) + 1))
    return result_df, failed_symbols


def save_outputs(result_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buy_df = result_df[result_df["Signal"].eq("BUY")].copy()
    sell_df = result_df[result_df["Signal"].eq("SELL")].copy()
    crossover_df = result_df[result_df["Last_Crossover_Date"].astype(str).ne("")].copy()
    crossover_df = crossover_df.sort_values(
        ["Trading_Days_Since_Crossover", "Ticker"],
        ascending=[True, True],
        na_position="last",
    )

    result_df.to_csv(OUTPUT_ALL, index=False)
    buy_df.to_csv(OUTPUT_BUY, index=False)
    sell_df.to_csv(OUTPUT_SELL, index=False)
    crossover_df.to_csv(OUTPUT_CROSSOVERS, index=False)

    selected_fields = [
        "No", "Ticker", "Date", "Close", "Stoch_%K", "Stoch_%D", "Zone", "Signal",
        "Last_Crossover_Date", "Last_Crossover_Type", "Crossover_Close",
        "Pct_Change_Since_Crossover", "Trading_Days_Since_Crossover", "Chart",
    ]
    buy_df[selected_fields].to_csv(OUTPUT_BUY_WEB, index=False)
    return buy_df, sell_df, crossover_df


# ============================================================
# STREAMLIT APP
# ============================================================
st.title("📊 NSE Top 500 Stochastic Crossover Scanner")
st.caption("Stochastic 14,3,3 · Latest crossover · Change since crossover · Trading sessions since crossover")

with st.sidebar:
    st.header("Scanner settings")
    uploaded_csv = st.file_uploader("Upload NSE symbols CSV", type=["csv"])
    st.caption(f"Default local file: {CSV_FILE}")
    run_button = st.button("Run Scanner", type="primary", use_container_width=True)

if run_button:
    try:
        symbols = load_symbols(uploaded_csv)
        st.info(f"Loaded {len(symbols)} symbols.")
        progress_bar = st.progress(0)
        status = st.empty()
        result_df, failed_symbols = run_scanner(symbols, progress_bar, status)

        if result_df.empty:
            st.error("No stocks were successfully processed. Check the CSV and internet/data access.")
        else:
            buy_df, sell_df, crossover_df = save_outputs(result_df)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Processed stocks", len(result_df))
            c2.metric("BUY signals", len(buy_df))
            c3.metric("SELL signals", len(sell_df))
            c4.metric("Stocks with crossover", len(crossover_df))

            st.subheader("Latest crossover results")
            crossover_fields = [
                "Ticker", "Date", "Close", "Stoch_%K", "Stoch_%D",
                # "Last_Crossover_Date", "Last_Crossover_Type", "Crossover_Close",
                # "Pct_Change_Since_Crossover", "Trading_Days_Since_Crossover",
                "Last_Signal_Date", "Last_Signal", "Pct_Change_Since_Last_Signal",
                "Trading_Days_Since_Last_Signal", "Signal", "Chart",
            ]
            crossover_view = crossover_df[
                [c for c in crossover_fields if c in crossover_df.columns]
            ].copy()

            crossover_view = crossover_view.sort_values(
                    by=["Trading_Days_Since_Last_Signal","Last_Signal", "Ticker"],
                    ascending=[True, True, True],
                )
            
            st.dataframe(
                crossover_view,
                column_config={
                    "Chart": st.column_config.LinkColumn("TradingView", display_text="Open Chart"),
                },
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("BUY signals")
            if buy_df.empty:
                st.info("No BUY signals found.")
            else:
                st.dataframe(
                    buy_df,
                    column_config={
                        "Chart": st.column_config.LinkColumn("TradingView", display_text="Open Chart"),
                        "Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance", display_text="Open Quote"),
                        "News": st.column_config.LinkColumn("Latest News", display_text="View News"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            st.subheader("SELL signals")
            if sell_df.empty:
                st.info("No SELL signals found.")
            else:
                st.dataframe(
                    sell_df,
                    column_config={
                        "Chart": st.column_config.LinkColumn("TradingView", display_text="Open Chart"),
                        "Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance", display_text="Open Quote"),
                        "News": st.column_config.LinkColumn("Latest News", display_text="View News"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )


            buy_df_history = result_df[result_df["Last_Signal"] == "BUY"].copy()
            buy_df_history_View = buy_df_history[[c for c in crossover_fields if c in buy_df_history.columns]].copy()
            buy_df_history_View = buy_df_history_View.sort_values(by="Trading_Days_Since_Last_Signal", ascending=True)
        
            sell_df_history = result_df[result_df["Last_Signal"] == "SELL"].copy()
            sell_df_history_View = sell_df_history[[c for c in crossover_fields if c in sell_df_history.columns]].copy() 
            sell_df_history_View = sell_df_history_View.sort_values(by="Trading_Days_Since_Last_Signal", ascending=True) 

            st.subheader("Buy signals - History")
            if buy_df_history.empty:
                st.info("No SELL signals found.")
            else:
                st.dataframe(
                    buy_df_history_View,
                    column_config={
                        "Chart": st.column_config.LinkColumn("TradingView", display_text="Open Chart"),
                        "Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance", display_text="Open Quote"),
                        "News": st.column_config.LinkColumn("Latest News", display_text="View News"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            st.subheader("SELL signals - History")
            if sell_df_history_View.empty:
                st.info("No SELL signals found.")
            else:
                st.dataframe(
                    sell_df_history_View,
                    column_config={
                        "Chart": st.column_config.LinkColumn("TradingView", display_text="Open Chart"),
                        "Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance", display_text="Open Quote"),
                        "News": st.column_config.LinkColumn("Latest News", display_text="View News"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            st.download_button(
                "⬇️ Download all results CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="NSE500_Stochastic_1433_All.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download latest crossovers CSV",
                data=crossover_df.to_csv(index=False).encode("utf-8"),
                file_name="NSE500_Stochastic_1433_Last_Crossovers.csv",
                mime="text/csv",
            )
            if failed_symbols:
                with st.expander(f"Failed / unavailable symbols ({len(failed_symbols)})"):
                    st.write(failed_symbols)

            st.success(
                f"Scan complete. CSV files saved to {OUTPUT_DIR} "
                "(when that folder is writable)."
            )
    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Upload your NSE symbols CSV if needed, then click **Run Scanner**.")
