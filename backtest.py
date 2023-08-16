import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
from finta import TA

# from matplotlib import pyplot
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")  # setting ignore as a parameter

with open("Models/spot_model_full_60min_1year_till_07.json", "r") as fin:
    model = model_from_json(fin.read())  # Load model

# forecast_data = model.make_future_dataframe(periods=500, freq='H')
# COLUMNS = [
#     "open_time",
#     "open",
#     "high",
#     "low",
#     "close",
#     "volume",
#     "close_time",
#     "quote_volume",
#     "count",
#     "taker_buy_volume",
#     "taker_buy_quote_volume",
#     "ignore",
# ]


df = pd.read_csv(
    "Dataset-Spot/BTCUSDT/60min/BTCUSDT-60min-1year-till-07.csv",  # , skiprows=[i for i in range(0, 202367) if i % 4 != 0]
    # names=COLUMNS,
    # header=None,
)

df = df.dropna()

test = pd.read_csv(
    "Dataset-Spot/BTCUSDT/60min/BTCUSDT-60min-2023-08.csv",  # , skiprows=[i for i in range(0, 202367) if i % 4 != 0]
    # names=COLUMNS,
    # header=None,
)

test = test.dropna()

result = {}

up_count = 0
down_count = 0

other_count = 0

diff_sum = 0

inventory = []

max_inventory = 0

total_profit = 0

prediction = pd.DataFrame()

for test_index in test.index.tolist():
    df = df.append(test.loc[test_index], ignore_index=True)

    df["Returns"] = df.close.pct_change()
    df["Log Returns"] = np.log(1 + df["Returns"])

    df.fillna(0, inplace=True)

    df["RSI"] = TA.RSI(df)
    df["MACD"] = TA.VWAP(df)
    df["SMA"] = TA.SMA(df)
    df["BBANDS"] = TA.ROC(df)
    df["EMA"] = TA.EMA(df)

    df.fillna(0, inplace=True)

    future = pd.DataFrame()

    future["ds"] = [
        pd.to_datetime(df.iloc[-1]["open_time"], unit="ms") + timedelta(minutes=60)
    ]

    future["y"] = [df.iloc[-1]["open"]]

    future["high"] = [df.iloc[-1]["high"]]
    future["low"] = [df.iloc[-1]["low"]]
    future["close"] = [df.iloc[-1]["close"]]
    future["volume"] = [df.iloc[-1]["volume"]]
    future["Returns"] = [df.iloc[-1]["Returns"]]
    future["Log Returns"] = [df.iloc[-1]["Log Returns"]]
    future["BBANDS"] = [df.iloc[-1]["BBANDS"]]
    future["RSI"] = [df.iloc[-1]["RSI"]]
    future["MACD"] = [df.iloc[-1]["MACD"]]
    future["SMA"] = [df.iloc[-1]["SMA"]]
    future["EMA"] = [df.iloc[-1]["EMA"]]

    forecast = model.predict(future)

    prediction = prediction.append(forecast.iloc[0], ignore_index=True)

    current_time = f"{pd.to_datetime(df.iloc[-1]['open_time'], unit='ms')}"
    next_time = (
        f"{pd.to_datetime(df.iloc[-1]['open_time'], unit='ms') + timedelta(minutes=60)}"
    )

    if current_time in result:
        result[current_time]["current"] = df.iloc[-1]["open"]

    else:
        result[current_time] = {}
        result[current_time]["current"] = df.iloc[-1]["open"]

    if next_time in result:
        result[next_time]["predicted"] = forecast.iloc[0]["yhat"]
    else:
        result[next_time] = {}
        result[next_time]["predicted"] = forecast.iloc[0]["yhat"]

    if "current" in result[current_time] and "predicted" in result[current_time]:
        result[current_time]["diff"] = (
            result[current_time]["predicted"] - result[current_time]["current"]
        )

        diff_sum += abs(result[current_time]["diff"])
    else:
        result[current_time]["diff"] = 0

    if len(result.keys()) > 2:
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________
        #                                 15 mins buy/sell/hold
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________

        # price = price["k"]

        # new_row = {
        #     "open_time": int(price["t"]),
        #     "open": float(price["o"]),
        #     "high": float(price["h"]),
        #     "low": float(price["l"]),
        #     "close": float(price["c"]),
        #     "volume": float(price["v"]),
        #     "close_time": int(price["T"]),
        #     "quote_volume": float(price["q"]),
        #     "count": float(price["n"]),
        #     "taker_buy_volume": float(price["V"]),
        #     "taker_buy_quote_volume": float(price["Q"]),
        #     "ignore": float(price["B"]),
        # }

        pred_15min_trend = list(result)[-3:]

        # print(
        #     f"Buying after 15 mins {list(result.items())[-3:]} | {pred_15min_trend}"
        # )

        if (
            "current" in result[pred_15min_trend[0]]
            and "current" in result[pred_15min_trend[1]]
            and "predicted" in result[pred_15min_trend[1]]
            and "predicted" in result[pred_15min_trend[2]]
        ):
            prev_value_of_15min_trend = result[pred_15min_trend[0]]["current"]

            current_value_of_15min_trend = result[pred_15min_trend[1]]["current"]

            current_pred_value_of_15min_trend = result[pred_15min_trend[1]]["predicted"]

            next_pred_value_of_15min_trend = result[pred_15min_trend[2]]["predicted"]

            if (
                next_pred_value_of_15min_trend > current_value_of_15min_trend
                and next_pred_value_of_15min_trend > current_pred_value_of_15min_trend
            ):
                buy_value = current_value_of_15min_trend * 0.01
                inventory.append(
                    {
                        "open_time": f"{current_time}",
                        "actual_open": current_value_of_15min_trend,
                        "open": buy_value,
                        "with_tx_fee": buy_value - (buy_value * 0.001),
                        # "time": pred_15min_trend,
                    }
                )
                if len(inventory) > max_inventory:
                    max_inventory = len(inventory)

                print(f"signal 'Buy' at {inventory[-1]}")

            if (
                next_pred_value_of_15min_trend < current_value_of_15min_trend
                or next_pred_value_of_15min_trend < current_pred_value_of_15min_trend
                or current_value_of_15min_trend < prev_value_of_15min_trend
            ):
                sell_value = current_value_of_15min_trend * 0.01
                sell_value_with_tx_fee = sell_value - (sell_value * 0.001)

                inventory_to_remove = []

                for index in range(len(inventory)):
                    current_inventory = inventory[index]

                    profit = sell_value_with_tx_fee - current_inventory["with_tx_fee"]

                    if profit > 0:
                        temp = {
                            "open_time": f"{current_time}",
                            "actual_open": current_value_of_15min_trend,
                            "open": sell_value,
                            "with_tx_fee": sell_value_with_tx_fee,
                            "profit": profit,
                        }

                        inventory_to_remove.append(current_inventory)
                        total_profit = total_profit + temp["profit"]
                        print(f"signal 'Sell' at {temp}")

                inventory = [i for i in inventory if i not in inventory_to_remove]

            # if not buy_or_sell:
            #     print(
            #         f"signal 'do nothing' current_actual {current_value_of_15min_trend} current_pred {current_pred_value_of_15min_trend} next_pred {next_pred_value_of_15min_trend}"
            #     )

            prev_actual_value = prev_value_of_15min_trend
            current_actual_value = current_value_of_15min_trend
            current_predicted_value = current_pred_value_of_15min_trend

            actual_diff = current_actual_value - prev_actual_value
            predicted_diff = current_predicted_value - prev_actual_value

            # if df['open_time'][test_index] > date_time_obj:
            if actual_diff > 0 and predicted_diff > 0:
                up_count = up_count + 1
            elif actual_diff <= 0 and predicted_diff <= 0:
                down_count = down_count + 1
            else:
                other_count = other_count + 1

            if (other_count + up_count + down_count) > 0:
                print(
                    f"up_count {up_count} | down_count {down_count} | wrong | {other_count} | total_correct {up_count+down_count} | mean_diff | {diff_sum/(other_count+up_count+down_count)}  "
                )

print(total_profit)
print(inventory)
print(len(inventory))
# print(result)
print(max_inventory)
