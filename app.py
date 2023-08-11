import numpy as np
import pandas as pd

from prophet.serialize import model_from_json
from finta import TA

from datetime import datetime, timedelta
import time

import requests
import json

import logging
from binance import ThreadedWebsocketManager


logging.basicConfig(
    filename="update.log",
    format="%(asctime)s |||| %(message)s",
    datefmt="%d-%b-%y %H:%M",
    level=logging.WARNING,
)

logging.warning("Starting")


# loaidng model
with open("serialized_model_full_15min_till_07.json", "r") as fin:
    global model_15_mins
    model_15_mins = model_from_json(fin.read())
with open("serialized_model_full_30min_till_07.json", "r") as fin:
    global model_30_mins
    model_30_mins = model_from_json(fin.read())
with open("serialized_model_full_1h_till_07.json", "r") as fin:
    global model_60_mins
    model_60_mins = model_from_json(fin.read())


def get_minutes_diff(then):
    now = datetime.fromisoformat(datetime.now().isoformat(timespec="minutes"))
    duration = now - then
    duration_in_s = duration.total_seconds()
    minutes = int(divmod(duration_in_s, 60)[0])
    return minutes


global inventory
inventory = []

global result
result = {"15mins": {}, "30mins": {}, "60mins": {}}

global last_time
last_time = None

global return_signal
return_signal = False


global count
count = (0,0)

twm = ThreadedWebsocketManager()
twm.start()


def trade(price):
    try:
        global inventory

        global result

        global last_time
        global return_signal

        global model_15_mins
        global model_30_mins
        global model_60_mins

        global count

        if last_time is not None:
            minutes = get_minutes_diff(last_time)
            if minutes not in [0, 15, 30, 45, 60]:
                return_signal = False
                return

            if minutes in [0, 15, 30, 45, 60] and return_signal == True:
                return

            if minutes == 60:
                last_time = datetime.fromisoformat(
                    datetime.now().isoformat(timespec="minutes")
                )
                # result = {"15mins": {}, "30mins": {}, "60mins": {}}
                minutes = 0
        else:
            last_time = datetime.fromisoformat(
                datetime.now().isoformat(timespec="minutes")
            )
            minutes = 0

        return_signal = True

        if minutes == 0 or minutes == 15 or minutes == 30 or minutes == 45:
            interval_15min_data = pd.DataFrame(
                json.loads(
                    requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "15m",
                            "limit": 1,
                        },
                    ).text
                )
            )

            interval_15min_data.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]

            interval_15min_data = interval_15min_data.astype(float)

            actual_dataframe = (
                pd.read_csv(
                    f"BTCUSDT-15min-till-now.csv",
                )
            ).dropna()

            actual_dataframe = actual_dataframe.loc[
                :, ~actual_dataframe.columns.str.contains("^Unnamed")
            ]

            recent_result = pd.concat([actual_dataframe, interval_15min_data])

            recent_result = recent_result.loc[
                :, ~recent_result.columns.str.contains("^Unnamed")
            ]

            recent_result.to_csv("BTCUSDT-15min-till-now.csv", index=False)

            # calculating all factors
            recent_result["open_time"] = pd.to_datetime(
                recent_result["open_time"], unit="ms", utc=False
            )
            recent_result["Returns"] = recent_result.close.pct_change()
            recent_result["Log Returns"] = np.log(1 + recent_result["Returns"])
            recent_result["RSI"] = TA.RSI(recent_result)
            recent_result["MACD"] = TA.VWAP(recent_result)
            recent_result["SMA"] = TA.SMA(recent_result)
            recent_result["BBANDS"] = TA.ROC(recent_result)
            recent_result["EMA"] = TA.EMA(recent_result)

            recent_result.fillna(0, inplace=True)

            # getting predicted value
            predicted_time = recent_result.iloc[-1]["open_time"] + timedelta(minutes=15)

            future = pd.DataFrame()
            future["ds"] = [predicted_time]

            future["y"] = [recent_result.iloc[-1]["open"]]
            future["high"] = [recent_result.iloc[-1]["high"]]
            future["low"] = [recent_result.iloc[-1]["low"]]
            future["close"] = [recent_result.iloc[-1]["close"]]
            future["volume"] = [recent_result.iloc[-1]["volume"]]
            future["Returns"] = [recent_result.iloc[-1]["Returns"]]
            future["Log Returns"] = [recent_result.iloc[-1]["Log Returns"]]
            future["BBANDS"] = [recent_result.iloc[-1]["BBANDS"]]
            future["RSI"] = [recent_result.iloc[-1]["RSI"]]
            future["MACD"] = [recent_result.iloc[-1]["MACD"]]
            future["SMA"] = [recent_result.iloc[-1]["SMA"]]
            future["EMA"] = [recent_result.iloc[-1]["EMA"]]

            # forecasting the predicted value
            forecast = model_15_mins.predict(future)

            # current_time = f"{df.iloc[-1]['open_time']}"
            # current_time = current_time[:-3]
            current_time = datetime.fromisoformat(
                recent_result.iloc[-1]["open_time"].isoformat(timespec="minutes")
            )

            current_time = f"{current_time}"

            next_time = datetime.fromisoformat(
                predicted_time.isoformat(timespec="minutes")
            )
            next_time = f"{next_time}"

            if current_time in result["15mins"]:
                result["15mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]
            else:
                result["15mins"][current_time] = {}
                result["15mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]

            if next_time in result["15mins"]:
                result["15mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]
            else:
                result["15mins"][next_time] = {}
                result["15mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]

            future = None
            recent_result = None
            actual_dataframe = None

            logging.warning(
                f"Inside 15 mintes bianance current is {current_time} server current is {datetime.now()} and last time is {last_time}"
            )

        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________
        #                                  now 30 mins
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________

        if minutes == 0 or minutes == 30:
            interval_30min_data = pd.DataFrame(
                json.loads(
                    requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "30m",
                            "limit": 1,
                        },
                    ).text
                )
            )

            interval_30min_data.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]

            interval_30min_data = interval_30min_data.astype(float)

            actual_dataframe = (
                pd.read_csv(
                    f"BTCUSDT-30min-till-now.csv",
                )
            ).dropna()

            actual_dataframe = actual_dataframe.loc[
                :, ~actual_dataframe.columns.str.contains("^Unnamed")
            ]

            recent_result = pd.concat([actual_dataframe, interval_30min_data])

            recent_result = recent_result.loc[
                :, ~recent_result.columns.str.contains("^Unnamed")
            ]

            recent_result.to_csv("BTCUSDT-30min-till-now.csv", index=False)

            # calculating all factors
            recent_result["open_time"] = pd.to_datetime(
                recent_result["open_time"], unit="ms", utc=False
            )
            recent_result["Returns"] = recent_result.close.pct_change()
            recent_result["Log Returns"] = np.log(1 + recent_result["Returns"])
            recent_result["RSI"] = TA.RSI(recent_result)
            recent_result["MACD"] = TA.VWAP(recent_result)
            recent_result["SMA"] = TA.SMA(recent_result)
            recent_result["BBANDS"] = TA.ROC(recent_result)
            recent_result["EMA"] = TA.EMA(recent_result)

            recent_result.fillna(0, inplace=True)

            # getting predicted value
            predicted_time = recent_result.iloc[-1]["open_time"] + timedelta(minutes=30)

            future = pd.DataFrame()
            future["ds"] = [predicted_time]

            future["y"] = [recent_result.iloc[-1]["open"]]
            future["high"] = [recent_result.iloc[-1]["high"]]
            future["low"] = [recent_result.iloc[-1]["low"]]
            future["close"] = [recent_result.iloc[-1]["close"]]
            future["volume"] = [recent_result.iloc[-1]["volume"]]
            future["Returns"] = [recent_result.iloc[-1]["Returns"]]
            future["Log Returns"] = [recent_result.iloc[-1]["Log Returns"]]
            future["BBANDS"] = [recent_result.iloc[-1]["BBANDS"]]
            future["RSI"] = [recent_result.iloc[-1]["RSI"]]
            future["MACD"] = [recent_result.iloc[-1]["MACD"]]
            future["SMA"] = [recent_result.iloc[-1]["SMA"]]
            future["EMA"] = [recent_result.iloc[-1]["EMA"]]

            # forecasting the predicted value
            forecast = model_30_mins.predict(future)

            # current_time = f"{df.iloc[-1]['open_time']}"
            # current_time = current_time[:-3]

            current_time = datetime.fromisoformat(
                recent_result.iloc[-1]["open_time"].isoformat(timespec="minutes")
            )
            current_time = f"{current_time}"

            next_time = datetime.fromisoformat(
                predicted_time.isoformat(timespec="minutes")
            )
            next_time = f"{next_time}"

            if current_time in result["30mins"]:
                result["30mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]
            else:
                result["30mins"][current_time] = {}
                result["30mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]

            if next_time in result["30mins"]:
                result["30mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]
            else:
                result["30mins"][next_time] = {}
                result["30mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]

            future = None
            recent_result = None
            actual_dataframe = None

            logging.warning(
                f"Inside 30 mintes bianance current is {current_time} server current is {datetime.now()} and last time is {last_time}"
            )
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________
        #                                  now 60 mins
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________

        if minutes == 0:
            interval_60min_data = pd.DataFrame(
                json.loads(
                    requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "1h",
                            "limit": 1,
                        },
                    ).text
                )
            )

            interval_60min_data.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]

            interval_60min_data = interval_60min_data.astype(float)

            actual_dataframe = (
                pd.read_csv(
                    f"BTCUSDT-60min-till-now.csv",
                )
            ).dropna()

            actual_dataframe = actual_dataframe.loc[
                :, ~actual_dataframe.columns.str.contains("^Unnamed")
            ]

            recent_result = pd.concat([actual_dataframe, interval_60min_data])

            recent_result = recent_result.loc[
                :, ~recent_result.columns.str.contains("^Unnamed")
            ]

            recent_result.to_csv("BTCUSDT-60min-till-now.csv", index=False)

            # calculating all factors
            recent_result["open_time"] = pd.to_datetime(
                recent_result["open_time"], unit="ms", utc=False
            )
            recent_result["Returns"] = recent_result.close.pct_change()
            recent_result["Log Returns"] = np.log(1 + recent_result["Returns"])
            recent_result["RSI"] = TA.RSI(recent_result)
            recent_result["MACD"] = TA.VWAP(recent_result)
            recent_result["SMA"] = TA.SMA(recent_result)
            recent_result["BBANDS"] = TA.ROC(recent_result)
            recent_result["EMA"] = TA.EMA(recent_result)

            recent_result.fillna(0, inplace=True)

            # getting predicted value
            predicted_time = recent_result.iloc[-1]["open_time"] + timedelta(minutes=60)

            future = pd.DataFrame()
            future["ds"] = [predicted_time]

            future["y"] = [recent_result.iloc[-1]["open"]]
            future["high"] = [recent_result.iloc[-1]["high"]]
            future["low"] = [recent_result.iloc[-1]["low"]]
            future["close"] = [recent_result.iloc[-1]["close"]]
            future["volume"] = [recent_result.iloc[-1]["volume"]]
            future["Returns"] = [recent_result.iloc[-1]["Returns"]]
            future["Log Returns"] = [recent_result.iloc[-1]["Log Returns"]]
            future["BBANDS"] = [recent_result.iloc[-1]["BBANDS"]]
            future["RSI"] = [recent_result.iloc[-1]["RSI"]]
            future["MACD"] = [recent_result.iloc[-1]["MACD"]]
            future["SMA"] = [recent_result.iloc[-1]["SMA"]]
            future["EMA"] = [recent_result.iloc[-1]["EMA"]]

            # forecasting the predicted value
            forecast = model_60_mins.predict(future)

            # current_time = f"{df.iloc[-1]['open_time']}"
            # current_time = current_time[:-3]

            current_time = datetime.fromisoformat(
                recent_result.iloc[-1]["open_time"].isoformat(timespec="minutes")
            )
            current_time = f"{current_time}"

            next_time = datetime.fromisoformat(
                predicted_time.isoformat(timespec="minutes")
            )
            next_time = f"{next_time}"

            if current_time in result["60mins"]:
                result["60mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]
            else:
                result["60mins"][current_time] = {}
                result["60mins"][current_time]["actual"] = recent_result.iloc[-1][
                    "open"
                ]

            if next_time in result["60mins"]:
                result["60mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]
            else:
                result["60mins"][next_time] = {}
                result["60mins"][next_time]["pred"] = forecast.iloc[0]["yhat"]

            future = None
            recent_result = None
            actual_dataframe = None

            logging.warning(
                f"Inside 60 mintes bianance current is {current_time} server current is {datetime.now()} and last time is {last_time}"
            )

        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________
        #                                  now calculating results
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________

        if len(result["15mins"].keys()) > 2:
            # ___________________________________________________________________________________________________
            # ___________________________________________________________________________________________________
            #                                 15 mins buy/sell/hold
            # ___________________________________________________________________________________________________
            # ___________________________________________________________________________________________________

            price = price["k"]

            new_row = {
                "open_time": int(price["t"]),
                "open": float(price["o"]),
                "high": float(price["h"]),
                "low": float(price["l"]),
                "close": float(price["c"]),
                "volume": float(price["v"]),
                "close_time": int(price["T"]),
                "quote_volume": float(price["q"]),
                "count": float(price["n"]),
                "taker_buy_volume": float(price["V"]),
                "taker_buy_quote_volume": float(price["Q"]),
                "ignore": float(price["B"]),
            }

            pred_15min_trend = list(result["15mins"])[-3:]

            logging.warning(
                f"Buying after 15 mins {list(result['15mins'].items())[-3:]} | {pred_15min_trend}"
            )

            if (
                "actual" in result["15mins"][pred_15min_trend[0]]
                and "actual" in result["15mins"][pred_15min_trend[1]]
                and "pred" in result["15mins"][pred_15min_trend[1]]
                and "pred" in result["15mins"][pred_15min_trend[2]]
            ):
                prev_value_of_15min_trend = result["15mins"][pred_15min_trend[0]][
                    "actual"
                ]

                current_value_of_15min_trend = result["15mins"][pred_15min_trend[1]][
                    "actual"
                ]

                current_pred_value_of_15min_trend = result["15mins"][
                    pred_15min_trend[1]
                ]["pred"]

                next_pred_value_of_15min_trend = result["15mins"][pred_15min_trend[2]][
                    "pred"
                ]

                buy_or_sell = False

                if (
                    next_pred_value_of_15min_trend > current_value_of_15min_trend
                    and next_pred_value_of_15min_trend
                    > current_pred_value_of_15min_trend
                ):
                    buy_value = 1 / new_row["open"] * 200
                    inventory.append(
                        {
                            "open_time": f'{datetime.fromtimestamp(new_row["open_time"] / 1000)}',
                            "actual_open": new_row["open"],
                            "open": buy_value,
                            "with_tx_fee": buy_value - (buy_value * 0.001),
                            # "time": pred_15min_trend,
                        }
                    )

                    logging.warning(f"signal 'Buy' at {inventory[-1]}")
                    buy_or_sell = True

                if (
                    next_pred_value_of_15min_trend < current_value_of_15min_trend
                    or next_pred_value_of_15min_trend
                    < current_pred_value_of_15min_trend
                    or current_value_of_15min_trend < prev_value_of_15min_trend
                ):
                    sell_value = 1 / new_row["open"] * 200
                    sell_value_with_tx_fee = sell_value - (sell_value * 0.001)

                    inventory_to_remove = []

                    for index in range(len(inventory)):
                        current_inventory = inventory[index]

                        profit = (
                            sell_value_with_tx_fee - current_inventory["with_tx_fee"]
                        )

                        if profit > 0:
                            temp = {
                                "open_time": f'{datetime.fromtimestamp(new_row["open_time"] / 1000)}',
                                "actual_open": new_row["open"],
                                "open": sell_value,
                                "with_tx_fee": sell_value_with_tx_fee,
                                "profit": profit,
                                "profit_usdt": new_row["open"] * profit,
                            }

                            inventory_to_remove.append(current_inventory)

                            logging.warning(f"signal 'Sell' at {temp}")
                            buy_or_sell = True

                    inventory = [i for i in inventory if i not in inventory_to_remove]

                    # buy_time = datetime.strptime(inventory[index]["open_time"], '%y-%m-%d %H:%M:%S')

                    # current_time = datetime.fromtimestamp(
                    #     new_row["open_time"] / 1000
                    # )

                    # duration = current_time - buy_time
                    # duration_in_s = duration.total_seconds()
                    # buy_sell_minutes_diff = int(divmod(duration_in_s, 60)[0])

                    # if buy_sell_minutes_diff > 15:

                if not buy_or_sell:
                    logging.warning(
                        f"signal 'do nothing' current_actual {current_value_of_15min_trend} current_pred {current_pred_value_of_15min_trend} next_pred {next_pred_value_of_15min_trend}"
                    )
            
                prev_actual = prev_value_of_15min_trend
                current_value = current_value_of_15min_trend
                pred_value = current_pred_value_of_15min_trend

                if current_value - prev_actual > 0 and pred_value - prev_actual > 0:
                    count[0] = count[0] + 1

                elif current_value - prev_actual < 0 and pred_value - prev_actual < 0:
                    count[0] = count[0] + 1
                else:
                    count[1] = count[1] + 1
                
                logging.warning(
                    f"correct {count[0]} | wrong {count[1]} | inventory {inventory} | {len(inventory)}"
                )
            else:
                logging.warning(
                    f"Data is incomplete in {result['15mins']} for {pred_15min_trend}"
                )
            
            
        # elif len(result["30mins"].keys()) > 2:
        #     logging.warning(list(result["30mins"].items())[-3:])
        # elif len(result["60mins"].keys()) > 2:
        #     logging.warning(list(result["60mins"].items())[-3:])
        else:
            logging.warning(f"waiting for complete data | {result}")

        # if minutes == 30:

        # else:
        #     logging.warning(f"signal 'Do nothing' {result}")

        # current_value_of_30min_trend = result["30mins"][pred_30min_trend[0]]["actual"]
        # next_pred_value_of_30min_trend = result["30mins"][pred_30min_trend[1]]["pred"]

        # if next_pred_value_of_30min_trend > current_value_of_30min_trend:
        #     up_trend = up_trend + 1
        # else:
        #     down_trend = down_trend + 1
        # current_value_of_60min_trend = result["60mins"][pred_60min_trend[0]]["actual"]
        # next_pred_value_of_60min_trend = result["60mins"][pred_60min_trend[1]]["pred"]

        # if next_pred_value_of_60min_trend > current_value_of_60min_trend:
        #     up_trend = up_trend + 1
        # else:
        #     down_trend = down_trend + 1

        # if up_trend > 2 and len(inventory) == 0:
        #     inventory.append(
        #         {
        #             "open_time": f'{datetime.fromtimestamp(new_row["open_time"] / 1000)}',
        #             "open": new_row["open"],
        #             "with_tx_fee": new_row["open"] + (new_row["open"] * 0.001),
        #         }
        #     )

        #     logging.warning(f"Buy at {inventory[0]}")

        # if down_trend > 2 and len(inventory) == 1:
        #     sell_at = inventory.pop()
        #     actual_sell = new_row["open"] - sell_at["open"]
        #     actual_with_fee = (new_row["open"] - (new_row["open"] * 0.001)) - sell_at[
        #         "with_tx_fee"
        #     ]

        #     logging.warning(
        #         f"Sell at actual_profit: {actual_sell} , actual_profit_with_fee: {actual_with_fee}, sell_at {datetime.fromtimestamp(new_row['open_time'] / 1000)}"
        #     )

    except Exception as e:
        logging.warning(f"Error: {e}")
    # twm.stop()

    # next_time = f"{predicted_time}"
    # next_time = next_time[:-3]

    # # adding data to dic
    # if current_time in result["15mins"]:
    #     result["15mins"][current_time]["current"] = df.iloc[-1]["open"]
    # else:
    #     result["15mins"][current_time] = {}
    #     result["15mins"][current_time]["current"] = df.iloc[-1]["open"]

    # if next_time in result["15mins"]:
    #     result["15mins"][next_time]["predicted"] = forecast.iloc[0]["yhat"]
    # else:
    #     result["15mins"][next_time] = {}
    #     result["15mins"][next_time]["predicted"] = forecast.iloc[0]["yhat"]

    # if "current" in result["15mins"][current_time] and "predicted" in result["15mins"][current_time]:
    #     result["15mins"][current_time]["diff"] = (
    #         result["15mins"][current_time]["predicted"] - result["15mins"][current_time]["current"]
    #     )

    # else:
    #     result["15mins"][current_time]["diff"] = 0

    # # calculating result acc
    # if not prev_time:
    #     prev_time = current_time

    # else:
    #     if (
    #         "current" in result[prev_time]
    #         and "current" in result[current_time]
    #         and "predicted" in result[current_time]
    #     ):
    #         prev_actual_value = result[prev_time]["current"]
    #         current_actual_value = result[current_time]["current"]
    #         current_predicted_value = result[current_time]["predicted"]

    #         actual_trend = current_actual_value - prev_actual_value
    #         predicted_trend = current_predicted_value - prev_actual_value

    #         if actual_trend > 0 and predicted_trend > 0:
    #             true_pred_trend.append({current_time: 1})
    #         elif actual_trend <= 0 and predicted_trend <= 0:
    #             true_pred_trend.append({current_time: 0})
    #         else:
    #             false_pred_trend.append(current_time)
    #     else:
    #         return

    # list_of_result = list(result)

    # # getting last 5 values with pred
    # if len(list_of_result) > 5:
    #     last_four = list_of_result[-5:-1]

    #     mean_diff = 0
    #     for last in last_four:
    #         if last in result:
    #             mean_diff = mean_diff + abs(result[last]["diff"])

    #     mean_diff = mean_diff / 4

    #     mean_diff = abs(mean_diff - (mean_diff * 0.25))
    #     # mean_diff = mean_diff * 0.01 # as actual is 10per so it is also 10%

    #     # current pred and actual value
    #     next_predicted_value = result[next_time]["predicted"]

    #     current_actual_value = result[current_time]["current"]

    #     # adding becouse we want to check if it is profitable to buy againt pred value or not
    #     if next_predicted_value > current_actual_value and len(inventory) == 0:
    #         # finding 1 per
    #         current_actual_value_one_per = current_actual_value * 0.01
    #         buy_tx_fee = current_actual_value_one_per * 0.001  # getting tx fee

    #         next_predicted_value_mean = (
    #             next_predicted_value - mean_diff
    #         )  # subtracting mean diff becouse this is not an actual price

    #         next_predicted_value_mean_one_per = (
    #             next_predicted_value_mean * 0.01
    #         )  # finding 1 per of pred price
    #         sell_tx_fee = (
    #             next_predicted_value_mean_one_per * 0.001
    #         )  # adding tx fee to pred price

    #         pred_profit = (next_predicted_value_mean_one_per - sell_tx_fee) - (
    #             current_actual_value_one_per + buy_tx_fee
    #         )

    #         logging.warning(
    #             f"next_predicted_value_mean_one_per {next_predicted_value_mean_one_per} | sell_tx_fee {sell_tx_fee} | current_actual_value_one_per {current_actual_value_one_per} | buy_tx_fee {buy_tx_fee} "
    #         )

    #         if pred_profit > 0:
    #             inventory.append(
    #                 (
    #                     current_actual_value_one_per,
    #                     current_actual_value_one_per + buy_tx_fee,
    #                 )
    #             )

    #             logging.warning(
    #                 f"Signal: 'buy' | buy_price_actual {current_actual_value_one_per}  buy_price_pred {next_predicted_value * 0.01} | mean_diff {mean_diff}"
    #             )

    #     if next_predicted_value < current_actual_value and len(inventory) > 0:
    #         buy_asset = inventory.pop(0)

    #         current_actual_value_one_per = current_actual_value * 0.01
    #         sell_tx_fee = current_actual_value_one_per * 0.001

    #         sell_amount = (current_actual_value_one_per - sell_tx_fee) - buy_asset(
    #             1
    #         )

    #         logging.warning(
    #             f"Signal: 'sell' | buy_price {buy_asset} | sell_price {current_actual_value_one_per} | ptofit {sell_amount}"
    #         )

    #     logging.warning(
    #         f"Last four mean diff {mean_diff} | current {current_actual_value} at {current_time} | pred {next_predicted_value} "
    #     )

    # if len(result.keys()) > 3:
    #     logging.warning(list(result.items())[-3:])
    # else:
    #     logging.warning(result)

    # # loggin results
    # if (len(true_pred_trend)) > 1 and (len(false_pred_trend)) > 1:
    #     logging.warning(
    #         f"total_c {len(true_pred_trend)} | total_w {len(false_pred_trend)} | last_correct_pred {true_pred_trend[-1]} | last_wrong_pred | {false_pred_trend[-1]} | mean_diff | {diff_sum/(len(true_pred_trend) + len(false_pred_trend))} "
    #     )


try:
    minutes = [15, 30, 60]
    for min in minutes:
        # reading dataset
        actual_dataframe = (
            pd.read_csv(
                f"BTCUSDT-{min}min-till-07.csv",
            )
        ).dropna()

        # getting no of days passed
        diff = get_minutes_diff(datetime(2023, 8, 1, 00, 00, 00))

        if min == 60:
            # getting data from aug till now
            new_data = pd.DataFrame(
                json.loads(
                    requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": "BTCUSDT",
                            "interval": f"1h",
                            "limit": int(diff / min),
                        },
                    ).text
                )
            )
        else:
            # getting data from aug till now
            new_data = pd.DataFrame(
                json.loads(
                    requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": "BTCUSDT",
                            "interval": f"{min}m",
                            "limit": int(diff / min),
                        },
                    ).text
                )
            )

        new_data.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]

        new_data = new_data.astype(float)

        # combining old and ndew data
        recent_result = pd.concat([actual_dataframe, new_data])

        recent_result = recent_result.loc[
            :, ~recent_result.columns.str.contains("^Unnamed")
        ]

        # saving latest data till now
        recent_result.to_csv(f"BTCUSDT-{min}min-till-now.csv", index=False)

except Exception as e:
    logging.warning(e)

time_diff = datetime.now().minute

if time_diff < 15:
    logging.warning("Inside sleep")
    time.sleep((16 - time_diff) * 60)
elif time_diff < 30:
    logging.warning("Inside sleep")
    time.sleep((31 - time_diff) * 60)
elif time_diff < 60:
    logging.warning("Inside sleep")
    time.sleep((61 - time_diff) * 60)

twm.start_kline_socket(callback=trade, symbol="BTCUSDT", interval="1m")
twm.join()
