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
count = [0,0]

twm = ThreadedWebsocketManager()
twm.start()


def trade(price):
    try:
        global inventory

        global result

        global last_time
        global return_signal

        # global model_15_mins
        # global model_30_mins
        global model_60_mins

        global count

        if last_time is not None:
            minutes = get_minutes_diff(last_time)
            if minutes not in [60]:
                return_signal = False
                return

            if minutes in [60] and return_signal == True:
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
                    f"BTCUSDT-60min-1year-till-now.csv",
                )
            ).dropna()

            actual_dataframe = actual_dataframe.loc[
                :, ~actual_dataframe.columns.str.contains("^Unnamed")
            ]

            recent_result = pd.concat([actual_dataframe, interval_60min_data])

            recent_result = recent_result.loc[
                :, ~recent_result.columns.str.contains("^Unnamed")
            ]

            recent_result.to_csv("BTCUSDT-60min-1year-till-now", index=False)

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
                f"Inside 60 mintes bianance current is {current_time} server current is {datetime.now()}"
            )

        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________
        #                                  now calculating results
        # ___________________________________________________________________________________________________
        # ___________________________________________________________________________________________________

        if len(result["60mins"].keys()) > 2:
            # ___________________________________________________________________________________________________
            # ___________________________________________________________________________________________________
            #                                 60 mins buy/sell/hold
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

            pred_60min_trend = list(result["60mins"])[-3:]

            logging.warning(
                f"Buying after 60 mins {list(result['60mins'].items())[-3:]} | {pred_60min_trend}"
            )

            if (
                "actual" in result["60mins"][pred_60min_trend[0]]
                and "actual" in result["60mins"][pred_60min_trend[1]]
                and "pred" in result["60mins"][pred_60min_trend[1]]
                and "pred" in result["60mins"][pred_60min_trend[2]]
            ):
                prev_value_of_60min_trend = result["60mins"][pred_60min_trend[0]][
                    "actual"
                ]

                current_value_of_60min_trend = result["60mins"][pred_60min_trend[1]][
                    "actual"
                ]

                current_pred_value_of_60min_trend = result["60mins"][
                    pred_60min_trend[1]
                ]["pred"]

                next_pred_value_of_60min_trend = result["60mins"][pred_60min_trend[2]][
                    "pred"
                ]

                buy_or_sell = False

                if (
                    next_pred_value_of_60min_trend > current_value_of_60min_trend
                ):
                    buy_value =  new_row["open"] * 0.01

                    inventory.append(
                        {
                            "open_time": f'{datetime.fromtimestamp(new_row["open_time"] / 1000)}',
                            "actual_open": new_row["open"],
                            "open": buy_value,
                            "with_tx_fee": buy_value + (buy_value * 0.001),
                            # "time": pred_15min_trend,
                        }
                    )

                    logging.warning(f"signal 'Buy' at {inventory[-1]}")
                    buy_or_sell = True

                if (
                    next_pred_value_of_60min_trend < current_value_of_60min_trend
                    or current_value_of_60min_trend < prev_value_of_60min_trend
                ):
                    sell_value = new_row["open"] * 0.01
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
                            }

                            inventory_to_remove.append(current_inventory)

                            logging.warning(f"signal 'Sell' at {temp}")
                            buy_or_sell = True

                    inventory = [i for i in inventory if i not in inventory_to_remove]

                if not buy_or_sell:
                    logging.warning(
                        f"signal 'do nothing' | current_actual {current_value_of_60min_trend} | current_pred {current_pred_value_of_60min_trend} | next_pred {next_pred_value_of_60min_trend} | inventory {inventory}"
                    )
            
                prev_actual = prev_value_of_60min_trend
                current_value = current_value_of_60min_trend
                pred_value = current_pred_value_of_60min_trend

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
                    f"Data is incomplete in {result['60mins']} for {pred_60min_trend}"
                )
            
        else:
            logging.warning(f"waiting for complete data | {result}")

    except Exception as e:
        logging.warning(f"Error: {e}")


try:
    # reading dataset
    actual_dataframe = (
        pd.read_csv(
            f"BTCUSDT-60min-1year-till-07.csv",
        )
    ).dropna()

    # getting no of days passed
    diff = get_minutes_diff(datetime(2023, 8, 1, 00, 00, 00))

    # getting data from aug till now
    
    new_data = pd.DataFrame(
        json.loads(
            requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": f"1h",
                    "limit": int(diff / 60),
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
    recent_result.to_csv(f"BTCUSDT-60min-1year-till-now.csv", index=False)

except Exception as e:
    logging.warning(e)

time_diff = datetime.now().minute

# if time_diff > 1 and time_diff < 60:
#     logging.warning("Inside sleep")
#     time.sleep((60 - time_diff) * 60)
    
twm.start_kline_socket(callback=trade, symbol="BTCUSDT", interval="1m")
twm.join()
