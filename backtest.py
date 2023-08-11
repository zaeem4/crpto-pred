
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
from finta import TA
# from matplotlib import pyplot
from datetime import datetime, timedelta

with open('serialized_model_full_15min_till_07_newton.json', 'r') as fin:
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


test = pd.read_csv('BTCUSDT-15min-2023-08.csv', #, skiprows=[i for i in range(0, 202367) if i % 4 != 0]
        # names=COLUMNS,
        # header=None,
    )

test = test.dropna()
df = pd.read_csv('BTCUSDT-15min-till-07.csv', #, skiprows=[i for i in range(0, 202367) if i % 4 != 0]
        # names=COLUMNS,
        # header=None,
    )


# print(df.tail())

result = {}
prev_time = None

up_count = 0
down_count = 0

other_count = 0

diff_sum = 0

inventory = []

total_profit = 0

for test_index in test.index.tolist():
  # print(test_index)
  # print(df.tail())


  df = df.append(test.loc[test_index], ignore_index=True)

  # df['open_time'] = df['ds']
  # df['open'] = test.loc[test_index]['open']
  
  df['Returns'] = df.close.pct_change()
  df['Log Returns'] = np.log(1+df['Returns'])

  df.fillna(0,inplace=True)

  df['RSI'] = TA.RSI(df)
  df['MACD'] = TA.VWAP(df)
  df['SMA'] = TA.SMA(df)
  df['BBANDS'] = TA.ROC(df)
  df['EMA'] = TA.EMA(df)

  # print(df['open_time'])

  # test['Returns'] = test.close.pct_change()
  # test['Log Returns'] = np.log(1+test['Returns'])

  # test.fillna(0,inplace=True)

  # test['RSI'] = TA.RSI(test)
  # test['MACD'] = TA.VWAP(test)
  # test['SMA'] = TA.SMA(test)
  # test['BBANDS'] = TA.ROC(test)
  # test['EMA'] = TA.EMA(test)

  test.fillna(0, inplace=True)

  future = pd.DataFrame()

  future['ds'] = [pd.to_datetime(df.iloc[-1]['open_time'], unit="ms") + timedelta(minutes=15)]

  future['y'] = [df.iloc[-1]['open']]

  future['high'] = [df.iloc[-1]['high']]
  future['low'] = [df.iloc[-1]['low']]
  future['close'] = [df.iloc[-1]['close']]
  future['volume'] = [df.iloc[-1]['volume']]
  future['Returns'] = [df.iloc[-1]['Returns']]
  future['Log Returns'] = [df.iloc[-1]['Log Returns']]
  future['BBANDS'] = [df.iloc[-1]['BBANDS']]
  future['RSI'] = [df.iloc[-1]['RSI']]
  future['MACD'] = [df.iloc[-1]['MACD']]
  future['SMA'] = [df.iloc[-1]['SMA']]
  future['EMA'] = [df.iloc[-1]['EMA']]

  # print(future.head())


  # forecast_data = model.make_future_dataframe(periods=36, freq= 60 * 15)

  forecast = model.predict(future)

  current_time = f"{pd.to_datetime(df.iloc[-1]['open_time'], unit='ms')}"
  next_time = f"{pd.to_datetime(df.iloc[-1]['open_time'], unit='ms') + timedelta(minutes=15)}"

  if current_time in result:
    result[current_time]['current'] = df.iloc[-1]['open']

  else:
    result[current_time] = {}
    result[current_time]['current'] = df.iloc[-1]['open']

  if next_time in result:
    result[next_time]['predicted'] = forecast.iloc[0]['yhat']
  else:
    result[next_time] = {}
    result[next_time]['predicted'] = forecast.iloc[0]['yhat']

  if 'current' in result[current_time] and 'predicted' in result[current_time] :
    result[current_time]['diff'] = result[current_time]['predicted'] - result[current_time]['current']

    # if df['open_time'][current_index] > date_time_obj:
    #   if result[current_time]['diff'] > -100 and result[current_time]['diff'] < 100:
    #     range_count = range_count + 1
      # if result[df['open_time'][current_index]]['current'] > result[forecast['ds'][forecast_index]]['predicted']:

    diff_sum += abs(result[current_time]['diff'])
  else:
    result[current_time]['diff'] = 0

  # print("cyrrent", current_time)
  # print("Result",result)
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

    print(
        f"Buying after 15 mins {list(result.items())[-3:]} | {pred_15min_trend}"
    )

    if (
        "current" in result[pred_15min_trend[0]]
        and "current" in result[pred_15min_trend[1]]
        and "predicted" in result[pred_15min_trend[1]]
        and "predicted" in result[pred_15min_trend[2]]
    ):
        prev_value_of_15min_trend = result[pred_15min_trend[0]][
            "current"
        ]

        current_value_of_15min_trend = result[pred_15min_trend[1]][
            "current"
        ]

        current_pred_value_of_15min_trend = result[
            pred_15min_trend[1]
        ]["predicted"]

        next_pred_value_of_15min_trend = result[pred_15min_trend[2]][
            "predicted"
        ]

        buy_or_sell = False

        if (
            next_pred_value_of_15min_trend > current_value_of_15min_trend
            and next_pred_value_of_15min_trend
            > current_pred_value_of_15min_trend
        ):
            buy_value = 1 / current_value_of_15min_trend * 200
            inventory.append(
                {
                    "open_time": f'{current_time}',
                    "actual_open": current_value_of_15min_trend,
                    "open": buy_value,
                    "with_tx_fee": buy_value - (buy_value * 0.001),
                    # "time": pred_15min_trend,
                }
            )

            # print(f"signal 'Buy' at {inventory[-1]}")
            buy_or_sell = True

        if (
            next_pred_value_of_15min_trend < current_value_of_15min_trend
            or next_pred_value_of_15min_trend
            < current_pred_value_of_15min_trend
            or current_value_of_15min_trend < prev_value_of_15min_trend
        ):
            sell_value = 1 / current_value_of_15min_trend * 200
            sell_value_with_tx_fee = sell_value - (sell_value * 0.001)
            
            inventory_to_remove = []

            for index in range(len(inventory)):
                current_inventory = inventory[index]

                profit = (
                    sell_value_with_tx_fee - current_inventory["with_tx_fee"]
                )

                if profit > 0:
                    temp = {
                        "open_time": f'{current_time}',
                        "actual_open": current_value_of_15min_trend,
                        "open": sell_value,
                        "with_tx_fee": sell_value_with_tx_fee,
                        "profit": profit,
                        "profit_usdt":current_value_of_15min_trend * profit,
                    }

                    inventory_to_remove.append(current_inventory)
                    total_profit = total_profit+temp["profit_usdt"]
                    print(f"signal 'Sell' at {temp}")

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

        # if not buy_or_sell:
        #     print(
        #         f"signal 'do nothing' current_actual {current_value_of_15min_trend} current_pred {current_pred_value_of_15min_trend} next_pred {next_pred_value_of_15min_trend}"
        #     )

        prev_actual_value = result[prev_time]['current']
        current_actual_value = result[current_time]['current']
        current_predicted_value = result[current_time]['predicted']

        actual_diff = current_actual_value - prev_actual_value
        predicted_diff = current_predicted_value - prev_actual_value

        # if df['open_time'][test_index] > date_time_obj:
        if actual_diff > 0 and predicted_diff > 0:
          up_count = up_count + 1
        elif actual_diff <= 0 and predicted_diff <= 0:
          down_count = down_count + 1
        else:
          other_count = other_count + 1

        if (other_count+up_count+down_count) > 0:
          print(f"{prev_time} | {current_time} | {pred_15min_trend} | up_count {up_count} | down_count {down_count} | wrong | {other_count} | total_correct {up_count+down_count} | mean_diff | {diff_sum/(other_count+up_count+down_count)}  ")
        # print("Current", df['open_time'][current_index],df['open'][current_index] )
        # print("Predicted", forecast['ds'][forecast_index], forecast['yhat'][forecast_index])

    # else:
        # print(
        #     f"Data is incomplete in {result} for {pred_15min_trend}"
        # )
  else:
    prev_time = current_time
  # print(f"waiting for complete data | {result}")

  # if not prev_time:
  #   prev_time = current_time
  # else:
    
  #   # print(result)

  #   prev_actual_value = result[prev_time]['current']
  #   current_actual_value = result[current_time]['current']
  #   current_predicted_value = result[current_time]['predicted']

  #   actual_diff = current_actual_value - prev_actual_value
  #   predicted_diff = current_predicted_value - prev_actual_value

  #   # if df['open_time'][test_index] > date_time_obj:
  #   if actual_diff > 0 and predicted_diff > 0:
  #     up_count = up_count + 1
  #   elif actual_diff <= 0 and predicted_diff <= 0:
  #     down_count = down_count + 1
  #   else:
  #     other_count = other_count + 1
  #else:
    #break;



print(total_profit)
print(inventory)
print(len(inventory))

  # result = {}

  # prev_time = ""

  # up_count = 0
  # down_count = 0

  # other_count = 0

  # diff_sum = 0

  # date_time_str = "2023-06-03 00:00:00"
  # date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

  # range_count = 0

  # for current_index,forecast_index in zip(df.index,forecast.index):

  #   current_time = f"{df['open_time'][current_index]}"
  #   next_time = f"{forecast['ds'][forecast_index]}"

  #   if current_time in result:
  #     result[current_time]['current'] = df['open'][current_index]

  #   else:
  #     result[current_time] = {}
  #     result[current_time]['current'] = df['open'][current_index]

  #   if next_time in result:
  #     result[next_time]['predicted'] = forecast['yhat'][forecast_index]
  #   else:
  #     result[next_time] = {}
  #     result[next_time]['predicted'] = forecast['yhat'][forecast_index]

  #   if 'current' in result[current_time] and 'predicted' in result[current_time] :
  #     result[current_time]['diff'] = result[current_time]['predicted'] - result[current_time]['current']

  #     if df['open_time'][current_index] > date_time_obj:
  #       diff_sum += abs(result[current_time]['diff'])
  #       if result[current_time]['diff'] > -100 and result[current_time]['diff'] < 100:
  #         range_count = range_count + 1
  #       # if result[df['open_time'][current_index]]['current'] > result[forecast['ds'][forecast_index]]['predicted']:

  #   else:
  #     result[current_time]['diff'] = 0

  #   if current_index == 0:
  #     prev_time = current_time
  #   else:
  #     prev_actual_value = result[prev_time]['current']
  #     current_actual_value = result[current_time]['current']
  #     current_predicted_value = result[current_time]['predicted']

  #     actual_diff = current_actual_value - prev_actual_value
  #     predicted_diff = current_predicted_value - prev_actual_value

  #     if df['open_time'][current_index] > date_time_obj:
  #       if actual_diff > 0 and predicted_diff > 0:
  #         up_count = up_count + 1
  #       elif actual_diff <= 0 and predicted_diff <= 0:
  #         down_count = down_count + 1
  #       else:
  #         other_count = other_count + 1


  # # print(result)
  # print(f"up_count {up_count} | down_count {down_count} | wrong | {other_count} | total_correct {up_count+down_count} | mean_diff | {diff_sum/(other_count+up_count+down_count)} | rane_count {range_count} ")
  #   # print("Current", df['open_time'][current_index],df['open'][current_index] )
  #   # print("Predicted", forecast['ds'][forecast_index], forecast['yhat'][forecast_index])


  # # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
  # # print(df[['open_time','open']].head())
  # y_true = future['y']
  # # y_true.pop(0)

  # # y_true = future['open'].tolist().pop(0)
  # y_pred = forecast[['yhat']]

  # # mae = mean_absolute_error(y_true, y_pred)
  # # print('MAE: %.3f' % mae)

  # fig = pyplot.figure(figsize = (30, 15))
  # pyplot.plot( y_true, label='Actual')
  # pyplot.plot(y_pred,  label='Predicted')
  # pyplot.legend()
  # pyplot.show()

