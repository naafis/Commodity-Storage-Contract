import pandas as pd
from data_processing import DataProcessor
from model import Model

def main():
    # Load and preprocess data
    processor = DataProcessor('Nat_Gas.csv')
    processor.plot_data(processor.df)
    processor.decompose_series()
    processor.plot_rolling_statistics()

    # Interpolate to daily data
    daily_data = processor.spline_interpolation()
    processor.plot_interpolation(daily_data)

    # Make data stationary
    stationary_data = processor.make_stationary()

    # Train ARIMA model
    model = Model(stationary_data)
    arima_order = (1, 1, 1)
    model.train_arima(order=arima_order)

    # Get user import for the date
    date_input = input("Enter the date for price estimation (YYYY-MM-DD:")
    try:
        date = pd.to_datetime(date_input)
    except ValueError:
        print("Invalid date format. Please enter in YYYY-MM-DD format.")
        return

    # Forecast steps based on how far into the future the date is
    last_date = stationary_data.index[-1]
    steps = (date.year - last_date.year) * 12 + (date.month - last_date.month)

    # Get forecast for specified date
    if steps > 0:
        mean_forecast, _ = model.forecast(steps)
        future_index = pd.date_range(start=last_date, periods=steps, freq='D')
        future_data = pd.Series(mean_forecast, index=future_index)
        daily_forecast = processor.spline_interpolation(future_data.to_frame('Prices'))
        forecast_price = daily_forecast.get(date, "Date out of forecast range.")
        print(f"Estimated price for {date_input}: {forecast_price}")
    else:
        daily_data = processor.spline_interpolation()
        forecast_price = daily_data.get(date, "Date not found in the data.")
        print(f"Price for {date_input}: {forecast_price}")


if __name__ == "__main__":
    main()