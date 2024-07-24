import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


class Model:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.results = None

    def train_arima(self, order):
        """Train ARIMA model on the data."""
        self.model = ARIMA(self.df, order=order)
        self.results = self.model.fit()

    def forecast(self, steps):
        """Forecast future values using trained ARIMA model."""
        forecast = self.results.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        return mean_forecast, confidence_intervals

    def plot_forecast(self, mean_forecast, confidence_intervals):
        """Plot the original data and the forecasted values with confidence intervals."""
        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(self.df, label='Historical Data')
        plt.plot(mean_forecast, label='Forecasted Data')
        plt.fill_between(confidence_intervals.index,
                         confidence_intervals.iloc[:, 0],
                         confidence_intervals.iloc[:, 1], color='k', alpha=0.2)
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Natural Gas Prices')
        plt.legend()
        plt.show()

    def get_forecast_for_date(self, date, steps, start_date):
        """Get forecast for a specific date."""
        forecast_index = pd.date_range(start=start_date, periods=steps, freq='D')
        forecast = self.results.get_forecast(steps=steps)
        mean_forecast = forecast_index.predicted_mean
        forecast_series = pd.Series(mean_forecast, index=forecast_index)
        return forecast_series.get(date, "Date out of forecast range.")
