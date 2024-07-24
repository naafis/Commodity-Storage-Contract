import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA


class DataProcessor:
    def __init__(self, filepath):
        self.df = self.load_data(filepath)
        self.stationary_df = None

    def load_data(self, filepath):
        """Load CSV data from file path."""
        df = pd.read_csv(filepath, parse_dates=['Dates'], index_col='Dates')
        return df

    def plot_data(self, df, title='Natural Gas Prices'):
        """Plot the time series data."""
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(df, label='Natural Gas Prices')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def decompose_series(self):
        """Decompose the time series to identify trend and seasonality"""
        result = seasonal_decompose(self.df, model='multiplicative', extrapolate_trend='freq')
        result.plot()
        plt.show()

        # Extract the Components
        df_reconstructed = pd.concat([result.seasonal, result.trend, result.resid, result.observed], axis=1)
        df_reconstructed.columns = ['seasonal', 'trend', 'resid', 'actual_values']
        return df_reconstructed

    def plot_rolling_statistics(self):
        """Plot rolling mean and standard deviation"""
        rolling_mean = self.df.rolling(window=12).mean()
        rolling_std = self.df.rolling(window=12).std()

        plt.figure(figsize=(10, 6))
        plt.plot(self.df, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label='Rolling Mean')
        plt.plot(rolling_std, color='green', label='Rolling Std')
        plt.title('Rolling Mean and Standard Deviation')
        plt.legend()
        plt.show()

    def stationarity_tests(self, df):
        """Perform Augmented Dickey-Fuller test to check for stationarity"""
        print("Results of Dickey-Fuller Test:")
        result = adfuller(df, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critical Values:')
            print(f'    {key}, {value}')

        print("\nResults of KPSS Test:")
        result = kpss(df, regression='c')
        print(f'KPSS Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[3].items():
            print('Critical Values:')
            print(f'    {key}, {value}')

    def plot_acf_pacf(self, df):
        """Plot Autocorrelation and Partial Autocorrelation."""
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(df, lags=12, ax=plt.gca())
        plt.subplot(122)
        plot_pacf(df, lags=12, ax=plt.gca())
        plt.show()

    def difference_data(self, df, interval=1):
        """Apply differencing to make series stationary."""
        diff = df.diff(interval).dropna()
        return diff

    def log_transform(self, df):
        """Apply logarithmic transformation to stabilize variance."""
        return np.log(df)

    def seasonal_differencing(self, df, seasonal_lag=12):
        """Apply seasonal differencing to remove seasonal effects"""
        seasonal_diff = df.diff(seasonal_lag).dropna()
        return seasonal_diff

    def make_stationary(self):
        """Apply transformations to make the series stationary"""
        df_log = self.log_transform(self.df)
        self.plot_data(df_log, title='Log Transformed Data')

        df_log_diff = self.difference_data(df_log)
        self.plot_data(df_log_diff, title='Log Transformed and Differenced Data')

        df_log_diff_seasonal = self.seasonal_differencing(df_log_diff)
        self.plot_data(df_log_diff_seasonal, title='Log Transformed, Differenced, and Seasonally Difffernced Data')

        self.stationarity_tests(df_log_diff_seasonal)
        self.stationary_df = df_log_diff_seasonal

        return df_log_diff_seasonal

    def spline_interpolation(self, df=None):
        """Apply cubic spline interpolation to the data."""
        if df is None:
            df = self.df

        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        daily_df = df.reindex(daily_index)

        # Get numeric representation of dates for interpolation
        x = np.arange(len(df))
        y = df['Prices'].values

        # Create cubic spline interpolation
        spline = CubicSpline(x, y)

        # Generate new x values for the daily data
        x_new = np.arange(len(daily_df))

        # Interpolate the y values using the cubic spline
        daily_df_interpolated = pd.DataFrame(spline(x_new), index=daily_index, columns=['Prices'])
        return daily_df_interpolated

    def plot_interpolation(self, interpolated_df):
        """Plot original and interpolated data for comparison."""
        plt.figure(figsize=(15, 6))
        plt.plot(self.df, 'o', label='Original Data')
        plt.plot(interpolated_df, label='Spline Interpolation')
        plt.legend()
        plt.title('Spline Interpolation Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


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