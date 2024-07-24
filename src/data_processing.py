import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, kpss


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
