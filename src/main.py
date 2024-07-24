import pandas as pd
from data_processing import DataProcessor
from model import Model
from storage_contract_pricing import PricingModel

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
    model.train_arima_model(order=arima_order)

    # Initialize PricingModel with the trained ARIMA model
    pricing_model = PricingModel(model)

    # Example usage of pricing model (replace with actual user input)
    injection_dates = ['2023-01-15', '2023-04-15']
    withdrawal_dates = ['2023-07-15', '2023-10-15']
    injection_rate = 1e6  # 1 million MMBtu per day
    withdrawal_rate = 1e6  # 1 million MMBtu per day
    max_storage_volume = 5e6  # 5 million MMBtu
    storage_cost = 100000  # $100,000 per month

    # Calculate contract value
    contract_value = pricing_model.calculate_contract_value(
        injection_dates,
        withdrawal_dates,
        injection_rate,
        withdrawal_rate,
        max_storage_volume,
        storage_cost
    )
    print(f"Estimated contract value: ${contract_value:.2f}")

if __name__ == "__main__":
    main()
