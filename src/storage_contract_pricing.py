import pandas as pd


class PricingModel:
    def __init__(self, model):
        """
        Initialize the PricingModel with a trained ARIMA model.
        :param model: A trained ARIMA model object that can forecast prices.
        """
        self.model = model

    def get_price_estimate(self, date):
        """
        Get the estimated price of gas on a specific date using the ARIMA model.
        :param date: Date for which the price estimate is needed.
        :return: Estimated price of gas on the given date.
        """
        # Ensure the date is within model's forecast range
        if date not in self.model.df.index:
            raise ValueError(f'Date {date} is not within the range of the available date.')

        # Fetch the forecast price
        forecast_index = self.model.df.index.get_loc(date)
        forecast_price = self.model.results.get_forecast(steps=forecast_index + 1).predicted_mean.iloc[-1]
        return forecast_price

    def calculate_contract_value(self, injection_dates, withdrawal_dates, injection_rate, withdrawal_rate,
                                 max_storage_volume, storage_cost):
        """
        Calculate the value of the gas storage contract.
        :param injection_dates: List of dates for gas injection.
        :param withdrawal_dates: List of dates for gas withdrawal.
        :param injection_rate: Rate at which gas is injected.
        :param withdrawal_rate: Rate at which gas is withdrawn.
        :param max_storage_volume: Maximum volume that can be stored.
        :param storage_cost: Monthly storage cost.
        :return: Net value of the contract.
        """
        if len(injection_dates) != len(withdrawal_dates):
            raise ValueError("The number of injection dates must be equal to the number of withdrawal dates.")

        total_purchase_cost = 0
        total_sale_revenue = 0
        total_storage_cost = 0
        total_injection_volume = 0

        for i in range(len(injection_dates)):
            injection_date = pd.to_datetime(injection_dates[i])
            withdrawal_date = pd.to_datetime(withdrawal_dates[i])

            # Calculate the volume injected and ensure it doesn't exceed storage capacity
            duration = (withdrawal_date - injection_date).days
            injection_volume = injection_rate * duration
            if total_injection_volume + injection_volume > max_storage_volume:
                raise ValueError('Injection volume exceeds the maximum storage volume.')
            total_injection_volume += injection_volume

            # Calculate the purchase cost
            purchase_price = self.get_price_estimate(injection_date)
            purchase_cost = purchase_price * injection_volume
            total_purchase_cost += purchase_cost

            # Calculate sale revenue
            withdrawal_volume = withdrawal_rate * duration
            sale_price = self.get_price_estimate(withdrawal_date)
            sale_revenue = sale_price * withdrawal_volume
            total_sale_revenue += sale_revenue

            # Calculate storage cost
            storage_duration = duration / 30    # Approximate number of months
            storage_cost_total = storage_cost * storage_duration
            total_storage_cost += storage_cost_total

        # Calculate net contract value
        contract_value = total_sale_revenue - (total_purchase_cost + total_storage_cost)
        return contract_value