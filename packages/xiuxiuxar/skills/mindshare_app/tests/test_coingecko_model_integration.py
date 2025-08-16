"""Integration tests for the Coingecko model."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko


load_dotenv()
COINGECKO_API_KEY = os.environ.get("SKILL_MINDSHARE_APP_MODELS_PARAMS_ARGS_COINGECKO_API_KEY")


@pytest.mark.integration
@pytest.mark.skipif(not COINGECKO_API_KEY, reason="CoinGecko API key not found in environment variables")
class TestCoingeckoModelIntegration(BaseSkillTestCase):
    """Test Coingecko model with real API calls."""

    path_to_skill = Path(__file__).parent.parent

    def setup(self):
        """Setup the test class."""
        super().setup()
        self.coingecko_model = Coingecko(
            name="coingecko_integration",
            skill_context=self.skill.skill_context,
        )
        self.coingecko_model.context.params.coingecko_api_key = COINGECKO_API_KEY
        self.logger = self.skill.skill_context.logger

    def test_coin_price_by_id_integration(self):
        """Test coin_price_by_id with a real API call."""
        query_params = {"ids": "bitcoin", "vs_currencies": "usd"}
        result = self.coingecko_model.coin_price_by_id(query_params)
        self.logger.info(f"Integration test (coin_price_by_id) result: {result}")

        assert result is not None
        assert "bitcoin" in result
        assert "usd" in result["bitcoin"]
        assert isinstance(result["bitcoin"]["usd"], int | float)

    def test_coin_ohlc_data_by_id_integration(self):
        """Test coin_ohlc_data_by_id with a real API call."""
        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}
        result = self.coingecko_model.coin_ohlc_data_by_id(path_params, query_params)
        self.logger.info(f"Integration test (coin_ohlc_data_by_id) result: {result}")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        # Check structure of the first element: [timestamp, open, high, low, close]
        assert isinstance(result[0], list)
        assert len(result[0]) == 5
        assert all(isinstance(val, int | float) for val in result[0])

    def test_coin_historical_chart_data_by_id_integration(self):
        """Test coin_historical_chart_data_by_id with a real API call."""
        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}
        result = self.coingecko_model.coin_historical_chart_data_by_id(path_params, query_params)
        self.logger.info(f"Integration test (coin_historical_chart_data_by_id) result: {result}")

        assert result is not None
        assert "prices" in result
        assert "market_caps" in result
        assert "total_volumes" in result
        assert isinstance(result["total_volumes"], list)

    def test_get_ohlcv_data_integration(self):
        """Test get_ohlcv_data with a real API call."""
        # This test relies on the other methods working correctly.
        self.coingecko_model.context.coingecko = self.coingecko_model

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}
        result = self.coingecko_model.get_ohlcv_data(path_params, query_params)
        self.logger.info(f"Integration test (get_ohlcv_data) result: {result}")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        # Check structure of the first element: [timestamp, open, high, low, close, volume]
        assert isinstance(result[0], list)
        assert len(result[0]) == 6
        assert all(isinstance(val, int | float) for val in result[0])
