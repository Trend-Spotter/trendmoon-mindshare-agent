"""Test the models in the mindshare_app skill."""

from typing import cast
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app import PUBLIC_ID
from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko


ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent


class TestCoingecko(BaseSkillTestCase):
    """Test CoinGecko model."""

    path_to_skill = Path(ROOT_DIR, "packages", PUBLIC_ID.author, "skills", PUBLIC_ID.name)

    @classmethod
    def setup(cls):  # pylint: disable=W0221
        """Setup the test class."""
        super().setup_class()
        cls.coingecko = cast(Coingecko, cls._skill.skill_context.coingecko)
        cls.coingecko_api_key = "test_api_key"
        cls.coingecko.coingecko_api_key = cls.coingecko_api_key

    def test_init(self):
        """Test the initialization of CoinGecko model."""
        # Test with API key
        coingecko_with_key = Coingecko(skill_context=self._skill.skill_context, coingecko_api_key="test_key")
        assert coingecko_with_key.coingecko_api_key == "test_key"

        # Test without API key (should default to empty string)
        coingecko_without_key = Coingecko(skill_context=self._skill.skill_context)
        assert coingecko_without_key.coingecko_api_key == ""

    def test_validate_required_params_success(self):
        """Test validate_required_params with valid parameters."""
        params = {"id": "bitcoin", "vs_currency": "usd"}
        required_keys = ["id", "vs_currency"]

        # Should not raise any exception
        self.coingecko.validate_required_params(params, required_keys, "test_params")

    def test_validate_required_params_empty_params(self):
        """Test validate_required_params with empty parameters."""
        required_keys = ["id", "vs_currency"]

        # Test with None
        with pytest.raises(ValueError, match="test_params is required"):
            self.coingecko.validate_required_params(None, required_keys, "test_params")

        # Test with empty dict
        with pytest.raises(ValueError, match="test_params is required"):
            self.coingecko.validate_required_params({}, required_keys, "test_params")

    def test_validate_required_params_missing_key(self):
        """Test validate_required_params with missing required key."""
        params = {"id": "bitcoin"}
        required_keys = ["id", "vs_currency"]

        with pytest.raises(ValueError, match="vs_currency is required in test_params"):
            self.coingecko.validate_required_params(params, required_keys, "test_params")

    def test_validate_required_params_none_value(self):
        """Test validate_required_params with None value."""
        params = {"id": "bitcoin", "vs_currency": None}
        required_keys = ["id", "vs_currency"]

        with pytest.raises(ValueError, match="vs_currency is required in test_params"):
            self.coingecko.validate_required_params(params, required_keys, "test_params")

    @patch("requests.get")
    def test_coin_ohlc_data_by_id_success(self, mock_get):
        """Test successful OHLC data retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            [1609459200000, 29374.152, 29600.625, 28803.586, 29374.152],
            [1609545600000, 29374.152, 31875.000, 28850.000, 31875.000],
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko.coin_ohlc_data_by_id(path_params, query_params)

        # Verify the request was made correctly
        expected_url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
        expected_headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

        mock_get.assert_called_once_with(expected_url, headers=expected_headers, timeout=10)
        mock_response.raise_for_status.assert_called_once()

        # Verify the result
        assert len(result) == 2
        assert result[0][0] == 1609459200000

    @patch("requests.get")
    def test_coin_ohlc_data_by_id_request_exception(self, mock_get):
        """Test OHLC data retrieval with request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        with patch("logging.exception") as mock_logging:
            result = self.coingecko.coin_ohlc_data_by_id(path_params, query_params)

            # Should return empty list on error
            assert result == []
            mock_logging.assert_called_once()

    def test_coin_ohlc_data_by_id_invalid_params(self):
        """Test OHLC data retrieval with invalid parameters."""
        # Test missing path params
        with pytest.raises(ValueError, match="id is required in path_params"):
            self.coingecko.coin_ohlc_data_by_id({}, {"vs_currency": "usd", "days": "1"})

        # Test missing query params
        with pytest.raises(ValueError, match="vs_currency is required in query_params"):
            self.coingecko.coin_ohlc_data_by_id({"id": "bitcoin"}, {"days": "1"})

    @patch("requests.get")
    def test_coin_historical_chart_data_by_id_success(self, mock_get):
        """Test successful historical chart data retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "prices": [[1609459200000, 29374.152], [1609545600000, 31875.000]],
            "market_caps": [[1609459200000, 547184700000], [1609545600000, 594000000000]],
            "total_volumes": [[1609459200000, 50000000000], [1609545600000, 60000000000]],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko.coin_historical_chart_data_by_id(path_params, query_params)

        # Verify the request was made correctly
        expected_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
        expected_headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

        mock_get.assert_called_once_with(expected_url, headers=expected_headers, timeout=10)
        mock_response.raise_for_status.assert_called_once()

        # Verify the result
        assert "prices" in result
        assert "market_caps" in result
        assert "total_volumes" in result
        assert len(result["prices"]) == 2

    @patch("requests.get")
    def test_coin_historical_chart_data_by_id_request_exception(self, mock_get):
        """Test historical chart data retrieval with request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        with patch("logging.exception") as mock_logging:
            result = self.coingecko.coin_historical_chart_data_by_id(path_params, query_params)

            # Should return empty list on error
            assert result == []
            mock_logging.assert_called_once()

    def test_coin_historical_chart_data_by_id_invalid_params(self):
        """Test historical chart data retrieval with invalid parameters."""
        # Test missing path params
        with pytest.raises(ValueError, match="id is required in path_params"):
            self.coingecko.coin_historical_chart_data_by_id({}, {"vs_currency": "usd", "days": "1"})

        # Test missing query params
        with pytest.raises(ValueError, match="days is required in query_params"):
            self.coingecko.coin_historical_chart_data_by_id({"id": "bitcoin"}, {"vs_currency": "usd"})

    @patch("requests.get")
    def test_coin_price_by_id_success(self, mock_get):
        """Test successful price data retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"bitcoin": {"usd": 45000.0, "eur": 37000.0}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        query_params = {"ids": "bitcoin", "vs_currencies": "usd,eur"}

        result = self.coingecko.coin_price_by_id(query_params)

        # Verify the request was made correctly
        expected_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,eur"
        expected_headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

        mock_get.assert_called_once_with(expected_url, headers=expected_headers, timeout=10)
        mock_response.raise_for_status.assert_called_once()

        # Verify the result
        assert "bitcoin" in result
        assert result["bitcoin"]["usd"] == 45000.0
        assert result["bitcoin"]["eur"] == 37000.0

    @patch("requests.get")
    def test_coin_price_by_id_request_exception(self, mock_get):
        """Test price data retrieval with request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        query_params = {"ids": "bitcoin", "vs_currencies": "usd"}

        with patch("logging.exception") as mock_logging:
            result = self.coingecko.coin_price_by_id(query_params)

            # Should return empty dict on error
            assert result == {}
            mock_logging.assert_called_once()

    def test_coin_price_by_id_invalid_params(self):
        """Test price data retrieval with invalid parameters."""
        # Test missing query params
        with pytest.raises(ValueError, match="vs_currencies is required in query_params"):
            self.coingecko.coin_price_by_id({"ids": "bitcoin"})

        # Test with empty params
        with pytest.raises(ValueError, match="query_params is required"):
            self.coingecko.coin_price_by_id({})

    @classmethod
    def teardown(cls, *args, **kwargs):  # noqa
        """Teardown the test class."""
        super().teardown_class()
        # Clean up any test artifacts if needed
        db_fn = Path("test.db")
        if db_fn.exists():
            db_fn.unlink()
