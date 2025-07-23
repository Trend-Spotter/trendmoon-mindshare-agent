"""Test the Coingecko model."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko


coingecko_api_key = "test-key"


class TestCoingeckoModel(BaseSkillTestCase):
    """Test Coingecko model."""

    path_to_skill = Path(__file__).parent.parent

    def setup(self):
        """Setup the test class."""
        super().setup()

        self.coingecko_model = Coingecko(
            name="coingecko",
            skill_context=self.skill.skill_context,
        )
        self.logger = self.skill.skill_context.logger
        self.coingecko_model.coingecko_api_key = coingecko_api_key

    def test_set_api_key(self):
        """Test the set_api_key method."""
        self.coingecko_model.set_api_key("new_key")
        assert self.coingecko_model.coingecko_api_key == "new_key"

    def test_validate_required_params(self):
        """Test the validate_required_params method."""
        with pytest.raises(ValueError, match="path_params is required"):
            self.coingecko_model.validate_required_params(None, ["id"], "path_params")

        with pytest.raises(ValueError, match="id is required in path_params"):
            self.coingecko_model.validate_required_params({"currency": "usd", "days": "1"}, ["id"], "path_params")

        # This should not raise an error
        self.coingecko_model.validate_required_params({"id": "bitcoin"}, ["id"], "path_params")

    def test_make_coingecko_request(self):
        """Test make_coingecko_request with incorrect/missing API key."""
        # Test with None API key
        self.coingecko_model.coingecko_api_key = None
        with pytest.raises(ValueError, match="Coingecko API key is not set"):
            self.coingecko_model.make_coingecko_request(
                "https://api.coingecko.com/", {"vs_currency": "usd", "days": "1"}
            )

        self.coingecko_model.coingecko_api_key = ""
        # Test with empty string API key
        with pytest.raises(ValueError, match="Coingecko API key is not set"):
            self.coingecko_model.make_coingecko_request(
                "https://api.coingecko.com/", {"vs_currency": "usd", "days": "1"}
            )

        # Reset API key for subsequent tests and test with valid key
        self.coingecko_model.coingecko_api_key = coingecko_api_key
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "OK"}
            mock_get.return_value = mock_response

            result = self.coingecko_model.make_coingecko_request("https://api.coingecko.com/api/v3/ping", {})
            assert result is not None
            assert result == {"status": "OK"}

    @patch("requests.get")
    def test_make_coingecko_request_empty_query_params(self, mock_get):
        """Test make_coingecko_request with empty query params."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}
        mock_get.return_value = mock_response

        result = self.coingecko_model.make_coingecko_request("https://api.coingecko.com/api/v3/ping", {})
        assert result is not None
        assert result == {"status": "OK"}

    @patch("requests.get")
    def test_coin_ohlc_data_by_id_success(self, mock_get):
        """Test coin_ohlc_data_by_id success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[1672531200000, 16572.4, 16630.5, 16512.2, 16625.3]]
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko_model.coin_ohlc_data_by_id(path_params, query_params)
        self.logger.info(f"coin_ohlc_data_by_id result: {result}")

        assert result is not None
        assert len(result) == 1
        assert result[0][0] == 1672531200000
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_coin_ohlc_data_by_id_failure(self, mock_get):
        """Test coin_ohlc_data_by_id failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko_model.coin_ohlc_data_by_id(path_params, query_params)

        assert result is None

    @patch("requests.get")
    def test_coin_historical_chart_data_by_id_success(self, mock_get):
        """Test coin_historical_chart_data_by_id success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prices": [], "market_caps": [], "total_volumes": []}
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko_model.coin_historical_chart_data_by_id(path_params, query_params)

        assert result is not None
        assert "total_volumes" in result
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_coin_historical_chart_data_by_id_failure(self, mock_get):
        """Test coin_historical_chart_data_by_id failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        path_params = {"id": "bitcoin"}
        query_params = {"vs_currency": "usd", "days": "1"}

        result = self.coingecko_model.coin_historical_chart_data_by_id(path_params, query_params)

        assert result is None

    @patch("requests.get")
    def test_coin_price_by_id_success(self, mock_get):
        """Test coin_price_by_id success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bitcoin": {"usd": 60000}}
        mock_get.return_value = mock_response

        query_params = {"ids": "bitcoin", "vs_currencies": "usd"}

        result = self.coingecko_model.coin_price_by_id(query_params)

        assert result is not None
        assert result["bitcoin"]["usd"] == 60000
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_coin_price_by_id_failure(self, mock_get):
        """Test coin_price_by_id failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        query_params = {"ids": "bitcoin", "vs_currencies": "usd"}

        result = self.coingecko_model.coin_price_by_id(query_params)

        assert result is None

    def test_get_ohlcv_data_success(self):
        """Test get_ohlcv_data success."""
        ohlc_data = [[1672531200000, 1, 2, 3, 4]]
        volume_data = {"total_volumes": [[1672531200000, 100]]}

        with (
            patch.object(self.coingecko_model, "coin_ohlc_data_by_id", return_value=ohlc_data) as mock_ohlc,
            patch.object(
                self.coingecko_model, "coin_historical_chart_data_by_id", return_value=volume_data
            ) as mock_volume,
        ):
            self.coingecko_model.context.coingecko = self.coingecko_model
            result = self.coingecko_model.get_ohlcv_data({}, {})

            assert result is not None
            assert len(result) == 1
            assert result[0] == [1672531200000, 1, 2, 3, 4, 100]
            mock_ohlc.assert_called_once()
            mock_volume.assert_called_once()

    def test_get_ohlcv_data_missing_data(self):
        """Test get_ohlcv_data with missing data."""
        with (
            patch.object(self.coingecko_model, "coin_ohlc_data_by_id", return_value=None),
            patch.object(self.coingecko_model, "coin_historical_chart_data_by_id", return_value=None),
        ):
            self.coingecko_model.context.coingecko = self.coingecko_model
            result = self.coingecko_model.get_ohlcv_data({}, {})

            assert result is None
