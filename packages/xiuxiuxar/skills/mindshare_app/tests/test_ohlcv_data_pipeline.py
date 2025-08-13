"""Test the technical indicators functions in DataCollectionRound."""

import time
import operator
from pathlib import Path
from unittest.mock import patch

import pytest
from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.behaviours import DataCollectionRound


class TestOhlcvDataPipeline(BaseSkillTestCase):
    """Test OHLCV data pipeline in DataCollectionRound."""

    path_to_skill = Path(__file__).parent.parent

    def setup_method(self):
        """Setup the test class."""
        super().setup()

        self.data_collection_round = DataCollectionRound(
            name="data_collection_round",
            skill_context=self.skill.skill_context,
        )
        self.logger = self.skill.skill_context.logger

        self.token_info = {
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "symbol": "USDC",
            "coingecko_id": "usd-coin",
        }

        # Generate 150 OHLCV data entries with 4-hour intervals, latest being 1 hour ago
        current_time_ms = int(time.time() * 1000)
        one_hour_ago_ms = current_time_ms - (1 * 60 * 60 * 1000)  # 1 hour ago
        four_hours_ms = 4 * 60 * 60 * 1000  # 4 hours in milliseconds

        self.valid_ohlcv_data = []

        # Generate 150 entries going backwards in time from 1 hour ago
        for i in range(150):
            timestamp = one_hour_ago_ms - (i * four_hours_ms)
            # Vary the OHLCV values slightly to make realistic data
            base_price = 16000.0 + (i * 10)  # Gradually increase price going back in time
            open_price = base_price + (i % 5) * 50
            high_price = open_price + 100 + (i % 3) * 50
            low_price = open_price - 100 - (i % 2) * 50
            close_price = open_price + (i % 7 - 3) * 25  # Some variation around open
            volume = 1000.0 + (i * 10)

            self.valid_ohlcv_data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        # Reverse the list so timestamps are in chronological order (oldest first)
        self.valid_ohlcv_data.reverse()

    def teardown_method(self):
        """Teardown the test class."""
        super().teardown()

    def test_process_ohlcv_data(self):
        """Test _process_ohlcv_data with valid data."""
        with (
            patch.object(self.data_collection_round, "_check_candle_data", return_value=True) as mock_check_candle_data,
            patch.object(self.data_collection_round, "_staleness_check", return_value=True) as mock_staleness_check,
        ):
            self.data_collection_round._process_ohlcv_data(self.valid_ohlcv_data, self.token_info)  # noqa: SLF001

            mock_check_candle_data.assert_called_once()
            mock_staleness_check.assert_called_once()

    def test_process_ohlcv_data_raises_error(self):
        """Test for _process_ohlcv_data to raise error."""

        # Test when _check_candle_data returns False
        with (
            patch.object(self.data_collection_round, "_check_candle_data", return_value=False),
            patch.object(self.data_collection_round, "_staleness_check", return_value=True),
            pytest.raises(ValueError, match="OHLCV data for USDC is not valid"),
        ):
            self.data_collection_round._process_ohlcv_data(self.valid_ohlcv_data, self.token_info)  # noqa: SLF001

        # Test when _staleness_check returns False
        with (
            patch.object(self.data_collection_round, "_check_candle_data", return_value=True),
            patch.object(self.data_collection_round, "_staleness_check", return_value=False),
            pytest.raises(ValueError, match="OHLCV data for USDC is stale"),
        ):
            self.data_collection_round._process_ohlcv_data(self.valid_ohlcv_data, self.token_info)  # noqa: SLF001

    def test_check_candle_data(self):
        """Test _check_candle_data with valid data."""
        result = self.data_collection_round._check_candle_data(self.valid_ohlcv_data)  # noqa: SLF001

        assert result is True

    def test_check_candle_data_not_enough_data(self):
        """Test _check_candle_data with less than 120 candles."""

        # Test to see if the function returns True when there are 120 or more candles
        result = self.data_collection_round._check_candle_data(self.valid_ohlcv_data)  # noqa: SLF001
        assert result is True

        # Reduce the number of candles to less than 120
        invalid_candles = self.valid_ohlcv_data[:10]
        result = self.data_collection_round._check_candle_data(invalid_candles)  # noqa: SLF001
        assert result is False

    def test_check_candle_data_too_old(self):
        """Test _check_candle_data with data older than 4 hours."""

        # Generate timestamps that are at least 5 hours old, with each previous timestamp 1 hour before
        current_time_ms = int(time.time() * 1000)
        five_hours_ago_ms = current_time_ms - (5 * 60 * 60 * 1000)  # 5 hours ago
        one_hour_ms = 60 * 60 * 1000  # 1 hour in milliseconds

        # Generate 150 candles, each 1 hour apart, with the latest being 5 hours old
        invalid_candles = []
        base_price = 16000.0
        base_volume = 1000.0

        for i in range(150):
            # Calculate timestamp (oldest first, newest last)
            hours_back = 149 - i  # Start from 154 hours ago, end at 5 hours ago
            timestamp = five_hours_ago_ms - (hours_back * one_hour_ms)

            # Generate realistic OHLCV data with some variation
            price_variation = (i % 20 - 10) * 10  # Price varies by ±100 over time
            open_price = base_price + price_variation + (i * 2)  # Slight upward trend
            high_price = open_price + (50 + (i % 10) * 5)  # High is 50-95 above open
            low_price = open_price - (30 + (i % 8) * 3)  # Low is 30-51 below open
            close_price = open_price + ((i % 15) - 7) * 8  # Close varies around open
            volume = base_volume + (i % 50) * 10  # Volume varies

            invalid_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])

        # Ensure candles are sorted by timestamp (oldest to newest)
        invalid_candles.sort(key=operator.itemgetter(0))

        result = self.data_collection_round._check_candle_data(invalid_candles)  # noqa: SLF001
        assert result is False

    def test_staleness_check(self):
        """Test _staleness_check with valid data."""
        result = self.data_collection_round._staleness_check(self.valid_ohlcv_data, self.token_info)  # noqa: SLF001

        assert result is True

    def test_staleness_check_stale_coin(self):
        """Test _staleness_check with stale coin."""

        # Generate 25 candles with the same closing price
        base_timestamp = 1751616000000  # Starting timestamp
        one_hour_ms = 60 * 60 * 1000  # 1 hour in milliseconds
        closing_price = 1.59  # Same closing price for all candles

        invalid_candles = []
        for i in range(25):
            timestamp = base_timestamp + (i * one_hour_ms)
            # Vary open, high, low but keep close constant
            open_price = closing_price + ((i % 10) - 5) * 0.01  # Small variation around close
            high_price = max(open_price, closing_price) + 0.02 + (i % 3) * 0.01  # Always higher
            low_price = min(open_price, closing_price) - 0.02 - (i % 3) * 0.01  # Always lower
            volume = 200000000 + (i % 20) * 10000000  # Varying volume

            invalid_candles.append([timestamp, open_price, high_price, low_price, closing_price, volume])

        result = self.data_collection_round._staleness_check(invalid_candles, self.token_info)  # noqa: SLF001

        assert result is False
