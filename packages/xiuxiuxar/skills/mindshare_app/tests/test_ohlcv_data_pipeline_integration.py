"""Integration test for OHLCV data pipeline."""

import time
from pathlib import Path

from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.behaviours import DataCollectionRound


class TestOhlcvDataPipelineIntegration(BaseSkillTestCase):
    """Test OHLCV data pipeline with real data processing."""

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

    def test_process_ohlcv_data_integration(self):
        """Test _process_ohlcv_data with real OHLCV data."""
        # Assert that no ValueError exception is raised
        try:
            self.data_collection_round._process_ohlcv_data(self.valid_ohlcv_data, self.token_info)  # noqa: SLF001
            # If we get here, no exception was raised
            no_exception_raised = True
        except ValueError:
            no_exception_raised = False

        assert no_exception_raised, "ValueError should not be raised for process_ohlcv_data"
