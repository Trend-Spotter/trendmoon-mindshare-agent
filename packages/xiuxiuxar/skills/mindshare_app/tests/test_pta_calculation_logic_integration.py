"""Integration tests for technical indicators calculation."""

from pathlib import Path

from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.behaviours import DataCollectionRound


class TestTechnicalIndicatorsIntegration(BaseSkillTestCase):
    """Test technical indicators calculation with real data processing."""

    path_to_skill = Path(__file__).parent.parent

    def setup_method(self):
        """Setup the test class."""
        super().setup()

        self.data_collection_round = DataCollectionRound(
            name="data_collection_round",
            skill_context=self.skill.skill_context,
        )
        self.logger = self.skill.skill_context.logger

        self.technical_data_all_conditions_met = {
            "SMA": 100.0,  # Current price (105.0) > SMA
            "EMA": 101.0,  # Current price (105.0) > EMA
            "RSI": 50.0,  # RSI in range (30 < 50 < 70)
            "MACD": {  # MACD structure as dict
                "MACD": 10.0,  # MACD line
                "MACDs": 8.0,  # Signal line (MACD > Signal)
                "MACDh": 2.0,  # Histogram
            },
            "ADX": 50.0,  # ADX > threshold (25)
            "BB": {  # Bollinger Bands as dict
                "Lower": 95.0,
                "Middle": 102.0,
                "Upper": 110.0,
            },
            "OBV": 10000.0,
            "CMF": 0.15,
        }

    def teardown_method(self):
        """Teardown the test class."""
        super().teardown()

    def test_calculate_ta_score_integration(self):
        """Test _calculate_ta_score with real technical data and calculation."""

        result = self.data_collection_round._calculate_ta_score(self.technical_data_all_conditions_met, 105.0)  # noqa: SLF001
        assert result == 1.0
