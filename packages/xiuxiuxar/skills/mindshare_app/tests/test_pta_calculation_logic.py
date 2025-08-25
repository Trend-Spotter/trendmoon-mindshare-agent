"""Test the technical indicators functions in DataCollectionRound."""

from pathlib import Path
from unittest.mock import patch

from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.behaviours import DataCollectionRound


class TestTechnicalIndicators(BaseSkillTestCase):
    """Test technical indicators functions in DataCollectionRound."""

    path_to_skill = Path(__file__).parent.parent

    def setup_method(self):
        """Setup the test class."""
        super().setup()

        self.data_collection_round = DataCollectionRound(
            name="data_collection_round",
            skill_context=self.skill.skill_context,
        )
        self.logger = self.skill.skill_context.logger

        # Mock technical data that matches the expected format from _build_technical_result
        self.technical_data_all_conditions_met = {
            "SMA": 100.0,  # Current price (105.0) > SMA
            "EMA": 101.0,  # Current price (105.0) > EMA
            "RSI": 50.0,  # RSI in range (30 < 50 < 70)
            "MACD": {  # MACD structure as dict
                "MACD": 10.0,  # MACD line
                "MACDs": 8.0,  # Signal line (MACD > Signal)
                "MACDh": 2.0,  # Histogram
            },
            "ADX": 30.0,  # ADX > threshold (25)
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

    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound.context")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound._check_rsi_condition")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound._check_macd_condition")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound._check_adx_condition")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound._check_moving_average_conditions")
    def test_calculate_ta_score_all_conditions_met(
        self,
        mock_context,
        mock_check_rsi_condition,
        mock_check_macd_condition,
        mock_check_adx_condition,
        mock_check_moving_average_conditions,
    ):
        """Test _calculate_ta_score when all conditions are met (entry_long = 1.0)."""

        mock_context.params.rsi_lower_limit = 30.0
        mock_context.params.rsi_upper_limit = 70.0
        mock_context.params.adx_threshold = 25.0

        mock_check_rsi_condition.return_value = 1.0
        mock_check_macd_condition.return_value = 1.0
        mock_check_adx_condition.return_value = 1.0
        mock_check_moving_average_conditions.return_value = [1.0, 1.0]

        result = self.data_collection_round._calculate_ta_score(self.technical_data_all_conditions_met, 105.0)  # noqa: SLF001
        assert result == 1.0

    def test_check_moving_average_conditions_sma(self):
        """Test _check_moving_average_conditions with SMA."""

        self.technical_data_all_conditions_met["SMA"] = 105.0

        result = self.data_collection_round._check_moving_average_conditions(  # noqa: SLF001
            self.technical_data_all_conditions_met, 105.0
        )
        assert result[0] == 0.0

        self.technical_data_all_conditions_met["SMA"] = 104.9

        result = self.data_collection_round._check_moving_average_conditions(  # noqa: SLF001
            self.technical_data_all_conditions_met, 105.0
        )
        assert result[0] == 1.0

    def test_check_moving_average_conditions_ema(self):
        """Test _check_moving_average_conditions with EMA."""

        self.technical_data_all_conditions_met["EMA"] = 105.0

        result = self.data_collection_round._check_moving_average_conditions(  # noqa: SLF001
            self.technical_data_all_conditions_met, 105.0
        )
        assert result[1] == 0.0

        self.technical_data_all_conditions_met["EMA"] = 104.9

        result = self.data_collection_round._check_moving_average_conditions(  # noqa: SLF001
            self.technical_data_all_conditions_met, 105.0
        )
        assert result[1] == 1.0

    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound.context")
    def test_check_rsi_condition_rsi_strict_inequality(self, mock_context):
        """Test _check_rsi_condition."""
        # Mock context parameters
        mock_context.params.rsi_lower_limit = 50.0
        mock_context.params.rsi_upper_limit = 70.0

        result = self.data_collection_round._check_rsi_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 0.0

        mock_context.params.rsi_lower_limit = 30.0
        mock_context.params.rsi_upper_limit = 50.0

        result = self.data_collection_round._check_rsi_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 0.0

    def test_check_macd_condition_macd_crossover(self):
        """Test _check_macd_condition."""

        # Ensure MACD crossover is not met. Should return 0
        self.technical_data_all_conditions_met["MACD"]["MACD"] = 11.0
        self.technical_data_all_conditions_met["MACD"]["MACDs"] = 11.0

        result = self.data_collection_round._check_macd_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 0.0

        # MACD becomes greater than MACD signal. Should return 1
        self.technical_data_all_conditions_met["MACD"]["MACD"] = 11.1
        self.technical_data_all_conditions_met["MACD"]["MACDs"] = 11.0

        result = self.data_collection_round._check_macd_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 1.0

    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.DataCollectionRound.context")
    def test_check_adx_condition_adx_threshold(self, mock_context):
        """Test _check_adx_condition."""
        # Mock context parameters
        mock_context.params.adx_threshold = 25.0

        self.technical_data_all_conditions_met["ADX"] = 25

        result = self.data_collection_round._check_adx_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 0.0

        self.technical_data_all_conditions_met["ADX"] = 25.1

        result = self.data_collection_round._check_adx_condition(self.technical_data_all_conditions_met)  # noqa: SLF001
        assert result == 1.0
