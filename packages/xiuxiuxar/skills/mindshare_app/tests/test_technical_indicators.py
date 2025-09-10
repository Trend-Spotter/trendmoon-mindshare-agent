"""Test the technical indicators functions in DataCollectionRound."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from aea.test_tools.test_skill import BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis import AnalysisRound


class TestTechnicalIndicators(BaseSkillTestCase):  # noqa: PLR0904
    """Test technical indicators functions in AnalysisRound."""

    path_to_skill = Path(__file__).parent.parent

    def setup(self):
        """Setup the test class."""
        super().setup()

        self.analysis_round = AnalysisRound(
            name="analysis_round",
            skill_context=self.skill.skill_context,
        )
        self.logger = self.skill.skill_context.logger

        # Sample valid OHLCV data for testing
        self.valid_ohlcv_data = [
            [1672531200000, 16500.0, 16700.0, 16400.0, 16600.0, 1000.0],
            [1672617600000, 16600.0, 16800.0, 16500.0, 16750.0, 1100.0],
            [1672704000000, 16750.0, 16900.0, 16600.0, 16850.0, 1200.0],
            [1672790400000, 16850.0, 17000.0, 16700.0, 16900.0, 1300.0],
            [1672876800000, 16900.0, 17100.0, 16800.0, 17000.0, 1400.0],
        ] * 10  # Repeat to have enough data for indicators

        self.ma_length = 20

    def test_validate_ohlcv_data_valid(self):
        """Test _validate_ohlcv_data with valid data."""
        # Should not raise any exception
        self.analysis_round._validate_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

    def test_validate_ohlcv_data_empty_list(self):
        """Test _validate_ohlcv_data with empty list."""
        with pytest.raises(ValueError, match="ohlcv_data must be a non-empty list of lists"):
            self.analysis_round._validate_ohlcv_data([])  # noqa: SLF001

    def test_validate_ohlcv_data_not_list(self):
        """Test _validate_ohlcv_data with non-list input."""
        with pytest.raises(ValueError, match="ohlcv_data must be a non-empty list of lists"):
            self.analysis_round._validate_ohlcv_data(None)  # noqa: SLF001

        with pytest.raises(ValueError, match="ohlcv_data must be a non-empty list of lists"):
            self.analysis_round._validate_ohlcv_data("invalid")  # noqa: SLF001

    def test_validate_ohlcv_data_invalid_row_format(self):
        """Test _validate_ohlcv_data with invalid row format."""
        invalid_data = [
            [1672531200000, 16500.0, 16700.0, 16400.0, 16600.0],  # Missing volume
            [1672617600000, 16600.0, 16800.0, 16500.0, 16750.0, 1100.0],
        ]
        with pytest.raises(ValueError, match="Each row must be a list with at least 6 elements"):
            self.analysis_round._validate_ohlcv_data(invalid_data)  # noqa: SLF001

    def test_validate_ohlcv_data_non_list_row(self):
        """Test _validate_ohlcv_data with non-list row."""
        invalid_data = [
            "not_a_list",
            [1672617600000, 16600.0, 16800.0, 16500.0, 16750.0, 1100.0],
        ]
        with pytest.raises(ValueError, match="Each row must be a list with at least 6 elements"):
            self.analysis_round._validate_ohlcv_data(invalid_data)  # noqa: SLF001

    def test_preprocess_ohlcv_data_success(self):
        """Test _preprocess_ohlcv_data with valid data."""
        result = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.valid_ohlcv_data)
        assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert result.index.name == "date"

    def test_preprocess_ohlcv_data_with_invalid_numbers(self):
        """Test _preprocess_ohlcv_data with some invalid numbers."""
        data_with_nans = [
            [1672531200000, 16500.0, 16700.0, 16400.0, 16600.0, 1000.0],
            [1672617600000, "invalid", 16800.0, 16500.0, 16750.0, 1100.0],
            [1672704000000, 16750.0, 16900.0, 16600.0, 16850.0, 1200.0],
        ]
        result = self.analysis_round._preprocess_ohlcv_data(data_with_nans)  # noqa: SLF001

        # Should drop rows with invalid data
        assert len(result) == 2  # Only valid rows remain

    def test_calculate_trend_indicators(self):
        """Test _calculate_trend_indicators."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with (
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.sma") as mock_sma,
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.ema") as mock_ema,
        ):
            mock_sma.return_value = pd.Series([100.0] * len(data), index=data.index)
            mock_ema.return_value = pd.Series([101.0] * len(data), index=data.index)

            self.analysis_round._calculate_trend_indicators(data, self.ma_length)  # noqa: SLF001

            mock_sma.assert_called_once()
            mock_ema.assert_called_once()
            assert "SMA" in data.columns
            assert "EMA" in data.columns

    def test_calculate_momentum_indicators(self):
        """Test _calculate_momentum_indicators."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with (
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.rsi") as mock_rsi,
            patch.object(self.analysis_round, "_process_macd_indicator") as mock_macd,
            patch.object(self.analysis_round, "_process_adx_indicator") as mock_adx,
        ):
            # Create a simple Series that works with .clip() - use same index as the close column
            mock_rsi.return_value = pd.Series([50.0] * len(data), index=data.index)

            self.analysis_round._calculate_momentum_indicators(data)  # noqa: SLF001

            mock_rsi.assert_called_once()
            mock_macd.assert_called_once()
            mock_adx.assert_called_once()
            assert "RSI" in data.columns

    def test_process_macd_indicator_success(self):
        """Test _process_macd_indicator with successful calculation."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Mock MACD DataFrame with expected column names
        mock_macd_df = pd.DataFrame(
            {"MACD_12_26_9": [1.0] * len(data), "MACDh_12_26_9": [0.5] * len(data), "MACDs_12_26_9": [0.8] * len(data)},
            index=data.index,
        )

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.macd", return_value=mock_macd_df):
            self.analysis_round._process_macd_indicator(data)  # noqa: SLF001

            assert "MACD" in data.columns
            assert "MACDh" in data.columns
            assert "MACDs" in data.columns

    def test_process_macd_indicator_none_result(self):
        """Test _process_macd_indicator when pandas_ta returns None."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.macd", return_value=None):
            self.analysis_round._process_macd_indicator(data)  # noqa: SLF001

            # Should not add any columns
            assert "MACD" not in data.columns
            assert "MACDh" not in data.columns
            assert "MACDs" not in data.columns

    def test_process_macd_indicator_empty_result(self):
        """Test _process_macd_indicator when pandas_ta returns empty DataFrame."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        empty_df = pd.DataFrame()
        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.macd", return_value=empty_df):
            self.analysis_round._process_macd_indicator(data)  # noqa: SLF001

            # Should not add any columns
            assert "MACD" not in data.columns

    def test_process_adx_indicator_success(self):
        """Test _process_adx_indicator with successful calculation."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Mock ADX DataFrame
        mock_adx_df = pd.DataFrame(
            {"ADX_14": [25.0] * len(data), "DMP_14": [20.0] * len(data), "DMN_14": [15.0] * len(data)}, index=data.index
        )

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.adx", return_value=mock_adx_df):
            self.analysis_round._process_adx_indicator(data)  # noqa: SLF001

            assert "ADX" in data.columns

    def test_process_adx_indicator_none_result(self):
        """Test _process_adx_indicator when pandas_ta returns None."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.adx", return_value=None):
            self.analysis_round._process_adx_indicator(data)  # noqa: SLF001

            assert "ADX" not in data.columns

    def test_calculate_volatility_indicators_success(self):
        """Test _calculate_volatility_indicators with successful calculation."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Mock Bollinger Bands DataFrame
        mock_bb_df = pd.DataFrame(
            {
                "BBL_20_2.0": [16000.0] * len(data),
                "BBM_20_2.0": [16500.0] * len(data),
                "BBU_20_2.0": [17000.0] * len(data),
            },
            index=data.index,
        )

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.bbands", return_value=mock_bb_df):
            self.analysis_round._calculate_volatility_indicators(data)  # noqa: SLF001

            assert "BBL" in data.columns
            assert "BBM" in data.columns
            assert "BBU" in data.columns

    def test_calculate_volatility_indicators_none_result(self):
        """Test _calculate_volatility_indicators when pandas_ta returns None."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.bbands", return_value=None):
            self.analysis_round._calculate_volatility_indicators(data)  # noqa: SLF001

            assert "BBL" not in data.columns
            assert "BBM" not in data.columns
            assert "BBU" not in data.columns

    def test_calculate_volume_indicators_success(self):
        """Test _calculate_volume_indicators with successful calculations."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        mock_obv = pd.Series([1000.0] * len(data), index=data.index)
        mock_cmf = pd.Series([0.1] * len(data), index=data.index)

        with (
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.obv", return_value=mock_obv),
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.cmf", return_value=mock_cmf),
        ):
            self.analysis_round._calculate_volume_indicators(data)  # noqa: SLF001

            assert "OBV" in data.columns
            assert "CMF" in data.columns

    def test_calculate_volume_indicators_none_results(self):
        """Test _calculate_volume_indicators when pandas_ta returns None."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        with (
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.obv", return_value=None),
            patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.cmf", return_value=None),
        ):
            self.analysis_round._calculate_volume_indicators(data)  # noqa: SLF001

            assert "OBV" not in data.columns
            assert "CMF" not in data.columns

    def test_build_technical_result_all_indicators(self):
        """Test _build_technical_result with all indicators present."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Add all indicators to the DataFrame
        data["SMA"] = 16500.0
        data["EMA"] = 16520.0
        data["RSI"] = 55.0
        data["MACD"] = 10.0
        data["MACDh"] = 5.0
        data["MACDs"] = 8.0
        data["ADX"] = 25.0
        data["BBL"] = 16000.0
        data["BBM"] = 16500.0
        data["BBU"] = 17000.0
        data["OBV"] = 10000.0
        data["CMF"] = 0.15

        result = self.analysis_round._build_technical_result(data)  # noqa: SLF001

        assert len(result) == 8  # All indicators should be present

        # Check individual indicators
        indicator_names = [item[0] for item in result]
        assert "SMA" in indicator_names
        assert "EMA" in indicator_names
        assert "RSI" in indicator_names
        assert "MACD" in indicator_names
        assert "ADX" in indicator_names
        assert "BB" in indicator_names
        assert "OBV" in indicator_names
        assert "CMF" in indicator_names

    def test_build_technical_result_partial_indicators(self):
        """Test _build_technical_result with only some indicators present."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Add only some indicators
        data["SMA"] = 16500.0
        data["RSI"] = 55.0
        data["OBV"] = 10000.0

        result = self.analysis_round._build_technical_result(data)  # noqa: SLF001

        assert len(result) == 3
        indicator_names = [item[0] for item in result]
        assert "SMA" in indicator_names
        assert "RSI" in indicator_names
        assert "OBV" in indicator_names
        assert "MACD" not in indicator_names  # Should not be present

    def test_build_technical_result_macd_incomplete(self):
        """Test _build_technical_result with incomplete MACD data."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Add only partial MACD components
        data["MACD"] = 10.0
        data["MACDh"] = 5.0
        # Missing MACDs

        result = self.analysis_round._build_technical_result(data)  # noqa: SLF001

        indicator_names = [item[0] for item in result]
        assert "MACD" not in indicator_names  # Should not be present due to incomplete data

    def test_build_technical_result_bb_incomplete(self):
        """Test _build_technical_result with incomplete Bollinger Bands data."""
        data = self.analysis_round._preprocess_ohlcv_data(self.valid_ohlcv_data)  # noqa: SLF001

        # Add only partial BB components
        data["BBL"] = 16000.0
        data["BBM"] = 16500.0
        # Missing BBU

        result = self.analysis_round._build_technical_result(data)  # noqa: SLF001

        indicator_names = [item[0] for item in result]
        assert "BB" not in indicator_names  # Should not be present due to incomplete data

    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.sma")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.ema")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.rsi")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.macd")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.adx")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.bbands")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.obv")
    @patch("packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis.ta.cmf")
    def test_get_technical_data_success(
        self, mock_cmf, mock_obv, mock_bbands, mock_adx, mock_macd, mock_rsi, mock_ema, mock_sma
    ):
        """Test _get_technical_data with successful calculation."""
        # Setup mocks
        data_length = len(self.valid_ohlcv_data)
        mock_sma.return_value = pd.Series([16500.0] * data_length)
        mock_ema.return_value = pd.Series([16520.0] * data_length)
        mock_rsi.return_value = pd.Series([55.0] * data_length)

        mock_macd_df = pd.DataFrame(
            {
                "MACD_12_26_9": [10.0] * data_length,
                "MACDh_12_26_9": [5.0] * data_length,
                "MACDs_12_26_9": [8.0] * data_length,
            }
        )
        mock_macd.return_value = mock_macd_df

        mock_adx_df = pd.DataFrame({"ADX_14": [25.0] * data_length})
        mock_adx.return_value = mock_adx_df

        mock_bb_df = pd.DataFrame(
            {
                "BBL_20_2.0": [16000.0] * data_length,
                "BBM_20_2.0": [16500.0] * data_length,
                "BBU_20_2.0": [17000.0] * data_length,
            }
        )
        mock_bbands.return_value = mock_bb_df

        mock_obv.return_value = pd.Series([10000.0] * data_length)
        mock_cmf.return_value = pd.Series([0.15] * data_length)

        result = self.analysis_round._get_technical_data(self.valid_ohlcv_data)  # noqa: SLF001

        assert isinstance(result, list)
        assert len(result) > 0

        # Check that we have expected indicators
        indicator_names = [item[0] for item in result]
        assert "SMA" in indicator_names
        assert "EMA" in indicator_names
        assert "RSI" in indicator_names

    def test_get_technical_data_invalid_input(self):
        """Test _get_technical_data with invalid input."""
        result = self.analysis_round._get_technical_data([])  # noqa: SLF001
        assert result == []

    def test_get_technical_data_exception_handling(self):
        """Test _get_technical_data exception handling."""
        with patch.object(self.analysis_round, "_validate_ohlcv_data", side_effect=ValueError("Test error")):
            result = self.analysis_round._get_technical_data(self.valid_ohlcv_data)  # noqa: SLF001
            assert result == []

    def test_get_technical_data_pandas_exception(self):
        """Test _get_technical_data with pandas processing exception."""
        # Create data that will cause pandas issues
        invalid_data = [["invalid", "data", "format", "test", "error", "handling"]]

        result = self.analysis_round._get_technical_data(invalid_data)  # noqa: SLF001
        assert result == []
