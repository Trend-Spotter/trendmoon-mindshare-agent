# ------------------------------------------------------------------------------
#
#   Copyright 2025 xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the implementation of the behaviours of Mindshare App skill."""

import json
import operator
from typing import Any
from pathlib import Path
from datetime import UTC, datetime

import pandas as pd
import pandas_ta_classic as ta
from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    ALLOWED_ASSETS,
    BaseState,
    TradingStrategy,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class AnalysisRound(BaseState):
    """This class implements the behaviour of the state AnalysisRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.ANALYSISROUND
        self.analysis_initialized: bool = False
        self.analysis_results: dict[str, Any] = {}
        self.pending_tokens: list[dict[str, str]] = []
        self.completed_analysis: list[dict[str, str]] = []
        self.failed_analysis: list[dict[str, str]] = []
        self.collected_data: dict[str, Any] = {}

    def setup(self) -> None:
        """Setup the state."""
        self._is_done = False
        self.analysis_initialized = False
        self.analysis_results = {}
        self.pending_tokens = []
        self.completed_analysis = []
        self.failed_analysis = []
        self.collected_data = {}

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")

        try:
            if not self.analysis_initialized:
                self._initialize_analysis()

            if self.pending_tokens:
                token_info = self.pending_tokens.pop(0)
                self._analyze_single_token(token_info)
                return

            if not self.pending_tokens:
                self._finalize_analysis()
                self._is_done = True
                self._event = MindshareabciappEvents.DONE

        except Exception as e:
            self.context.logger.exception(f"Analysis failed: {e}")
            self.context.error_context = {
                "error_type": "analysis_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _initialize_analysis(self) -> None:
        """Initialize the analysis."""
        if self.analysis_initialized:
            return

        self.context.logger.info("Initializing analysis...")

        if not self._load_collected_data():
            self.context.logger.error("Failed to load collected data")
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True
            return

        # Initialize consolidated trading strategy
        strategy_type = getattr(self.context.params, "trading_strategy", "balanced")
        self.trading_strategy = TradingStrategy(strategy_type, self.context)

        self.pending_tokens = ALLOWED_ASSETS["base"].copy()
        self.completed_analysis = []
        self.failed_analysis = []

        self.analysis_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_tokens": len(self.pending_tokens),
            "token_analysis": {},
            "social_scores": {},
            "technical_scores": {},
            "combined_scores": {},
            "analysis_summary": {},
        }

        self.analysis_initialized = True
        self.context.logger.info(f"Initialized analysis with {len(self.pending_tokens)} tokens")

    def _load_collected_data(self) -> bool:
        """Load the collected data."""
        try:
            if not self.context.store_path:
                self.context.logger.warning("No store path available, skipping data loading.")
                return False

            data_file = self.context.store_path / "collected_data.json"
            if not data_file.exists():
                self.context.logger.warning("No collected data file found")
                return False

            with open(data_file, encoding=DEFAULT_ENCODING) as f:
                self.collected_data = json.load(f)

            self.context.logger.info("Successfully loaded collected data")
            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to load collected data: {e}")
            return False

    def _analyze_single_token(self, token_info: dict[str, str]) -> None:
        """Analyze a single token."""
        symbol = token_info["symbol"]

        try:
            self.context.logger.info(f"Analyzing {symbol}...")

            social_scores = self._perform_social_analysis(token_info)

            technical_scores = self._perform_technical_analysis(token_info)

            combined_analysis = self._combine_analysis_results(token_info, social_scores, technical_scores)

            self.analysis_results["token_analysis"][symbol] = combined_analysis
            self.analysis_results["social_scores"][symbol] = social_scores
            self.analysis_results["technical_scores"][symbol] = technical_scores

            self.completed_analysis.append(token_info)
            self.context.logger.info(f"Completed analysis for {symbol}")

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.context.logger.warning(f"Failed to analyze {symbol}: {e}")
            self.failed_analysis.append(token_info)
            self.analysis_results["token_analysis"][symbol] = {
                "error": str(e),
                "p_social": 0.0,
                "p_technical": 0.0,
                "p_combined": 0.0,
            }

    def _perform_social_analysis(self, token_info: dict[str, str]) -> dict[str, Any]:
        symbol = token_info["symbol"]

        social_scores = {
            "social_mentions": 0.0,
            "social_dominance": 0.0,
            "correlation_1d": 0.0,
            "correlation_3d": 0.0,
            "correlation_7d": 0.0,
            "correlation_strength": 0.0,
            "social_trend": "neutral",
            "p_social": 0.5,
            "social_metrics_available": False,
        }

        try:
            social_data = self.collected_data.get("social_data", {}).get(symbol)
            if not social_data:
                self.context.logger.warning(f"No social data found for {symbol}")
                return social_scores

            social_scores["social_metrics_available"] = True

            mentions = social_data.get("social_mentions", 0)
            dominance = social_data.get("social_dominance", 0)

            social_scores["social_mentions"] = mentions
            social_scores["social_dominance"] = dominance

            p_social = self._calculate_social_probability_score(mentions, dominance)
            social_scores["p_social"] = p_social

            if p_social > 0.6:
                social_scores["social_trend"] = "bullish"
            elif p_social < 0.4:
                social_scores["social_trend"] = "bearish"
            else:
                social_scores["social_trend"] = "neutral"

            self.context.logger.info(f"Social analysis for {symbol}: p_social={p_social:.3f}")

        except Exception as e:
            self.context.logger.exception(f"Failed to perform social analysis for {symbol}: {e}")

        return social_scores

    def _perform_technical_analysis(self, token_info: dict[str, str]) -> dict[str, Any]:
        """Perform technical analysis for a single token."""
        symbol = token_info["symbol"]
        technical_scores = {
            "sma": None,
            "ema_20": None,
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "adx": None,
            "obv": None,
            "p_technical": 0.5,
            "technical_traits": {},
            "price_above_ma": False,
            "rsi_in_range": False,
            "macd_bullish": False,
            "adx_strong_trend": False,
            "obv_increasing": False,
        }

        try:
            ohlcv_data = self.collected_data.get("ohlcv", {}).get(symbol)
            if not ohlcv_data:
                self.context.logger.warning(f"No OHLCV data found for {symbol}")
                return technical_scores

            current_prices = self.collected_data.get("current_prices", {}).get(symbol)
            current_price = current_prices.get("usd", 0) if current_prices else 0

            technical_indicators = self._get_technical_data(ohlcv_data)

            indicators_dict = {}
            for indicator_name, value in technical_indicators:
                indicators_dict[indicator_name] = value

            technical_scores.update(
                {
                    "sma": indicators_dict.get("SMA"),
                    "ema_20": indicators_dict.get("EMA"),
                    "rsi": indicators_dict.get("RSI"),
                    "adx": indicators_dict.get("ADX"),
                    "obv": indicators_dict.get("OBV"),
                }
            )

            # Extract MACD Components
            macd_data = indicators_dict.get("MACD", {})
            if isinstance(macd_data, dict):
                technical_scores["macd"] = macd_data.get("MACD")
                technical_scores["macd_signal"] = macd_data.get("MACDs")

            p_technical, conditions_met = self._calculate_technical_probability_score(
                current_price=current_price, indicators=indicators_dict
            )

            technical_scores["p_technical"] = p_technical
            technical_scores["conditions_met"] = conditions_met

            # Update with condition results (excluding social_bullish)
            for condition_name, condition_data in conditions_met.items():
                if condition_name != "social_bullish":
                    # Map to old naming convention for compatibility
                    if condition_name == "price_above_ma":
                        technical_scores["price_above_ma"] = condition_data["condition"]
                    elif condition_name == "adx_strong":
                        technical_scores["adx_strong_trend"] = condition_data["condition"]
                    elif condition_name == "macd_bullish":
                        technical_scores["macd_bullish"] = condition_data["condition"]
                    else:
                        technical_scores[condition_name] = condition_data["condition"]

            self.context.logger.info(f"Technical analysis for {symbol}: p_technical={p_technical:.3f}")

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.context.logger.warning(f"Technical analysis failed for {symbol}: {e}")

        return technical_scores

    def _calculate_social_probability_score(self, mentions: int, dominance: int) -> float:
        """Calculate the social probability score."""
        try:
            mentions_score = min(mentions / 100.0, 1.0) if mentions > 0 else 0.0
            dominance_score = min(dominance / 10.0, 1.0) if dominance > 0 else 0.0

            social_weight_mentions = 0.4
            social_weight_dominance = 0.6

            p_social = mentions_score * social_weight_mentions + dominance_score * social_weight_dominance

            return max(0.0, min(1.0, p_social))  # Clamp to [0, 1]

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.context.logger.warning(f"Failed to calculate social score: {e}")
            return 0.5

    def _calculate_technical_probability_score(
        self, current_price: float, indicators: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Calculate the technical probability score using consolidated trading strategy."""
        # Create technical_scores dict for the consolidated strategy
        technical_scores = {
            "sma": indicators.get("SMA", current_price),
            "rsi": indicators.get("RSI", 50.0),
            "adx": indicators.get("ADX", 20.0),
            "obv": indicators.get("OBV", 0),
        }

        # Extract MACD data
        macd_data = indicators.get("MACD", {})
        if isinstance(macd_data, dict):
            technical_scores["macd"] = macd_data.get("MACD", 0)
            technical_scores["macd_signal"] = macd_data.get("MACDs", 0)
        else:
            technical_scores["macd"] = technical_scores["macd_signal"] = 0

        # Use consolidated trading strategy for evaluation
        # Create minimal social_scores for compatibility
        social_scores = {"p_social": 0.5}

        conditions_met = self.trading_strategy.evaluate_conditions(current_price, social_scores, technical_scores)

        # Calculate technical probability as binary (1 or 0) based on meeting all four conditions
        technical_conditions = self.trading_strategy.params["core_conditions"]
        all_conditions_met = all(
            conditions_met.get(condition, {}).get("condition", False) for condition in technical_conditions
        )
        p_technical = 1.0 if all_conditions_met else 0.0

        return p_technical, conditions_met

    def _combine_analysis_results(
        self, token_info: dict[str, str], social_scores: dict[str, Any], technical_scores: dict[str, Any]
    ) -> dict[str, Any]:
        """Combine the analysis results."""
        symbol = token_info["symbol"]

        try:
            # Extract probability scores
            p_social = social_scores.get("p_social", 0.5)
            p_technical = technical_scores.get("p_technical", 0.5)

            # Use consolidated trading strategy for weights and combination
            p_combined = self.trading_strategy.calculate_combined_probability(p_social, p_technical)

            return {
                "symbol": symbol,
                "timestamp": datetime.now(UTC).isoformat(),
                "p_social": round(p_social, 3),
                "p_technical": round(p_technical, 3),
                "p_combined": round(p_combined, 3),
                "strategy_type": self.trading_strategy.strategy_type,
                "weights_used": {
                    "social_weight": self.trading_strategy.params["social_weight"],
                    "technical_weight": self.trading_strategy.params["technical_weight"],
                },
            }

        except Exception as e:
            self.context.logger.exception(f"Failed to combine analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "p_social": 0.5,
                "p_technical": 0.5,
                "p_combined": 0.5,
            }

    def _finalize_analysis(self) -> None:
        """Finalize analysis and prepare results for next round."""
        try:
            completed_count = len(self.completed_analysis)
            failed_count = len(self.failed_analysis)
            total_count = completed_count + failed_count

            # Create analysis summary
            self.analysis_results["analysis_summary"] = {
                "total_tokens_analyzed": total_count,
                "successful_analysis": completed_count,
                "failed_analysis": failed_count,
                "success_rate": completed_count / total_count if total_count > 0 else 0,
                "completed_at": datetime.now(UTC).isoformat(),
            }

            # Find tokens with strong signals
            strong_signals = []
            for symbol, analysis in self.analysis_results["token_analysis"].items():
                if isinstance(analysis, dict) and analysis.get("p_combined", 0) > 0.6:
                    strong_signals.append(
                        {
                            "symbol": symbol,
                            "p_combined": analysis.get("p_combined"),
                            "signal_strength": analysis.get("signal_strength"),
                            "recommendation": analysis.get("trading_recommendation"),
                        }
                    )

            # Sort by combined score
            strong_signals.sort(key=operator.itemgetter("p_combined"), reverse=True)
            self.analysis_results["strong_signals"] = strong_signals

            # Store results for SignalAggregationRound
            self._store_analysis_results()

            # Set context for next round
            self.context.analysis_results = self.analysis_results

            self.context.logger.info(
                f"Analysis completed: {completed_count}/{total_count} successful, "
                f"{len(strong_signals)} strong signals identified"
            )

            if strong_signals:
                self.context.logger.info("Strong signals found:")
                for signal in strong_signals[:3]:  # Show top 3
                    self.context.logger.info(
                        f"  {signal['symbol']}: {signal['p_combined']:.3f} ({signal['signal_strength']})"
                    )

        except Exception as e:
            self.context.logger.exception(f"Failed to finalize analysis: {e}")

    def _store_analysis_results(self) -> None:
        """Store analysis results to persistent storage."""
        if not self.context.store_path:
            return

        try:
            summary_file = self.context.store_path / "analysis_results.json"
            summary_results = self._extract_summary_results()
            with open(summary_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(summary_results, f, indent=2)

            if self._has_detailed_technical_data():
                technical_file = self.context.store_path / "technical_data.parquet"
                self._store_technical_data_parquet(technical_file)

            self.context.logger.info(f"Analysis results stored to {summary_file}")

        except Exception as e:
            self.context.logger.exception(f"Failed to store analysis results: {e}")

    def _extract_summary_results(self) -> dict[str, Any]:
        """Extract JSON-serializable summary results."""
        return {
            "timestamp": self.analysis_results.get("timestamp"),
            "total_tokens": self.analysis_results.get("total_tokens"),
            "token_analysis": self._make_json_serializable(self.analysis_results.get("token_analysis", {})),
            "social_scores": self.analysis_results.get("social_scores", {}),
            "technical_scores": self.analysis_results.get("technical_scores", {}),
            "combined_scores": self.analysis_results.get("combined_scores", {}),
            "analysis_summary": self.analysis_results.get("analysis_summary", {}),
            "strong_signals": self.analysis_results.get("strong_signals", []),
        }

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to JSON-serializable Python types."""
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        return obj

    def _validate_ohlcv_data(self, ohlcv_data: list[list[Any]]) -> None:
        """Validate OHLCV data format."""
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) == 0:
            msg = "ohlcv_data must be a non-empty list of lists."
            raise ValueError(msg)

        for row in ohlcv_data:
            if not (isinstance(row, list) and len(row) >= 6):
                msg = "Each row must be a list with at least 6 elements: timestamp, open, high, low, close, volume."
                raise ValueError(msg)

    def _has_detailed_technical_data(self) -> bool:
        """Check if there is detailed technical data available."""
        return any(
            isinstance(data, dict) and "technical_indicators" in data
            for data in self.analysis_results.get("token_analysis", {}).values()
        )

    def _store_technical_data_parquet(self, filepath: Path) -> None:
        """Store technical data to Parquet file."""
        try:
            technical_data = []

            for symbol, analysis in self.analysis_results.get("token_analysis", {}).items():
                if "technical_indicators" in analysis:
                    # Convert technical indicators to DataFrame
                    indicators = analysis["technical_indicators"]
                    if isinstance(indicators, list):
                        # Convert list of tuples to DataFrame
                        indicators_df = pd.DataFrame(indicators, columns=["indicator", "value"])
                        indicators_df["symbol"] = symbol
                        indicators_df["timestamp"] = analysis.get("timestamp")
                        technical_data.append(indicators_df)

            if technical_data:
                combined_df = pd.concat(technical_data, ignore_index=True)
                combined_df.to_parquet(filepath, index=False)
                self.context.logger.info(f"Technical data stored to {filepath}")

        except (ValueError, TypeError) as e:
            self.context.logger.warning(f"Failed to store technical data: {e}")

    def _get_technical_data(self, ohlcv_data: list[list[Any]]) -> list:
        """Calculate core technical indicators for a coin using pandas-ta with validation."""
        try:
            self._validate_ohlcv_data(ohlcv_data)
            data = self._preprocess_ohlcv_data(ohlcv_data)

            # Calculate different groups of indicators
            self._calculate_trend_indicators(data, self.context.params.moving_average_length)
            self._calculate_momentum_indicators(data)
            self._calculate_volatility_indicators(data)
            self._calculate_volume_indicators(data)

            # Build and return the result
            return self._build_technical_result(data)

        except (ValueError, KeyError, ConnectionError, TimeoutError) as e:
            self.context.logger.warning(f"Technical data error: {e}")
            return []

    def _preprocess_ohlcv_data(self, ohlcv_data: list[list[Any]]) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame and clean it."""
        data = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["date"] = pd.to_datetime(data["timestamp"], unit="ms")
        data = data.set_index("date")

        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        return data.dropna()

    def _calculate_trend_indicators(self, data: pd.DataFrame, moving_average_length: int) -> None:
        """Calculate trend indicators (SMA, EMA)."""
        data["SMA"] = ta.sma(data["close"], length=moving_average_length)
        data["EMA"] = ta.ema(data["close"], length=moving_average_length)

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> None:
        """Calculate momentum indicators (RSI, MACD, ADX)."""
        # RSI (normalized 0-100)
        data["RSI"] = ta.rsi(data["close"], length=self.context.params.rsi_period_length)
        data["RSI"] = data["RSI"].clip(0, 100)

        # MACD
        self._process_macd_indicator(data)

        # ADX
        self._process_adx_indicator(data)

    def _process_macd_indicator(self, data: pd.DataFrame) -> None:
        """Process MACD indicator and handle different column naming conventions."""
        macd = ta.macd(
            data["close"],
            fast=self.context.params.macd_fast_period,
            slow=self.context.params.macd_slow_period,
            signal=self.context.params.macd_signal_period,
        )
        if macd is None or macd.empty:
            return

        macd_cols = macd.columns.tolist()
        macd_line = macd_histogram = macd_signal_line = None

        # Find the correct column names
        for col in macd_cols:
            if "MACD_" in col and "h_" not in col and "s_" not in col:
                macd_line = col
            elif "MACDh_" in col:
                macd_histogram = col
            elif "MACDs_" in col:
                macd_signal_line = col

        if macd_line:
            data["MACD"] = macd[macd_line]
        if macd_histogram:
            data["MACDh"] = macd[macd_histogram]
        if macd_signal_line:
            data["MACDs"] = macd[macd_signal_line]

    def _process_adx_indicator(self, data: pd.DataFrame) -> None:
        """Process ADX indicator and handle different column naming conventions."""
        adx = ta.adx(data["high"], data["low"], data["close"], length=self.context.params.adx_period_length)
        if adx is None or adx.empty:
            return

        # Find the correct ADX column name
        for col in adx.columns.tolist():
            if col.startswith("ADX"):
                data["ADX"] = adx[col]
                break

    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> None:
        """Calculate volatility indicators (Bollinger Bands)."""
        bbands = ta.bbands(data["close"], length=20, std=2)
        if bbands is None or bbands.empty:
            return

        bb_cols = bbands.columns.tolist()
        bb_lower = bb_middle = bb_upper = None

        # Find the correct column names
        for col in bb_cols:
            if col.startswith("BBL"):
                bb_lower = col
            elif col.startswith("BBM"):
                bb_middle = col
            elif col.startswith("BBU"):
                bb_upper = col

        if bb_lower and bb_middle and bb_upper:
            data["BBL"] = bbands[bb_lower]
            data["BBM"] = bbands[bb_middle]
            data["BBU"] = bbands[bb_upper]

    def _calculate_volume_indicators(self, data: pd.DataFrame) -> None:
        """Calculate volume indicators (OBV, CMF)."""
        # On-Balance Volume (OBV)
        obv = ta.obv(data["close"], data["volume"])
        if obv is not None and not obv.empty:
            data["OBV"] = obv

        # Chaikin Money Flow (CMF)
        cmf = ta.cmf(data["high"], data["low"], data["close"], data["volume"], length=20)
        if cmf is not None and not cmf.empty:
            data["CMF"] = cmf

    def _extract_value(self, latest_data: pd.Series, column: str):
        """Extract and normalize value from pandas Series."""
        value = latest_data[column]
        if hasattr(value, "item"):
            value = value.item()
        return value

    def _add_basic_indicators(self, latest_data: pd.Series, data: pd.DataFrame, result: list):
        """Add basic single-value indicators to result."""
        for indicator in ["SMA", "EMA", "RSI", "ADX"]:
            if indicator in data.columns:
                value = self._extract_value(latest_data, indicator)
                result.append((indicator, value))

    def _add_macd_indicators(self, latest_data: pd.Series, data: pd.DataFrame, result: list):
        """Add MACD composite indicators to result."""
        macd_columns = ["MACD", "MACDh", "MACDs"]
        if all(col in data.columns for col in macd_columns):
            macd_data = {col: self._extract_value(latest_data, col) for col in macd_columns}
            result.append(("MACD", macd_data))

    def _add_bollinger_bands(self, latest_data: pd.Series, data: pd.DataFrame, result: list):
        """Add Bollinger Bands composite indicators to result."""
        bb_columns = ["BBL", "BBM", "BBU"]
        bb_keys = ["Lower", "Middle", "Upper"]
        if all(col in data.columns for col in bb_columns):
            bb_data = {key: self._extract_value(latest_data, col) for col, key in zip(bb_columns, bb_keys, strict=True)}
            result.append(("BB", bb_data))

    def _add_volume_indicators(self, latest_data: pd.Series, data: pd.DataFrame, result: list):
        """Add volume indicators to result."""
        for indicator in ["OBV", "CMF"]:
            if indicator in data.columns:
                value = self._extract_value(latest_data, indicator)
                result.append((indicator, value))

    def _build_technical_result(self, data: pd.DataFrame) -> list:
        """Build the final result list from calculated indicators."""
        latest_data = data.iloc[-1]
        result = []

        self._add_basic_indicators(latest_data, data, result)
        self._add_macd_indicators(latest_data, data, result)
        self._add_bollinger_bands(latest_data, data, result)
        self._add_volume_indicators(latest_data, data, result)

        return result
