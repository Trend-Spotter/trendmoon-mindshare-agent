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
from datetime import UTC, datetime

from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    TradingStrategy,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class SignalAggregationRound(BaseState):
    """This class implements the behaviour of the state SignalAggregationRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.SIGNALAGGREGATIONROUND
        self.aggregation_initialized: bool = False
        self.analysis_results: dict[str, Any] = {}
        self.collected_data: dict[str, Any] = {}
        self.candidate_signals: list[dict[str, Any]] = []
        self.aggregated_signal: dict[str, Any] | None = None

    def setup(self) -> None:
        """Setup the state."""
        self._is_done = False
        self.aggregation_initialized = False
        self.analysis_results = {}
        self.collected_data = {}
        self.candidate_signals = []
        self.aggregated_signal = None

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")

        try:
            if not self.aggregation_initialized:
                self._initialize_aggregation()

            if not self._load_analysis_results():
                self.context.logger.error("Failed to load analysis results from previous round.")
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

            self._generate_candidate_signals()

            if self.candidate_signals:
                self._select_best_signal()

                if self.aggregated_signal:
                    self._store_aggregated_signal()
                    self.context.logger.info(
                        f"Trade signal generated for {self.aggregated_signal['symbol'] }"
                        f"with strength {self.aggregated_signal['signal_strength']}"
                    )
                    self._event = MindshareabciappEvents.SIGNAL_GENERATED
                else:
                    self.context.logger.info("No signals met minimum criteria.")
                    self._event = MindshareabciappEvents.NO_SIGNAL

            else:
                self.context.logger.info("No candidate signals generated.")
                self._event = MindshareabciappEvents.NO_SIGNAL

            self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Signal aggregation failed: {e}")
            self.context.error_context = {
                "error_type": "signal_aggregation_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _initialize_aggregation(self) -> None:
        """Initialize signal aggregation."""
        self.context.logger.info("Initializing signal aggregation...")

        # Initialize consolidated trading strategy
        strategy_type = getattr(self.context.params, "trading_strategy", "balanced")
        self.trading_strategy = TradingStrategy(strategy_type, self.context)

        self.context.logger.info(f"Initialized {strategy_type} trading strategy")

        self.aggregation_initialized = True

    def _load_analysis_results(self) -> bool:
        """Load analysis results from previous round."""
        try:
            if self._load_from_context():
                return True

            if not self.context.store_path:
                self.context.logger.warning("No store path available")
                return False

            return self._load_from_storage()

        except Exception as e:
            self.context.logger.exception(f"Failed to load analysis results: {e}")
            return False

    def _load_from_context(self) -> bool:
        """Load analysis results from context if available."""
        if not hasattr(self.context, "analysis_results") or not self.context.analysis_results:
            return False

        self.analysis_results = self.context.analysis_results
        self.context.logger.info("Loaded analysis results from context")

        if self.context.store_path:
            self._load_collected_data()

        return True

    def _load_from_storage(self) -> bool:
        """Load analysis results from storage files."""
        analysis_file = self.context.store_path / "analysis_results.json"
        if not analysis_file.exists():
            self.context.logger.warning("No analysis results file found")
            return False

        with open(analysis_file, encoding=DEFAULT_ENCODING) as f:
            self.analysis_results = json.load(f)

        self._load_collected_data()
        self.context.logger.info("Successfully loaded analysis results from storage")
        return True

    def _load_collected_data(self) -> None:
        """Load collected data from storage if available."""
        data_file = self.context.store_path / "collected_data.json"
        if data_file.exists():
            with open(data_file, encoding=DEFAULT_ENCODING) as f:
                self.collected_data = json.load(f)
            self.context.logger.debug("Loaded collected data from storage")
        else:
            self.context.logger.warning("No collected data file found")

    def _generate_candidate_signals(self) -> None:
        """Generate trade signals for each analyzed token."""
        token_analysis = self.analysis_results.get("token_analysis", {})

        for symbol, analysis in token_analysis.items():
            if isinstance(analysis, dict) and "error" not in analysis:
                signal = self._evaluate_token_signal(symbol, analysis)
                if signal:
                    self.candidate_signals.append(signal)
                    self.context.logger.info(
                        f"Generated candidate signal for {symbol}: "
                        f"direction={signal['direction']}, p_trade={signal['p_trade']:.3f}"
                    )

    def _evaluate_token_signal(self, symbol: str, analysis: dict[str, Any]) -> dict[str, Any] | None:
        """Evaluate if a token meets trading conditions and generate signal."""
        try:
            # Get technical and social scores from analysis
            technical_scores = self.analysis_results.get("technical_scores", {}).get(symbol, {})
            social_scores = self.analysis_results.get("social_scores", {}).get(symbol, {})

            # Get current price
            current_prices = self.collected_data.get("current_prices", {}).get(symbol, {})
            current_price = current_prices.get("usd", 0) if current_prices else 0

            if current_price <= 0:
                self.context.logger.warning(f"No valid price for {symbol}")
                return None

            # Use consolidated trading strategy to evaluate conditions
            conditions_met = self.trading_strategy.evaluate_conditions(current_price, social_scores, technical_scores)

            # Calculate signal strength and count met conditions
            signal_strength, num_conditions_met = self.trading_strategy.calculate_signal_strength(conditions_met)

            # Log conditions for debugging
            met_conditions = {k: v["condition"] for k, v in conditions_met.items()}
            self.context.logger.debug(
                f"Signal evaluation for {symbol}: "
                f"conditions_met={num_conditions_met}, "
                f"signal_strength={signal_strength:.3f}, "
                f"details={met_conditions}"
            )

            # Check if signal should be generated (all 4 core conditions must be met)
            if not self.trading_strategy.should_generate_signal(conditions_met):
                # Identify which core conditions are missing
                core_conditions = self.trading_strategy.params["core_conditions"]
                missing_conditions = [
                    cond
                    for cond in core_conditions
                    if cond not in conditions_met or not conditions_met[cond]["condition"]
                ]
                self.context.logger.debug(
                    f"Signal not generated for {symbol}: " f"Missing core conditions: {missing_conditions}"
                )
                return None

            # Create trade signal
            return {
                "signal_id": f"sig_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "symbol": symbol,
                "direction": "buy",
                "p_trade": signal_strength,
                "signal_strength": self._classify_signal_strength(signal_strength),
                "conditions_met": met_conditions,
                "conditions_details": conditions_met,
                "num_conditions_met": num_conditions_met,
                "analysis_scores": {
                    "p_social": analysis.get("p_social", 0),
                    "p_technical": analysis.get("p_technical", 0),
                    "p_combined": analysis.get("p_combined", 0),
                },
                "indicators": {
                    "current_price": current_price,
                    "sma": technical_scores.get("sma"),
                    "rsi": technical_scores.get("rsi"),
                    "macd": technical_scores.get("macd"),
                    "macd_signal": technical_scores.get("macd_signal"),
                    "adx": technical_scores.get("adx"),
                    "obv": technical_scores.get("obv"),
                },
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "generated",
            }

        except Exception as e:
            self.context.logger.exception(f"Failed to evaluate signal for {symbol}: {e}")
            return None

    def _classify_signal_strength(self, score: float) -> str:
        """Classify signal strength based on score."""
        if score >= 0.85:
            return "very_strong"
        elif score >= 0.75:
            return "strong"
        elif score >= 0.65:
            return "moderate"
        elif score >= 0.55:
            return "weak"
        else:
            return "very_weak"

    def _select_best_signal(self) -> None:
        """Select the best signal from candidates."""
        if not self.candidate_signals:
            return

        # Sort by p_trade (signal strength) descending
        sorted_signals = sorted(
            self.candidate_signals, key=operator.itemgetter("p_trade", "num_conditions_met"), reverse=True
        )

        # Select the top signal
        best_signal = sorted_signals[0]

        # Select the best signal (all signals here have passed core conditions check)
        self.aggregated_signal = best_signal

        self.context.logger.info(
            f"Selected best signal: {best_signal['symbol']} "
            f"with p_trade={best_signal['p_trade']:.3f}, "
            f"conditions_met={best_signal['num_conditions_met']}"
        )

        # Log runner-ups if any
        if len(sorted_signals) > 1:
            self.context.logger.info("Runner-up signals:")
            for signal in sorted_signals[1:3]:  # Show top 3
                self.context.logger.info(f"  {signal['symbol']}: p_trade={signal['p_trade']:.3f}")

    def _store_aggregated_signal(self) -> None:
        """Store the aggregated signal for the next round."""
        try:
            # Store in context for immediate access
            self.context.aggregated_trade_signal = self.aggregated_signal

            # Also persist to storage
            if self.context.store_path:
                signals_file = self.context.store_path / "signals.json"

                # Load existing signals
                signals_data = {"signals": [], "last_signal": None}
                if signals_file.exists():
                    with open(signals_file, encoding=DEFAULT_ENCODING) as f:
                        signals_data = json.load(f)

                # Add new signal
                signals_data["signals"].append(self.aggregated_signal)
                signals_data["last_signal"] = self.aggregated_signal
                signals_data["last_updated"] = datetime.now(UTC).isoformat()

                # Keep only last 100 signals
                if len(signals_data["signals"]) > 100:
                    signals_data["signals"] = signals_data["signals"][-100:]

                # Save updated signals
                with open(signals_file, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(signals_data, f, indent=2)

                self.context.logger.info(f"Stored aggregated signal for {self.aggregated_signal['symbol']}")

        except Exception as e:
            self.context.logger.exception(f"Failed to store aggregated signal: {e}")
