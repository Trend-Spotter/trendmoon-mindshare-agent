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
from typing import Any
from datetime import UTC, datetime

from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class RiskEvaluationRound(BaseState):
    """This class implements the behaviour of the state RiskEvaluationRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.RISKEVALUATIONROUND
        self.evaluation_intitalized: bool = False
        self.trade_signal: dict[str, Any] = {}
        self.candidate_signals: list[dict[str, Any]] = []
        self.risk_assessment: dict[str, Any] = {}
        self.approved_trade_proposal: dict[str, Any] = {}
        self.open_positions: list[dict[str, Any]] = []

    def setup(self) -> None:
        """Setup the state."""
        self._is_done = False
        self.evaluation_intitalized = False
        self.trade_signal = {}
        self.candidate_signals = []
        self.risk_assessment = {}
        self.approved_trade_proposal = {}
        self.open_positions = []

    def act(self) -> None:
        """Perform the act."""
        try:
            if not self.evaluation_intitalized:
                self.context.logger.info(f"Entering {self._state} state.")
                self._initialize_risk_evaluation()

            if not self._load_trade_signal():
                self.context.logger.error("Failed to load trade signal from previous round.")
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

            if not self._load_open_positions():
                self.context.logger.error("Failed to load open positions for duplication check.")
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

            # Try to evaluate the main signal first
            if self._evaluate_signal(self.trade_signal, signal_type="main"):
                self._create_approved_trade_proposal()
                self._store_approved_trade_proposal()
                self.context.logger.info("Risk evaluation passed - trade approved")
                self._event = MindshareabciappEvents.APPROVED
                self._is_done = True
                return

            # If main signal failed, try runner-up signals
            if len(self.candidate_signals) > 1:
                self.context.logger.info(
                    f"Main signal failed evaluation, trying {len(self.candidate_signals) - 1} runner-up signal(s)"
                )

                # Skip the first candidate (it's the main signal we just tried)
                for idx, candidate in enumerate(self.candidate_signals[1:], start=1):
                    self.context.logger.info(
                        f"Evaluating runner-up #{idx}: {candidate.get('symbol')} "
                        f"(p_trade={candidate.get('p_trade', 0):.3f})"
                    )

                    # Reset risk assessment for new signal
                    self._initialize_risk_evaluation()

                    if self._evaluate_signal(candidate, signal_type=f"runner-up #{idx}"):
                        # Update the trade signal to the successful runner-up
                        self.trade_signal = candidate
                        # Update context to reflect the selected signal
                        self.context.aggregated_trade_signal = candidate

                        self._create_approved_trade_proposal()
                        self._store_approved_trade_proposal()

                        self.context.logger.info(
                            f"Runner-up #{idx} ({candidate.get('symbol')}) passed evaluation - trade approved"
                        )
                        self._event = MindshareabciappEvents.APPROVED
                        self._is_done = True
                        return

            # All signals failed evaluation
            self.context.logger.info("All signals failed risk evaluation")
            self._event = MindshareabciappEvents.REJECTED
            self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Risk evaluation failed: {e}")
            self.context.error_context = {
                "error_type": "risk_evaluation_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _evaluate_signal(self, signal: dict[str, Any], signal_type: str = "main") -> bool:
        """Evaluate a single trade signal through all risk checks."""
        # Temporarily set trade_signal to the signal being evaluated
        original_signal = self.trade_signal
        self.trade_signal = signal

        try:
            evaluation_steps = [
                (self._check_trading_pair_duplication, "trading pair duplication check"),
                (self._calculate_volatility_metrics, "volatility metrics calculation"),
                (self._apply_risk_vetoes, "risk vetoes"),
                (self._calculate_risk_levels, "risk levels calculation"),
                (self._validate_risk_parameters, "risk parameters validation"),
            ]

            for step_func, step_name in evaluation_steps:
                if not step_func():
                    self.context.logger.info(f"{signal_type} signal failed at: {step_name}")
                    return False

            self.context.logger.info(f"{signal_type} signal passed all risk checks")
            return True

        finally:
            # Always restore original signal - it gets updated in act() if evaluation succeeds
            self.trade_signal = original_signal

    def _initialize_risk_evaluation(self) -> None:
        """Initialize the risk evaluation."""
        self.evaluation_intitalized = True

        self.risk_assessment = {
            "volatility_atr": 0.0,
            "volatility_tier": "unknown",
            "onchain_risk_score": 0.0,
            "risk_vetoes": [],
            "stop_loss_distance": 0.0,
            "take_profit_distance": 0.0,
            "risk_reward_ratio": 0.0,
            "max_loss_percentage": 0.0,
            "evaluation_timestamp": datetime.now(UTC).isoformat(),
        }

    def _load_open_positions(self) -> bool:
        """Load open positions from persistent storage for duplication check."""
        try:
            if not self.context.store_path:
                self.context.logger.warning("No store path available")
                return False

            positions_file = self.context.store_path / "positions.json"
            if positions_file.exists():
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
                    positions_data = json.load(f)

                self.open_positions = [
                    pos for pos in positions_data.get("positions", []) if pos.get("status") == "open"
                ]
                self.context.logger.info(f"Loaded {len(self.open_positions)} open positions for duplication check")
            else:
                self.context.logger.info("No positions file found - assuming empty portfolio")
                self.open_positions = []

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to load open positions: {e}")
            return False

    def _check_trading_pair_duplication(self) -> bool:
        """Check if there are already open positions in the same trading pair."""
        if not self.trade_signal or not self.trade_signal.get("symbol"):
            self.context.logger.error("No trade signal symbol available for duplication check")
            return False

        target_symbol = self.trade_signal.get("symbol", "").upper()

        # Check if we already have a position in this trading pair
        existing_positions = []
        for pos in self.open_positions:
            pos_symbol = pos.get("symbol", "").upper()

            # Handle both direct symbol matching and trading pair notation
            if pos_symbol == target_symbol:
                existing_positions.append(pos)
            elif "/" in pos_symbol:
                # Handle trading pair notation like "OLAS/USDC"
                base_symbol = pos_symbol.split("/")[0]
                if base_symbol == target_symbol:
                    existing_positions.append(pos)
            elif "/" not in pos_symbol and pos_symbol and pos_symbol == target_symbol:
                # Assume USDC quote for single symbols
                existing_positions.append(pos)

        if existing_positions:
            self.context.logger.warning(
                f"Trading pair duplication check: FAILED - Already have open position(s) "
                f" in {target_symbol}: {len(existing_positions)} position(s)"
            )

            # Log details of existing positions
            for i, pos in enumerate(existing_positions):
                entry_value = pos.get("position_size_usdc", 0)
                pnl = pos.get("unrealized_pnl", 0)
                side = pos.get("direction", "unknown")
                self.context.logger.info(f"  Existing position {i + 1}: {side} ${entry_value:.2f} (P&L: ${pnl:.2f})")

            return False

        self.context.logger.info(f"Trading pair duplication check: PASSED - no existing {target_symbol} positions")
        return True

    def _load_trade_signal(self) -> bool:
        """Load the trade signal and candidate signals from the previous round."""
        try:
            if hasattr(self.context, "aggregated_trade_signal") and self.context.aggregated_trade_signal:
                self.trade_signal = self.context.aggregated_trade_signal
                self.context.logger.info(
                    f"Loaded aggregated trade signal for {self.trade_signal.get('symbol', 'Unknown')}"
                )
                # Try to load candidate signals from storage even if main signal is in context
                self._load_candidate_signals_from_storage()
                return True

            if not self.context.store_path:
                self.context.logger.warning("No store path available for loading trade signal.")
                return False

            signals_file = self.context.store_path / "signals.json"
            if not signals_file.exists():
                self.context.logger.warning("No signals file found")
                return False

            with open(signals_file, encoding=DEFAULT_ENCODING) as f:
                signals_data = json.load(f)

            latest_signal = signals_data.get("last_signal")

            if not latest_signal or latest_signal.get("status") != "generated":
                self.context.logger.warning("No valid generated signal found")
                return False

            self.trade_signal = latest_signal

            # Load candidate signals (including the main signal)
            self.candidate_signals = signals_data.get("candidate_signals", [])

            num_runner_ups = max(0, len(self.candidate_signals) - 1)
            self.context.logger.info(
                f"Loaded aggregated trade signal for {self.trade_signal.get('symbol', 'Unknown')} "
                f"with {num_runner_ups} runner-up(s)"
            )
            return True

        except Exception as e:
            self.context.logger.exception(f"Error loading trade signal: {e}")
            return False

    def _load_candidate_signals_from_storage(self) -> None:
        """Load candidate signals from storage file."""
        try:
            if not self.context.store_path:
                return

            signals_file = self.context.store_path / "signals.json"
            if not signals_file.exists():
                return

            with open(signals_file, encoding=DEFAULT_ENCODING) as f:
                signals_data = json.load(f)

            self.candidate_signals = signals_data.get("candidate_signals", [])

        except (OSError, json.JSONDecodeError, KeyError) as e:
            self.context.logger.debug(f"Could not load candidate signals from storage: {e}")

    def _calculate_volatility_metrics(self) -> bool:
        """Calculate the volatility metrics using ATR-based approach."""
        try:
            symbol = self.trade_signal.get("symbol")
            if not symbol:
                return False

            ohlcv_data = self._get_ohlcv_data(symbol)
            if not ohlcv_data:
                self.context.logger.warning(f"No OHLCV data found for {symbol}")
                # Use default ATR estimation
                current_price = self._get_current_price(symbol)
                estimated_atr = current_price * 0.03 if current_price else 0  # 3% default ATR
                self.risk_assessment["volatility_atr"] = estimated_atr
                self.risk_assessment["volatility_tier"] = "high"  # Conservative assumption
                return True

            atr = self._calculate_atr(ohlcv_data)
            self.risk_assessment["volatility_atr"] = atr

            current_price = self._get_current_price(symbol)
            if current_price and current_price > 0:
                atr_percentage = (atr / current_price) * 100

                if atr_percentage >= 8.0:
                    volatility_tier = "very_high"
                elif atr_percentage >= 5.0:
                    volatility_tier = "high"
                elif atr_percentage >= 2.0:
                    volatility_tier = "medium"
                else:
                    volatility_tier = "low"

                self.risk_assessment["volatility_tier"] = volatility_tier
                self.risk_assessment["volatility_atr"] = atr_percentage

                self.context.logger.info(
                    f"Volatility for {symbol}: {volatility_tier} (ATR: {atr:.6f}, {atr_percentage:.2f}%)"
                )
            else:
                self.risk_assessment["volatility_tier"] = "unknown"

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate volatility metrics: {e}")
            return False

    def _apply_risk_vetoes(self) -> bool:
        """Apply risk vetoes to the trade signal."""
        try:
            vetoes = []

            volatility_tier = self.risk_assessment.get("volatility_tier")
            if volatility_tier == "very_high":
                vetoes.append(
                    {"type": "volatility", "reason": "Volatility tier is very high - too risky", "severity": "medium"}
                )

            p_trade = self.trade_signal.get("p_trade", 0.0)
            min_signal_threshold = 0.6  # Minimum 60% confidence
            if p_trade < min_signal_threshold:
                vetoes.append(
                    {
                        "type": "signal_strength",
                        "reason": f"Trade signal too weak: {p_trade:.3f} < {min_signal_threshold}",
                        "severity": "high",
                    }
                )

            self.risk_assessment["risk_vetoes"] = vetoes

            high_severity_vetoes = [v for v in vetoes if v["severity"] == "high"]
            if high_severity_vetoes:
                self.context.logger.warning(
                    f"Trade rejected due to {len(high_severity_vetoes)} high-severity risk veto(s)"
                )
                for veto in high_severity_vetoes:
                    self.context.logger.warning(f"  - {veto['type']}: {veto['reason']}")
                return False

            # Log medium-severity vetoes as warnings but don't reject
            medium_severity_vetoes = [v for v in vetoes if v["severity"] == "medium"]
            if medium_severity_vetoes:
                self.context.logger.warning(f"Trade has {len(medium_severity_vetoes)} medium-severity risk warning(s)")
                for veto in medium_severity_vetoes:
                    self.context.logger.warning(f"  - {veto['type']}: {veto['reason']}")

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to apply risk vetoes: {e}")
            return False

    def _calculate_risk_levels(self) -> bool:
        """Calculate the risk levels based on position size and volatility."""
        try:
            symbol = self.trade_signal.get("symbol")
            current_price = self._get_current_price(symbol)
            if not current_price or current_price <= 0.0:
                self.context.logger.warning(f"No valid current price found for {symbol}")
                return False

            stop_loss_pct = self.context.params.stop_loss_pct
            tailing_stop_loss_pct = self.context.params.trailing_stop_loss_pct
            trailing_activation = self.context.params.trailing_stop_loss_activation_level

            stop_loss_price = current_price * stop_loss_pct

            risk_amount = current_price - stop_loss_price
            take_profit_price = current_price + (risk_amount * 2)

            self.risk_assessment.update(
                {
                    "current_price": current_price,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "tailing_stop_loss_pct": tailing_stop_loss_pct,
                    "trailing_activation_level": trailing_activation,
                }
            )

            self.context.logger.info(
                f"Risk levels for {symbol}: "
                f"SL: ${stop_loss_price:.6f} ({-stop_loss_pct:.1f}%), "
                f"TP: ${take_profit_price:.6f} (1:{risk_amount:.1f} R/R)"
            )

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate risk levels: {e}")
            return False

    def _validate_risk_parameters(self) -> bool:
        """Validate the risk parameters."""
        try:
            # Check max loss percentage is acceptable
            max_loss_pct = self.risk_assessment["max_loss_percentage"]
            max_acceptable_loss = 10.0  # 10% maximum loss per trade
            if max_loss_pct > max_acceptable_loss:
                self.context.logger.error(
                    f"Maximum loss percentage too high: {max_loss_pct:.1f}% > {max_acceptable_loss}%"
                )
                return False

            # Check stop-loss and take-profit are valid
            current_price = self.risk_assessment["current_price"]
            stop_loss = self.risk_assessment["stop_loss_price"]
            take_profit = self.risk_assessment["take_profit_price"]
            direction = self.trade_signal.get("direction", "buy")

            if direction == "buy":
                if stop_loss >= current_price:
                    self.context.logger.error(f"Invalid stop-loss for buy: {stop_loss} >= {current_price}")
                    return False
                if take_profit <= current_price:
                    self.context.logger.error(f"Invalid take-profit for buy: {take_profit} <= {current_price}")
                    return False

            self.context.logger.info("Risk parameter validation passed")
            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to validate risk parameters: {e}")
            return False

    def _create_approved_trade_proposal(self) -> None:
        """Create an approved trade proposal."""
        self.approved_trade_proposal = {
            "signal_id": self.trade_signal.get("signal_id"),
            "symbol": self.trade_signal.get("symbol"),
            "direction": self.trade_signal.get("direction"),
            "entry_price": self.risk_assessment["current_price"],
            "stop_loss_price": self.risk_assessment["stop_loss_price"],
            "take_profit_price": self.risk_assessment["take_profit_price"],
            "risk_assessment": self.risk_assessment.copy(),
            "original_signal": self.trade_signal.copy(),
            "approval_timestamp": datetime.now(UTC).isoformat(),
            "status": "approved",
        }

    def _store_approved_trade_proposal(self) -> None:
        """Store the approved trade proposal for the next round."""
        try:
            # Store in context for immediate access by TradeConstructionRound
            self.context.approved_trade_signal = self.approved_trade_proposal

            # Also persist to storage
            if self.context.store_path:
                signals_file = self.context.store_path / "signals.json"

                # Load existing signals
                signals_data = {"signals": [], "last_signal": None}
                if signals_file.exists():
                    with open(signals_file, encoding=DEFAULT_ENCODING) as f:
                        signals_data = json.load(f)

                # Update the last signal with approval
                signals_data["last_signal"] = self.approved_trade_proposal
                signals_data["last_updated"] = datetime.now(UTC).isoformat()

                # Save updated signals
                with open(signals_file, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(signals_data, f, indent=2)

                self.context.logger.info("Stored approved trade proposal")

        except Exception as e:
            self.context.logger.exception(f"Failed to store approved trade: {e}")

    # Helper methods

    def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from collected data."""
        try:
            if not self.context.store_path:
                return None

            data_file = self.context.store_path / "collected_data.json"
            if not data_file.exists():
                return None

            with open(data_file, encoding=DEFAULT_ENCODING) as f:
                collected_data = json.load(f)

            current_prices = collected_data.get("current_prices", {})
            if symbol in current_prices:
                price_data = current_prices[symbol]
                return price_data.get("usd")

            return None

        except Exception as e:
            self.context.logger.exception(f"Failed to get current price for {symbol}: {e}")
            return None

    def _get_ohlcv_data(self, symbol: str) -> list[dict] | None:
        """Get OHLCV data from collected data."""
        try:
            if not self.context.store_path:
                return None

            data_file = self.context.store_path / "collected_data.json"
            if not data_file.exists():
                return None

            with open(data_file, encoding=DEFAULT_ENCODING) as f:
                collected_data = json.load(f)

            ohlcv_data = collected_data.get("ohlcv", {})
            return ohlcv_data.get(symbol)

        except Exception as e:
            self.context.logger.exception(f"Failed to get OHLCV data for {symbol}: {e}")
            return None

    def _calculate_atr(self, ohlcv_data: list[dict], period: int = 14) -> float:
        """Calculate Average True Range from OHLCV data."""
        try:
            if not ohlcv_data or len(ohlcv_data) < 2:
                return 0.0

            true_ranges = []
            for i in range(1, min(len(ohlcv_data), period + 1)):
                current = ohlcv_data[i]
                previous = ohlcv_data[i - 1]

                high = current[2]
                low = current[3]
                prev_close = previous[4]

                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)

                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)

            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate ATR: {e}")
            return 0.0
