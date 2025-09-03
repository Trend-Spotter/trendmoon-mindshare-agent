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

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class PositionMonitoringRound(BaseState):
    """This class implements the behaviour of the state PositionMonitoringRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.POSITIONMONITORINGROUND
        self.positions_to_exit: list[dict[str, Any]] = []
        self.position_updates: list[dict[str, Any]] = []
        self.pending_positions: list[dict[str, Any]] = []
        self.completed_positions: list[dict[str, Any]] = []
        self.monitoring_initialized: bool = False
        self.started_data: datetime | None = None

    def setup(self) -> None:
        """Perform the setup."""
        self._is_done = False
        self.positions_to_exit = []
        self.position_updates = []
        self.pending_positions = []
        self.completed_positions = []
        self.monitoring_initialized = False
        self.started_data = None
        super().setup()

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")

        try:
            self._initialize_monitoring()

            if not self.pending_positions and not self.completed_positions:
                self.context.logger.info("No open positions to monitor")
                self._event = MindshareabciappEvents.POSITIONS_CHECKED
                self._is_done = True
                return

            if self.pending_positions:
                position = self.pending_positions.pop(0)
                self._monitor_single_position(position)

                total_positions = len(self.completed_positions) + len(self.pending_positions)
                self.context.logger.debug(
                    f"Monitored position {position['symbol']} "
                    f"({len(self.completed_positions)}/{total_positions} complete)"
                )

                if self.pending_positions:
                    return

            self._finalize_monitoring()

        except Exception as e:
            self.context.logger.exception(f"Position monitoring failed: {e!s}")
            self.context.error_context = {
                "error_type": "position_monitoring_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _initialize_monitoring(self) -> None:
        """Initialize position monitoring on first call."""
        if self.monitoring_initialized:
            return

        self.context.logger.info("Initializing position monitoring...")

        # Load open positions from persistent storage
        positions = self._load_open_positions()
        self.pending_positions = positions.copy()
        self.completed_positions = []
        self.positions_to_exit = []
        self.position_updates = []

        self.started_at = datetime.now(UTC)
        self.monitoring_initialized = True

        self.context.logger.info(f"Initialized monitoring for {len(self.pending_positions)} positions")

        self._is_done = True
        self._event = MindshareabciappEvents.POSITIONS_CHECKED

    def _monitor_single_position(self, position: dict[str, Any]) -> None:
        """Monitor a single position for exit conditions."""
        try:
            position_updated = self._monitor_position(position)

            if position_updated.get("exit_signal"):
                self.positions_to_exit.append(position_updated)
                self.context.logger.info(
                    f"Exit signal detected for {position['symbol']}: {position_updated['exit_reason']}"
                )
            else:
                self.position_updates.append(position_updated)

            self.completed_positions.append(position_updated)

        except (KeyError, ValueError, TypeError) as e:
            self.context.logger.warning(f"Failed to monitor position {position.get('symbol', 'unknown')}: {e}")
            # Add to completed anyway to prevent getting stuck
            self.completed_positions.append(position)
        except Exception as e:
            self.context.logger.exception(
                f"Unexpected error monitoring position {position.get('symbol', 'unknown')}: {e}"
            )
            # Add to completed anyway to prevent getting stuck
            self.completed_positions.append(position)

    def _finalize_monitoring(self) -> None:
        """Finalize monitoring and determine transition."""
        if self.started_at:
            monitoring_time = (datetime.now(UTC) - self.started_at).total_seconds()
            self.context.logger.info(
                f"Position monitoring completed in {monitoring_time:.1f}s "
                f"({len(self.completed_positions)} positions processed)"
            )

        # Store updated positions
        all_updated_positions = self.position_updates + self.positions_to_exit
        if all_updated_positions:
            self._update_positions_storage(all_updated_positions)

        # Determine transition based on exit signals
        if self.positions_to_exit:
            self.context.positions_to_exit = self.positions_to_exit
            self.context.logger.info(f"Found {len(self.positions_to_exit)} positions to exit")
            self._event = MindshareabciappEvents.EXIT_SIGNAL
        else:
            self._event = MindshareabciappEvents.POSITIONS_CHECKED

        self._is_done = True

    def _load_open_positions(self) -> list[dict[str, Any]]:
        """Load open positions from persistent storage."""
        if not self.context.store_path:
            return []

        positions_file = self.context.store_path / "positions.json"
        if not positions_file.exists():
            return []

        try:
            with open(positions_file, encoding="utf-8") as f:
                data = json.load(f)
                return [pos for pos in data.get("positions", []) if pos.get("status") == "open"]
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.context.logger.warning(f"Failed to load positions file: {e}")
            return []
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.context.logger.warning(f"Failed to parse positions data: {e}")
            return []
        except Exception as e:
            self.context.logger.exception(f"Unexpected error loading positions: {e}")
            return []

    def _monitor_position(self, position: dict[str, Any]) -> dict[str, Any]:
        """Monitor a single position for exit conditions."""
        symbol = position["symbol"]
        entry_price = position["entry_price"]
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        position_size = position["position_size"]

        # Get current price from collected data
        current_price = self._get_current_price(symbol)
        if current_price is None:
            self.context.logger.warning(f"No current price data for {symbol}")
            return {**position, "exit_signal": False}

        # Calculate current P&L
        unrealized_pnl = (current_price - entry_price) * position_size
        pnl_percentage = ((current_price - entry_price) / entry_price) * 100

        rsi = self._get_current_rsi(symbol)

        # Update position with current data
        updated_position = {
            **position,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "pnl_percentage": pnl_percentage,
            "last_updated": datetime.now(UTC).isoformat(),
            "exit_signal": False,
            "exit_reason": None,
        }

        # Exit if RSI > 79
        if rsi and rsi > 79:
            updated_position.update(
                {
                    "exit_signal": True,
                    "exit_reason": "rsi_overbought",
                    "exit_price": current_price,
                    "exit_type": "rsi_exit",
                }
            )
            return updated_position

        # Check tailing stop loss activation
        entry_price = position["entry_price"]
        if current_price >= entry_price * self.context.params.trailing_stop_loss_activation_level and not position.get(
            "tailing_stop_active"
        ):
            trailing_stop = current_price * self.context.params.trailing_stop_loss_pct
            updated_position["stop_loss_price"] = max(
                trailing_stop,
                position.get("stop_loss_price", 0),
            )
            updated_position["tailing_stop_active"] = True

        # Check stop-loss conditions
        if stop_loss and self._check_stop_loss(current_price, stop_loss):
            updated_position.update(
                {"exit_signal": True, "exit_reason": "stop_loss", "exit_price": current_price, "exit_type": "stop_loss"}
            )
            return updated_position

        # Check take-profit conditions
        if take_profit and self._check_take_profit(current_price, take_profit):
            updated_position.update(
                {
                    "exit_signal": True,
                    "exit_reason": "take_profit",
                    "exit_price": current_price,
                    "exit_type": "take_profit",
                }
            )
            return updated_position

        # Update trailing stop if configured
        return self._update_trailing_stop(updated_position, current_price)

    def _get_current_price(self, symbol: str) -> float | None:
        """Get current price for a symbol from collected data."""
        price = None

        try:
            # Check if we have collected data available
            if not self.context.store_path:
                return price

            data_file = self.context.store_path / "collected_data.json"
            if not data_file.exists():
                return price

            with open(data_file, encoding="utf-8") as f:
                collected_data = json.load(f)

            # Get current price from collected price data
            current_prices = collected_data.get("current_prices", {})
            if symbol in current_prices:
                price_data = current_prices[symbol]
                price = price_data.get("usd")  # All assets are priced in USD from CoinGecko

        except (FileNotFoundError, PermissionError, OSError) as e:
            self.context.logger.warning(f"Failed to access price data file for {symbol}: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.context.logger.warning(f"Failed to parse price data for {symbol}: {e}")
        except Exception as e:
            self.context.logger.exception(f"Unexpected error getting current price for {symbol}: {e}")
        return price

    def _check_stop_loss(self, current_price: float, stop_loss: float) -> bool:
        """Check if stop-loss condition is triggered."""
        return current_price <= stop_loss

    def _check_take_profit(self, current_price: float, take_profit: float) -> bool:
        """Check if take-profit condition is triggered."""
        return current_price >= take_profit

    def _update_trailing_stop(self, position: dict[str, Any], current_price: float) -> dict[str, Any]:
        """Update trailing stop if configured."""
        if not position.get("trailing_stop_enabled"):
            return position

        trailing_distance = position.get("trailing_distance", 0.05)  # 5% default
        current_stop = position.get("stop_loss")

        new_stop = current_price * (1 - trailing_distance)
        if current_stop is None or new_stop > current_stop:
            position["stop_loss"] = new_stop
            self.context.logger.info(f"Updated trailing stop for {position['symbol']}: {new_stop:.6f}")

        return position

    def _update_positions_storage(self, updated_positions: list[dict[str, Any]]) -> None:
        """Update positions in persistent storage."""
        if not self.context.store_path:
            return

        positions_file = self.context.store_path / "positions.json"

        try:
            # Load existing data
            existing_data = {"positions": [], "last_updated": None}
            if positions_file.exists():
                with open(positions_file, encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Update positions with new data
            existing_positions = {pos["position_id"]: pos for pos in existing_data.get("positions", [])}

            for updated_pos in updated_positions:
                position_id = updated_pos["position_id"]
                existing_positions[position_id] = updated_pos

            all_positions = list(existing_positions.values())

            # Summary statistics
            open_positions = [pos for pos in all_positions if pos.get("status") == "open"]
            total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in open_positions)
            total_portfolio_value = sum(pos.get("entry_value_usdc", 0) for pos in open_positions)

            # Save updated data
            updated_data = {
                "positions": all_positions,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_positions": len(all_positions),
                "open_positions": len(open_positions),
                "total_unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_portfolio_value": round(total_portfolio_value, 2),
            }

            with open(positions_file, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, indent=2)

            self.context.logger.info(
                f"Updated {len(updated_positions)} positions in storage "
                f"(Total P&L: ${total_unrealized_pnl:.2f}, "
                f"Total Portfolio Value: ${total_portfolio_value:.2f})"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to update positions storage: {e}")
