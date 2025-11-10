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
from datetime import UTC, datetime, timedelta

from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.eightballer.protocols.orders.custom_types import OrderSide
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

        # Pending orders tracking
        self.pending_trades: list[dict[str, Any]] = []
        self.pending_orders_checked: bool = False
        self.pending_orders_request_submitted: bool = False

    def setup(self) -> None:
        """Perform the setup."""
        self._is_done = False
        self.positions_to_exit = []
        self.position_updates = []
        self.pending_positions = []
        self.completed_positions = []
        self.monitoring_initialized = False
        self.pending_trades = []
        self.pending_orders_checked = False
        self.pending_orders_request_submitted = False
        super().setup()

    def act(self) -> None:
        """Perform the act."""
        try:
            # Only log on first entry
            if not self.monitoring_initialized:
                self.context.logger.info(f"Entering {self._state} state.")
                self._initialize_monitoring()

            # First, check pending orders if not done yet
            if not self.pending_orders_checked:
                self._check_pending_orders()
                return

            # Then monitor existing positions
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

        # Don't mark as done here - let the act() method handle completion
        # after checking pending orders and monitoring positions

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
            with open(positions_file, encoding=DEFAULT_ENCODING) as f:
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
        stop_loss = position.get("stop_loss_price")
        take_profit = position.get("take_profit_price")
        token_quantity = position["token_quantity"]

        # Get current price from collected data
        current_price = self._get_current_price(symbol)
        if current_price is None:
            self.context.logger.warning(f"No current price data for {symbol}")
            return {**position, "exit_signal": False}

        # Calculate current P&L
        unrealized_pnl = (current_price - entry_price) * token_quantity
        pnl_percentage = ((current_price - entry_price) / entry_price) * 100

        # Calculate current market value in USDC
        current_value_usdc = current_price * token_quantity

        rsi = self._get_current_rsi(symbol)

        # Update position with current data
        updated_position = {
            **position,
            "current_price": current_price,
            "current_value_usdc": current_value_usdc,
            "unrealized_pnl": unrealized_pnl,
            "pnl_percentage": pnl_percentage,
            "last_updated": datetime.now(UTC).isoformat(),
            "exit_signal": False,
            "exit_reason": None,
        }

        # Exit if RSI > overbought limit
        if rsi and rsi > self.context.params.rsi_overbought_limit:
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

            with open(data_file, encoding=DEFAULT_ENCODING) as f:
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
        current_stop = position.get("stop_loss_price")

        new_stop = current_price * (1 - trailing_distance)
        if current_stop is None or new_stop > current_stop:
            position["stop_loss_price"] = new_stop
            self.context.logger.info(f"Updated trailing stop for {position['symbol']}: {new_stop:.6f}")

        return position

    def _get_current_rsi(self, symbol: str) -> float | None:
        """Get current RSI for a symbol from analysis results with freshness check."""
        try:
            analysis_data = self._load_analysis_data()
            if not analysis_data:
                return None

            rsi_value = self._extract_rsi_value(analysis_data, symbol)
            if rsi_value is None:
                return None

            if not self._is_analysis_data_fresh(analysis_data, symbol):
                return None

            self.context.logger.debug(f"Retrieved current RSI for {symbol}: {rsi_value:.2f}")
            return float(rsi_value)

        except Exception as e:
            self.context.logger.exception(f"Unexpected error getting RSI for {symbol}: {e}")
            return None

    def _load_analysis_data(self) -> dict | None:
        """Load analysis data from persistent storage."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available for RSI data")
            return None

        analysis_file = self.context.store_path / "analysis_results.json"
        if not analysis_file.exists():
            self.context.logger.warning("No analysis results file found for RSI data")
            return None

        try:
            with open(analysis_file, encoding=DEFAULT_ENCODING) as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as e:
            self.context.logger.warning(f"Failed to load analysis data: {e}")
            return None

    def _extract_rsi_value(self, analysis_data: dict, symbol: str) -> float | None:
        """Extract RSI value for a specific symbol from analysis data."""
        technical_scores = analysis_data.get("technical_scores", {})
        symbol_data = technical_scores.get(symbol)
        if not symbol_data:
            self.context.logger.warning(f"No technical scores found for {symbol}")
            return None

        rsi_value = symbol_data.get("rsi")
        if rsi_value is None:
            self.context.logger.warning(f"No RSI value found for {symbol}")
            return None

        return rsi_value

    def _is_analysis_data_fresh(self, analysis_data: dict, symbol: str) -> bool:
        """Check if analysis data is fresh (within 4 hours)."""
        analysis_timestamp = analysis_data.get("timestamp")
        if not analysis_timestamp:
            return True  # No timestamp means we can't validate, assume fresh

        try:
            analysis_time = datetime.fromisoformat(analysis_timestamp)
            current_time = datetime.now(UTC)
            time_diff = current_time - analysis_time

            # Consider data stale if older than 4 hours (4h candle updates)
            if time_diff > timedelta(hours=4):
                self.context.logger.warning(
                    f"RSI data for {symbol} is stale ({time_diff.total_seconds() / 3600:.1f} hours old)"
                )
                return False

            return True
        except (ValueError, TypeError) as e:
            self.context.logger.warning(f"Failed to parse analysis timestamp: {e}")
            return True  # If we can't parse, assume fresh

    def _update_positions_storage(self, updated_positions: list[dict[str, Any]]) -> None:
        """Update positions in persistent storage."""
        if not self.context.store_path:
            return

        positions_file = self.context.store_path / "positions.json"

        try:
            # Load existing data
            existing_data = {"positions": [], "last_updated": None}
            if positions_file.exists():
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
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
            total_portfolio_value = sum(pos.get("position_size_usdc", 0) for pos in open_positions)

            # Save updated data
            updated_data = {
                "positions": all_positions,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_positions": len(all_positions),
                "open_positions": len(open_positions),
                "total_unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_portfolio_value": round(total_portfolio_value, 2),
            }

            with open(positions_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(updated_data, f, indent=2)

            self.context.logger.info(
                f"Updated {len(updated_positions)} positions in storage "
                f"(Total P&L: ${total_unrealized_pnl:.2f}, "
                f"Total Portfolio Value: ${total_portfolio_value:.2f})"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to update positions storage: {e}")

    def _check_pending_orders(self) -> None:
        """Check status of pending orders using async pattern."""
        # Step 1: Submit request if not already submitted
        if not self.pending_orders_request_submitted:
            self.pending_trades = self._load_pending_trades()

            if not self.pending_trades:
                self.context.logger.info("No pending orders to check")
                self.pending_orders_checked = True
                return

            # Extract CoWSwap order IDs to monitor
            cowswap_order_ids = []
            for trade in self.pending_trades:
                cowswap_order_id = trade.get("cowswap_order_id")
                if cowswap_order_id:
                    cowswap_order_ids.append(cowswap_order_id)

            if cowswap_order_ids:
                self.context.logger.info(f"Checking status of {len(cowswap_order_ids)} pending CoWSwap orders")
                self.monitor_cowswap_orders(cowswap_order_ids)
                self.pending_orders_request_submitted = True
                return  # Exit early, wait for response
            self.context.logger.info("No CoWSwap order IDs found in pending trades")
            self.pending_orders_checked = True
            return

        # Step 2: Check if response received (will be processed by validation func)
        # The validation function sets pending_orders_checked = True when response is processed
        if self.pending_orders_checked:
            return

        # Still waiting for response
        self.context.logger.debug("Waiting for pending orders response...")

    def _process_order_updates(self, order_updates: dict[str, dict[str, Any]]) -> None:
        """Process order updates from CoWSwap monitoring."""
        try:
            updated_trades = []

            for trade in self.pending_trades:
                cowswap_order_id = trade.get("cowswap_order_id")
                if cowswap_order_id in order_updates:
                    update_info = order_updates[cowswap_order_id]
                    status = update_info["status"]
                    order = update_info["order"]

                    if status == "filled":
                        self.context.logger.info(
                            f"Order {cowswap_order_id} filled - creating position from pending trade"
                        )
                        self._create_position_from_filled_order(cowswap_order_id, order, trade)
                    elif status in {"cancelled", "expired"}:
                        self.context.logger.warning(f"Order {cowswap_order_id} {status} - removing from pending")
                        # Don't add to updated_trades (effectively removes it)
                    else:
                        # Order still open, keep in pending
                        updated_trades.append(trade)
                        self.context.logger.info(f"Order {cowswap_order_id} still open")
                else:
                    # No update for this trade, keep it
                    updated_trades.append(trade)

            # Update pending trades storage
            self._update_pending_trades(updated_trades)

            self.pending_orders_checked = True
            self.context.logger.info(f"Completed pending order check. {len(updated_trades)} orders still pending")

        except Exception as e:
            self.context.logger.exception(f"Failed to process order updates: {e}")
            self.pending_orders_checked = True

    def _create_position_from_filled_order(self, cowswap_order_id: str, order: Any, trade: dict[str, Any]) -> None:
        """Create a new position from filled CoWSwap order and pending trade data.

        Args:
        ----
            cowswap_order_id: The CoWSwap order ID
            order: The filled Order object with actual execution details from CoW API
            trade: The pending trade dict with metadata from pending_trades.json

        """
        if not self.context.store_path:
            self.context.logger.warning("No store path available to create position")
            return

        positions_file = self.context.store_path / "positions.json"

        try:
            # Load existing positions (or create new structure)
            existing_data = {"positions": [], "last_updated": None}
            if positions_file.exists():
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
                    existing_data = json.load(f)

            positions = existing_data.get("positions", [])

            # Create new position from trade + order data
            position = self._create_new_position(trade, order, cowswap_order_id)

            if not position:
                self.context.logger.error(f"Failed to create position from order {cowswap_order_id}")
                return

            # Add new position to list
            positions.append(position)

            # Calculate summary statistics
            open_positions = [pos for pos in positions if pos.get("status") == "open"]
            total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in open_positions)
            total_portfolio_value = sum(pos.get("position_size_usdc", 0) for pos in open_positions)

            # Save updated positions
            updated_data = {
                "positions": positions,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_positions": len(positions),
                "open_positions": len(open_positions),
                "total_unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_portfolio_value": round(total_portfolio_value, 2),
            }

            with open(positions_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(updated_data, f, indent=2)

            self.context.logger.info(
                f"Created position {position['position_id']} for {position['symbol']} "
                f"(entry: ${position['entry_price']:.6f}, quantity: {position['token_quantity']:.6f})"
            )

        except (FileNotFoundError, PermissionError, OSError) as e:
            self.context.logger.exception(f"Failed to access positions file: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.context.logger.exception(f"Failed to parse positions data: {e}")
        except Exception as e:
            self.context.logger.exception(f"Unexpected error creating position from fill: {e}")

    def _create_new_position(self, trade: dict[str, Any], order: Any, cowswap_order_id: str) -> dict[str, Any] | None:  # noqa: PLR0911, PLR0914
        """Create a new position from pending trade data and filled order details.

        This uses the same calculation logic as ExecutionRound to ensure consistency.

        Args:
        ----
            trade: The pending trade dict from pending_trades.json with metadata
            order: The filled Order object from CoW API with execution details
            cowswap_order_id: The CoWSwap order ID

        Returns:
        -------
            dict: The new position dict, or None if creation failed

        """
        try:
            # Get order details from order object or fall back to stored trade data
            has_order_fields = order and hasattr(order, "side") and hasattr(order, "price") and hasattr(order, "amount")

            if not has_order_fields:
                # Order object not available or incomplete, use stored order details from pending_trades.json
                order_price = trade.get("order_price")
                order_amount = trade.get("order_amount")
                side_name = trade.get("order_side")

                if not all([side_name, order_price, order_amount]):
                    self.context.logger.error(
                        f"Order {cowswap_order_id} not available from API and trade data missing required fields "
                        f"(order_side, order_price, or order_amount)"
                    )
                    return None

                # Convert order_side string back to OrderSide enum
                try:
                    order_side = OrderSide[side_name]
                except (KeyError, TypeError):
                    self.context.logger.exception(f"Invalid order_side value in trade data: {side_name}")
                    return None

                self.context.logger.info(
                    f"Using stored order details for {cowswap_order_id}: "
                    f"side={side_name}, price={order_price}, amount={order_amount}"
                )
            else:
                # Get order values from order object
                order_price = getattr(order, "price", 0) or 0
                order_amount = getattr(order, "amount", 0) or 0
                order_side = getattr(order, "side", None)

            if order_price <= 0:
                self.context.logger.error(f"Order {cowswap_order_id} has invalid price: {order_price}")
                return None

            # Calculate execution details using same logic as ExecutionRound
            # CoW Protocol treats "buy token with USDC" as SELL orders (selling USDC)
            if order_side == OrderSide.SELL:
                # CoW SELL order: selling USDC to buy token
                # order.amount = actual USDC spent
                # order.price = actual USDC per token price
                position_size_usdc = order_amount
                executed_price = order_price
                token_quantity = position_size_usdc / executed_price

                self.context.logger.info(
                    f"CoW SELL order (buy token): order_id={cowswap_order_id}, "
                    f"side={order_side.name}, USDC spent={position_size_usdc:.6f}, "
                    f"price={executed_price:.6f}, tokens={token_quantity:.6f}"
                )
            elif order_side == OrderSide.BUY:
                # Standard BUY order or exit orders
                # For BUY orders, order.amount is the token quantity
                executed_price = order_price
                token_quantity = order_amount
                position_size_usdc = token_quantity * executed_price

                self.context.logger.info(
                    f"BUY order: order_id={cowswap_order_id}, "
                    f"side={order_side.name}, tokens={token_quantity:.6f}, "
                    f"price={executed_price:.6f}, USDC amount={position_size_usdc:.6f}"
                )
            else:
                self.context.logger.error(f"Unknown order side for order {cowswap_order_id}: {order_side}")
                return None

            # Extract metadata from trade
            symbol = trade.get("symbol")
            contract_address = trade.get("contract_address") or trade.get("buy_token")
            coingecko_id = trade.get("coingecko_id")
            stop_loss_price = trade.get("stop_loss_price")
            take_profit_price = trade.get("take_profit_price")
            trade_id = trade.get("trade_id")

            # Validate required fields
            if not symbol or not contract_address:
                self.context.logger.error(f"Trade {trade_id} missing required fields (symbol or contract_address)")
                return None

            # Create position ID
            position_id = f"pos_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

            # Build new position dict
            position = {
                "position_id": position_id,
                "order_id": cowswap_order_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "contract_address": contract_address,
                "coingecko_id": coingecko_id,
                "trading_pair": f"{symbol}/USDC",
                "direction": "long",  # All entry trades are long positions
                "entry_price": executed_price,
                "current_price": executed_price,
                "token_quantity": token_quantity,
                "position_size_usdc": position_size_usdc,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "status": "open",
                "unrealized_pnl": 0.0,
                "pnl_percentage": 0.0,
                "current_value_usdc": position_size_usdc,
                "entry_time": datetime.now(UTC).isoformat(),
                "last_updated": datetime.now(UTC).isoformat(),
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None,
                "trailing_stop_enabled": False,
            }

            self.context.logger.info(
                f"Created new position: {position_id} for {symbol}, "
                f"entry_price={executed_price:.6f}, token_quantity={token_quantity:.6f}, "
                f"position_size_usdc={position_size_usdc:.6f}"
            )

            return position

        except Exception as e:
            self.context.logger.exception(f"Error creating position from trade {trade.get('trade_id')}: {e}")
            return None
