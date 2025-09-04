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
from typing import TYPE_CHECKING, Any
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from collections.abc import Generator

from autonomy.deploy.constants import DEFAULT_ENCODING
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.eightballer.connections.dcxt import PUBLIC_ID as DCXT_PUBLIC_ID
from packages.eightballer.protocols.tickers.message import TickersMessage
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    ALLOWED_ASSETS,
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


if TYPE_CHECKING:
    from packages.eightballer.protocols.tickers.custom_types import Ticker

MAX_DEVIATION = 0.50  # 50% deviation threshold
WARNING_DEVIATION = 0.20  # 20% deviation for warnings
PRICE_COLLECTION_TIMEOUT_SECONDS = 180


@dataclass
class PriceRequest:
    """Price request."""

    symbol: str
    exchange_id: str
    ledger_id: str
    ticker_dialogue: Any
    request_timestamp: datetime


class TradeConstructionRound(BaseState):
    """This class implements the behaviour of the state TradeConstructionRound."""

    supported_protocols = {
        TickersMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.TRADECONSTRUCTIONROUND
        self.construction_initialized: bool = False
        self.trade_signal: dict[str, Any] = {}
        self.constructed_trade: dict[str, Any] = {}
        self.market_data: dict[str, Any] = {}
        self.risk_parameters: dict[str, Any] = {}
        self.pending_price_requests: list[str] = []
        self.price_collection_started: bool = False
        self.started_at: datetime | None = None
        self.open_positions: list[dict[str, Any]] = []

    def setup(self) -> None:
        """Setup the state."""
        self._is_done = False
        self.construction_initialized = False
        self.trade_signal = {}
        self.constructed_trade = {}
        self.market_data = {}
        self.risk_parameters = {}
        self.pending_price_requests = []
        self.price_collection_started = False
        self.started_at = None
        self.open_positions = []
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def act(self) -> Generator:
        """Perform the act."""
        try:
            if self._process_trade_construction():
                self._event = MindshareabciappEvents.DONE
                self._is_done = True
            elif self._is_timeout_reached():
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Error in {self._state} state: {e!s}")
            self.context.error_context = {
                "error_type": "trade_construction_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

        return None

    def _load_open_positions(self) -> list[dict[str, Any]]:
        """Load the open positions from the previous round."""
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

    def _process_trade_construction(self) -> bool:
        """Process the trade construction workflow step by step."""
        if not self.construction_initialized:
            self.context.logger.info(f"Entering {self._state} state.")
            self._initialize_construction()

            if not self._load_trade_signal():
                self.context.logger.error("Failed to load trade signal from previous round.")
                self._set_error_state()
                return False

        if not self.price_collection_started:
            self.context.logger.info("Starting price collection.")
            self._start_price_collection()
            return False

        if not self._check_price_collection_complete():
            return False

        construction_steps = [
            (self._process_price_responses, "Failed to process price responses"),
            (self._construct_trade_parameters, "Failed to construct trade parameters"),
            (self._validate_constructed_trade, "Failed to validate constructed trade."),
        ]

        for step_func, error_msg in construction_steps:
            if not step_func():
                self.context.logger.error(error_msg)
                self._set_error_state()
                return False

        self._store_constructed_trade()
        self._log_construction_summary()
        self.context.logger.info("Trade construction completed successfully.")
        return True

    def _is_timeout_reached(self) -> bool:
        """Check if timeout has been reached during price collection."""
        return self.started_at and datetime.now(UTC) - self.started_at > timedelta(
            seconds=PRICE_COLLECTION_TIMEOUT_SECONDS
        )

    def _set_error_state(self) -> None:
        """Set the error state and mark as done."""
        self._event = MindshareabciappEvents.ERROR
        self._is_done = True

    def _initialize_construction(self) -> None:
        """Initialize the construction."""
        if self.construction_initialized:
            return

        self.context.logger.info("Initializing trade construction.")

        self.open_positions = self._load_open_positions()

        self.risk_parameters = {
            "strategy": self.context.params.trading_strategy,
            "max_slippage": self.context.params.max_slippage_bps / 10000,
            "stop_loss_multiplier": self.context.params.stop_loss_atr_multiplier,
            "take_profit_ratio": self.context.params.take_profit_risk_ratio,
            "min_position_size": self.context.params.min_position_size_usdc,
            "max_position_size": self.context.params.max_position_size_usdc,
        }

        self.construction_initialized = True

    def _load_trade_signal(self) -> bool:
        """Load the trade signal from the previous round."""
        try:
            if hasattr(self.context, "approved_trade_signal") and self.context.approved_trade_signal:
                self.trade_signal = self.context.approved_trade_signal
                self.context.logger.info(
                    f"Loaded approved trade signal for {self.trade_signal.get('symbol', 'Unknown')}"
                )
                return True

            if not self.context.store_path:
                self.context.logger.warning("No store path available, skipping trade signal loading.")
                return False

            signals_file = self.context.store_path / "signals.json"
            if not signals_file.exists():
                self.context.logger.warning("No signals file found")
                return False

            with open(signals_file, encoding=DEFAULT_ENCODING) as f:
                signals_data = json.load(f)

            latest_signal = signals_data.get("last_signal")

            if not latest_signal or latest_signal.get("status") != "approved":
                self.context.logger.warning("No approved signal found, skipping trade signal loading.")
                return False

            self.trade_signal = latest_signal
            self.context.logger.info(f"Loaded approved trade signal for {self.trade_signal.get('symbol', 'Unknown')}")
            return True

        except Exception as e:
            self.context.logger.exception(f"Error loading trade signal: {e!s}")
            return False

    def _start_price_collection(self) -> None:
        """Start price collection for the trade signal."""
        try:
            symbol = self.trade_signal.get("symbol")
            if not symbol:
                self.context.logger.error("No symbol in trade signal")
                return

            token_info = self._get_token_info(symbol)
            if not token_info:
                self.context.logger.error(f"Failed to get token info for {symbol}")
                return

            price_request = self._submit_ticker_request(token_info)
            if price_request:
                self.context.logger.info(f"Submitted ticker request for {symbol}: {price_request}")
                self.pending_price_requests.append(price_request)
                self.price_collection_started = True
                self.started_at = datetime.now(UTC)
                self.context.logger.info(f"Started price collection for {symbol}")

        except Exception as e:
            self.context.logger.exception(f"Error requesting market prices: {e!s}")

    def _get_token_info(self, symbol: str) -> dict[str, Any] | None:
        """Get token info from allowed assets."""
        for token in ALLOWED_ASSETS["base"]:
            if token.get("symbol") == symbol:
                return token
        return None

    def _submit_ticker_request(self, token_info: dict[str, str]) -> PriceRequest | None:
        """Submit a price request to the DCXT connection."""
        try:
            symbol = token_info.get("symbol")

            trading_pair = f"{symbol}/USDC"

            min_position = self.context.params.min_position_size_usdc
            max_position = self.context.params.max_position_size_usdc
            estimated_position_size = (min_position + max_position) / 2
            # Use a placeholder amount for ticker requests since we need price to calculate position size
            ticker_amount = max(estimated_position_size, 100.0)  # Use 1 USDC as standard amount for price discovery

            ticker_dialogue = self.submit_msg(
                TickersMessage.Performative.GET_TICKER,
                connection_id=str(DCXT_PUBLIC_ID),
                exchange_id="cowswap",
                ledger_id="base",
                symbol=trading_pair,
                params=json.dumps({"amount": ticker_amount}).encode(DEFAULT_ENCODING),
            )

            ticker_dialogue.validation_func = self._validate_ticker_msg
            ticker_dialogue.exchange_id = "cowswap"
            ticker_dialogue.ledger_id = "base"
            ticker_dialogue.symbol = symbol
            ticker_dialogue.trading_pair = trading_pair

            price_request = PriceRequest(
                symbol=symbol,
                exchange_id="cowswap",
                ledger_id="base",
                ticker_dialogue=ticker_dialogue,
                request_timestamp=datetime.now(UTC),
            )

            self.context.logger.info(f"Submitted ticker request for {trading_pair}")
            return price_request

        except Exception as e:
            self.context.logger.exception(f"Error submitting ticker request: {e!s}")
            return None

    def _validate_ticker_msg(self, ticker_msg: TickersMessage, _dialogue: BaseDialogue) -> bool:
        """Validate the ticker message."""
        try:
            if ticker_msg is None:
                return False

            if not isinstance(ticker_msg, TickersMessage):
                return False

            if ticker_msg.performative == TickersMessage.Performative.ERROR:
                self.context.logger.warning(
                    f"Received error from DCXT: {getattr(ticker_msg, 'error_msg', 'Unknown error')}"
                )
                return False

            if ticker_msg.performative == TickersMessage.Performative.TICKER and ticker_msg.ticker is not None:
                self.context.logger.info(f"Received valid ticker: {ticker_msg.ticker.symbol}")
                self.context.logger.info(f"Ticker: {ticker_msg.ticker.dict()}")
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating ticker message: {e!s}")
            return False

    def _check_price_collection_complete(self) -> bool:
        """Check if all price collection is complete."""
        try:
            # Check if we have received responses for all requests
            received_responses = len(self.supported_protocols.get(TickersMessage.protocol_id, []))
            expected_responses = len(self.pending_price_requests)

            if received_responses < expected_responses:
                self.context.logger.debug(f"Waiting for price responses: {received_responses}/{expected_responses}")
                return False

            # Validate all received messages
            for price_request in self.pending_price_requests:
                dialogue = price_request.ticker_dialogue
                if not dialogue.validation_func(dialogue.last_incoming_message, dialogue):
                    self.context.logger.error(f"Invalid ticker message for {price_request.symbol}")
                    return False

            return True

        except Exception as e:
            self.context.logger.exception(f"Error checking price collection completion: {e}")
            return False

    def _process_price_responses(self) -> bool:
        """Process the collected price responses from DCXT."""
        try:
            if not self.pending_price_requests:
                self.context.logger.error("No price requests to process")
                return False

            # Get the price request (should be only one for the target symbol)
            price_request = self.pending_price_requests[0]
            dialogue = price_request.ticker_dialogue
            ticker_msg = dialogue.last_incoming_message

            if not ticker_msg or not ticker_msg.ticker:
                self.context.logger.error("No ticker data in response")
                return False

            ticker: Ticker = ticker_msg.ticker

            # Extract pricing data from DCXT ticker
            current_price = float(ticker.bid)  # Use bid price for buying
            bid_price = float(ticker.bid)
            ask_price = float(ticker.ask)

            # Store market data
            self.market_data = {
                "current_price": current_price,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "symbol": ticker.symbol,
                "exchange": "balancer",
                "timestamp": ticker.datetime,
                "dcxt_ticker": ticker.dict(),
                # Calculate effective spread
                "spread": (ask_price - bid_price) / current_price if current_price > 0 else 0,
                "volume_24h": getattr(ticker, "volume", 0),
                "high_24h": getattr(ticker, "high", current_price),
                "low_24h": getattr(ticker, "low", current_price),
            }

            self.context.logger.info(
                f"Processed DCXT pricing for {ticker.symbol}: "
                f"${current_price:.6f} (bid: ${bid_price:.6f}, ask: ${ask_price:.6f})"
            )
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing price responses: {e}")
            return False

    def _validate_price_sanity(self, symbol: str, ticker_price: float) -> bool:
        """Validate ticker price against CoinGecko reference price with sanity checks."""
        try:
            # Check prerequisites and load data
            validation_data = self._load_price_validation_data(symbol)
            if not validation_data:
                return True  # Skip validation if data unavailable

            coingecko_usd_price = validation_data["coingecko_usd_price"]
            ticker_price_usd = ticker_price

            # Calculate price deviation percentage
            price_deviation = abs(ticker_price_usd - coingecko_usd_price) / coingecko_usd_price

            # Log price comparison for debugging
            self.context.logger.info(
                f"Price validation for {symbol}: "
                f"Ticker (USDC): ${ticker_price:.6f}, "
                f"CoinGecko (USD): ${coingecko_usd_price:.6f}, "
                f"Deviation: {price_deviation:.2%}"
            )

            # Check for extreme deviations (error condition)
            if price_deviation > MAX_DEVIATION:
                self._handle_price_validation_failure(symbol, ticker_price, coingecko_usd_price, price_deviation)
                return False

            # Log warning for significant deviations
            if price_deviation > WARNING_DEVIATION:
                self.context.logger.warning(
                    f"Significant price deviation detected for {symbol}: "
                    f"{price_deviation:.2%} deviation from CoinGecko reference"
                )

            return True

        except Exception as e:
            self.context.logger.exception(f"Error during price sanity check for {symbol}: {e}")
            # Don't fail the trade on validation errors, just log them
            return True

    def _load_price_validation_data(self, symbol: str) -> dict | None:
        """Load and validate price data prerequisites."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available for price validation")
            return None

        data_file = self.context.store_path / "collected_data.json"
        if not data_file.exists():
            self.context.logger.warning(f"Data file does not exist for price validation: {data_file}")
            return None

        with open(data_file, encoding=DEFAULT_ENCODING) as f:
            collected_data = json.load(f)

        # Get current prices from collected data
        current_prices = collected_data.get("current_prices", {})
        if symbol not in current_prices:
            self.context.logger.warning(f"No CoinGecko price data found for {symbol}")
            return None

        coingecko_price_data = current_prices[symbol]
        coingecko_usd_price = coingecko_price_data.get("usd")

        if not coingecko_usd_price:
            self.context.logger.warning(f"No USD price found in CoinGecko data for {symbol}")
            return None

        return {"coingecko_usd_price": coingecko_usd_price}

    def _handle_price_validation_failure(
        self, symbol: str, ticker_price: float, coingecko_usd_price: float, price_deviation: float
    ) -> None:
        """Handle price validation failure by logging error and setting context."""
        error_msg = (
            f"Price sanity check failed for {symbol}: "
            f"Ticker price ${ticker_price:.6f} deviates {price_deviation:.2%} "
            f"from CoinGecko reference ${coingecko_usd_price:.6f}"
        )

        self.context.logger.error(error_msg)

        self.context.error_context = {
            "error_type": "price_sanity_check_failed",
            "error_message": error_msg,
            "error_data": {
                "symbol": symbol,
                "ticker_price_usdc": ticker_price,
                "coingecko_price_usd": coingecko_usd_price,
                "price_deviation": price_deviation,
                "max_allowed_deviation": MAX_DEVIATION,
                "validation_timestamp": datetime.now(UTC).isoformat(),
            },
        }

    def _construct_trade_parameters(self) -> bool:
        """Construct complete trade parameters using DCXT pricing."""
        try:
            symbol = self.trade_signal.get("symbol")
            trade_direction = self.trade_signal.get("direction", "buy")
            signal_strength = self.trade_signal.get("signal_strength", 0.5)

            token_info = self._get_token_info(symbol)
            if not token_info:
                return False

            position_size_usdc = self._calculate_position_size(signal_strength)

            current_price = self.market_data.get("current_price", 0)
            if current_price <= 0:
                self.context.logger.error("Invalid current price from DCXT")

                self.context.error_context = {
                    "error_type": "invalid_price_error",
                    "error_message": "Invalid current price from DCXT",
                    "error_data": {
                        "current_price": current_price,
                        "market_data": self.market_data,
                    },
                }

                return False

            # Add price sanity check against CoinGecko data
            if not self._validate_price_sanity(symbol, current_price):
                return False

            slippage_tolerance = self._calculate_slippage_tolerance()
            stop_loss_price = self._calculate_stop_loss_price(current_price, trade_direction)
            take_profit_price = self._calculate_take_profit_price(current_price, trade_direction, stop_loss_price)

            if trade_direction == "buy":
                execution_price = self.market_data.get("ask_price", current_price)
            else:
                execution_price = self.market_data.get("bid_price", current_price)

            token_quantity = position_size_usdc / execution_price

            self.constructed_trade = {
                "trade_id": f"trade_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "constructed",
                # Asset details
                "symbol": symbol,
                "contract_address": token_info["address"],
                "coingecko_id": token_info["coingecko_id"],
                "trading_pair": f"{symbol}/USDC",
                # Trade direction and size
                "direction": trade_direction,
                "position_size_usdc": round(position_size_usdc, 2),
                "token_quantity": round(token_quantity, 6),
                # DCXT pricing
                "entry_price": current_price,
                "bid_price": self.market_data.get("bid_price", current_price),
                "ask_price": self.market_data.get("ask_price", current_price),
                "dcxt_spread": self.market_data.get("spread", 0),
                # Risk management
                "stop_loss_price": round(stop_loss_price, 6),
                "take_profit_price": round(take_profit_price, 6),
                "slippage_tolerance": slippage_tolerance,
                # DCXT execution parameters
                "execution_strategy": "dcxt",
                "exchange_id": "balancer",
                "ledger_id": "base",
                "max_execution_time": 300,  # 5 minutes
                "partial_fills_allowed": True,
                # Order configuration for DCXT
                "sell_token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC on Base
                "buy_token": token_info["address"],
                "sell_amount_usdc": position_size_usdc,
                "min_buy_amount": token_quantity * (1 - slippage_tolerance),
                # Signal context
                "signal_strength": signal_strength,
                "originating_signal": self.trade_signal.get("signal_id"),
                # Market data snapshot
                "market_data": self.market_data,
                # Risk parameters used
                "risk_parameters": self.risk_parameters,
            }

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to construct trade parameters: {e}")
            return False

    def _calculate_position_size(self, _signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk parameters."""
        try:
            # Get available trading capital from portfolio validation
            # Check if we have portfolio metrics from previous validation
            available_capital = getattr(self.context, "available_trading_capital", None)
            if available_capital is None:
                self.context.logger.error("No available trading capital found.")
                return 0.0
            self.context.logger.info(
                f"Using available trading capital from portfolio validation: ${available_capital:.2f}"
            )

            # Get portfolio metrics to calculate total portfolio value
            portfolio_metrics = getattr(self.context, "portfolio_metrics", {})
            total_exposure = portfolio_metrics.get("total_exposure", 0.0)

            # Total Portfolio Value = Available Cash + Existing Positions
            total_portfolio_value = available_capital + total_exposure

            # Each position should be 1/7 of total portfolio value
            target_position_size = total_portfolio_value * (1.0 / 7.0)

            # But we can only use available capital (minus buffer)
            min_capital_buffer = portfolio_metrics.get("min_capital_buffer", 500.0)
            usable_capital = available_capital - min_capital_buffer

            # Position size is the minimum of target and usable capital
            position_size = min(target_position_size, usable_capital)

            # Apply minimum position size constraint
            min_size = self.risk_parameters["min_position_size"]
            position_size = max(min_size, position_size)

            max_positions = self.context.params.max_positions
            current_positions = len(self.open_positions)
            if current_positions >= max_positions:
                self.context.logger.info(f"Max positions reached: {current_positions}/{max_positions}")
                return 0.0

            self.context.logger.info(
                f"Position sizing: Total portfolio: ${total_portfolio_value:.2f}, "
                f"Target per position: ${target_position_size:.2f}, "
                f"Actual position size: ${position_size:.2f}"
            )

            return position_size

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate position size: {e}")
            return self.risk_parameters["min_position_size"]

    def _calculate_slippage_tolerance(self) -> float:
        """Calculate slippage tolerance based on DCXT market conditions."""
        try:
            # Base slippage from configuration
            base_slippage = self.risk_parameters["max_slippage"]

            # Adjust based on DCXT spread
            spread = self.market_data.get("spread", 0)

            # Add spread to base slippage with minimum protection
            slippage = max(base_slippage, spread * 2.0)  # Use 2x spread as minimum
            slippage = min(slippage, 0.05)  # Cap at 5%

            return round(slippage, 4)

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate slippage tolerance: {e}")
            return self.risk_parameters["max_slippage"]

    def _calculate_stop_loss_price(self, current_price: float, direction: str) -> float:
        """Calculate stop-loss price based on ATR and risk parameters."""
        try:
            # Get ATR-based stop loss multiplier
            atr_multiplier = self.risk_parameters["stop_loss_multiplier"]

            # Calculate ATR approximation from DCXT market data
            high_24h = self.market_data.get("high_24h", current_price * 1.02)
            low_24h = self.market_data.get("low_24h", current_price * 0.98)
            estimated_atr = abs(high_24h - low_24h) / 2  # Simple ATR estimate

            # Ensure minimum ATR
            estimated_atr = max(estimated_atr, current_price * 0.015)  # Min 1.5% ATR

            stop_distance = estimated_atr * atr_multiplier

            stop_loss = current_price - stop_distance if direction == "buy" else current_price + stop_distance

            return max(stop_loss, 0.000001)  # Ensure positive price

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate stop loss price: {e}")
            return current_price * 0.90 if direction == "buy" else current_price * 1.10

    def _calculate_take_profit_price(self, current_price: float, direction: str, stop_loss_price: float) -> float:
        """Calculate take-profit price based on risk-reward ratio."""
        try:
            # Calculate risk (distance to stop loss)
            if direction == "buy":
                risk = current_price - stop_loss_price
                reward = risk * self.risk_parameters["take_profit_ratio"]
                take_profit = current_price + reward
            else:  # sell
                risk = stop_loss_price - current_price
                reward = risk * self.risk_parameters["take_profit_ratio"]
                take_profit = current_price - reward

            return max(take_profit, 0.000001)  # Ensure positive price

        except Exception as e:
            self.context.logger.exception(f"Failed to calculate take profit price: {e}")
            return current_price * 1.20 if direction == "buy" else current_price * 0.80

    def _validate_constructed_trade(self) -> bool:
        """Validate the constructed trade for completeness and sanity."""
        try:
            errors = []

            # Perform all validation checks
            errors.extend(self._validate_required_fields())
            errors.extend(self._validate_price_relationships())
            errors.extend(self._validate_position_size())
            errors.extend(self._validate_slippage_tolerance())

            # Report all errors or success
            if errors:
                for error in errors:
                    self.context.logger.error(error)
                return False

            self.context.logger.info("Trade construction validation passed")
            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to validate constructed trade: {e}")
            return False

    def _validate_required_fields(self) -> list[str]:
        """Validate that all required fields are present and valid."""
        required_fields = [
            "trade_id",
            "symbol",
            "contract_address",
            "direction",
            "position_size_usdc",
            "token_quantity",
            "entry_price",
            "stop_loss_price",
            "take_profit_price",
            "slippage_tolerance",
            "sell_token",
            "buy_token",
            "exchange_id",
            "ledger_id",
        ]

        errors = []
        for field in required_fields:
            if field not in self.constructed_trade:
                errors.append(f"Missing required field: {field}")
                continue

            value = self.constructed_trade[field]
            if value is None or (isinstance(value, int | float) and value <= 0):
                errors.append(f"Invalid value for {field}: {value}")

        return errors

    def _validate_price_relationships(self) -> list[str]:
        """Validate price relationships for buy orders."""
        errors = []

        if self.constructed_trade.get("direction") == "buy":
            entry = self.constructed_trade.get("entry_price", 0)
            stop = self.constructed_trade.get("stop_loss_price", 0)
            take_profit = self.constructed_trade.get("take_profit_price", 0)

            if stop >= entry:
                errors.append(f"Invalid stop loss: {stop} >= {entry}")

            if take_profit <= entry:
                errors.append(f"Invalid take profit: {take_profit} <= {entry}")

        return errors

    def _validate_position_size(self) -> list[str]:
        """Validate position size is within reasonable bounds."""
        errors = []
        position_size = self.constructed_trade.get("position_size_usdc", 0)

        if position_size < self.risk_parameters["min_position_size"]:
            errors.append(f"Position size too small: ${position_size}")

        if position_size > self.risk_parameters["max_position_size"]:
            errors.append(f"Position size too large: ${position_size}")

        return errors

    def _validate_slippage_tolerance(self) -> list[str]:
        """Validate slippage tolerance is within acceptable range."""
        errors = []
        slippage = self.constructed_trade.get("slippage_tolerance", 0)

        if slippage > 0.10:  # 10% max
            errors.append(f"Slippage tolerance too high: {slippage * 100:.2f}%")

        return errors

    def _store_constructed_trade(self) -> None:
        """Store the constructed trade for the ExecutionRound."""
        try:
            # Store in context for immediate access by ExecutionRound
            self.context.constructed_trade = self.constructed_trade

            # Also persist to storage
            if self.context.store_path:
                trades_file = self.context.store_path / "pending_trades.json"

                # Load existing pending trades
                pending_trades = {"trades": []}
                if trades_file.exists():
                    with open(trades_file, encoding=DEFAULT_ENCODING) as f:
                        pending_trades = json.load(f)

                # Add new trade
                pending_trades["trades"].append(self.constructed_trade)
                pending_trades["last_updated"] = datetime.now(UTC).isoformat()

                # Save updated trades
                with open(trades_file, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(pending_trades, f, indent=2)

                self.context.logger.info(f"Stored constructed trade: {self.constructed_trade['trade_id']}")

        except Exception as e:
            self.context.logger.exception(f"Failed to store constructed trade: {e}")

    def _log_construction_summary(self) -> None:
        """Log a summary of the constructed trade."""
        try:
            trade = self.constructed_trade
            symbol = trade["symbol"]
            direction = trade["direction"]
            position_size = trade["position_size_usdc"]
            entry_price = trade["entry_price"]
            stop_loss = trade["stop_loss_price"]
            take_profit = trade["take_profit_price"]
            slippage = trade["slippage_tolerance"]

            self.context.logger.info("=== DCXT Trade Construction Summary ===")
            self.context.logger.info(f"Trade ID: {trade['trade_id']}")
            self.context.logger.info(f"Asset: {symbol} ({trade['contract_address'][:10]}...)")
            self.context.logger.info(f"Direction: {direction.upper()}")
            self.context.logger.info(f"Position Size: ${position_size:.2f}")
            self.context.logger.info(f"Token Quantity: {trade['token_quantity']:.6f}")
            self.context.logger.info(f"Entry Price: ${entry_price:.6f}")
            self.context.logger.info(f"DCXT Spread: {trade['dcxt_spread']:.4f}%")
            self.context.logger.info(f"Stop Loss: ${stop_loss:.6f}")
            self.context.logger.info(f"Take Profit: ${take_profit:.6f}")
            self.context.logger.info(f"Slippage Tolerance: {slippage * 100:.2f}%")
            self.context.logger.info(f"Exchange: {trade['exchange_id']} on {trade['ledger_id']}")

            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss) * trade["token_quantity"]
            reward_amount = abs(take_profit - entry_price) * trade["token_quantity"]
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            self.context.logger.info(f"Risk Amount: ${risk_amount:.2f}")
            self.context.logger.info(f"Reward Amount: ${reward_amount:.2f}")
            self.context.logger.info(f"Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
            self.context.logger.info("==========================================")

        except Exception as e:
            self.context.logger.exception(f"Failed to log construction summary: {e}")
