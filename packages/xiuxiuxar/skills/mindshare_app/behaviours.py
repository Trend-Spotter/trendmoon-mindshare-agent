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

"""This package contains the Mindshare App behaviour."""

import os
import json
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from collections.abc import Generator

import pandas as pd
import pandas_ta as ta
from eth_utils import to_bytes
from aea.protocols.base import Message
from aea.skills.behaviours import State, FSMBehaviour
from aea.configurations.base import PublicId
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.eightballer.connections.dcxt import PUBLIC_ID as DCXT_PUBLIC_ID
from packages.eightballer.protocols.orders import OrdersMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.contracts.erc20.contract import ERC20
from packages.open_aea.protocols.signing.message import SigningMessage
from packages.valory.contracts.multisend.contract import MultiSendContract, MultiSendOperation
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.eightballer.contracts.erc_20.contract import Erc20
from packages.eightballer.protocols.tickers.message import TickersMessage
from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko, Trendmoon
from packages.valory.protocols.ledger_api.custom_types import Terms, TransactionDigest
from packages.xiuxiuxar.contracts.gnosis_safe.contract import SafeOperation, GnosisSafeContract
from packages.xiuxiuxar.skills.mindshare_app.dialogues import ContractApiDialogue
from packages.eightballer.protocols.orders.custom_types import Order, OrderSide, OrderType, OrderStatus


LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)
DEFAULT_ENCODING = "utf-8"
ETHER_VALUE = 0
PRICE_COLLECTION_TIMEOUT_SECONDS = 60
ORDER_PLACEMENT_TIMEOUT_SECONDS = 30
SAFE_TX_GAS = 300_000  # Non-zero value to prevent Safe revert during gas estimation
MAX_UINT256 = 2**256 - 1
NULL_ADDRESS = "0x" + "0" * 40
BALANCER_VAULT = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"  # Balancer V2 Vault on Base
MULTISEND_ADDRESS = "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"  # Base chain multisend


def truncate_to_decimals(amount: float, decimals: int = 4) -> float:
    """Truncate amount to specified number of decimal places to avoid precision issues."""
    if isinstance(amount, int | float):
        # Multiply by 10^decimals, truncate to int, then divide back
        factor = 10**decimals
        return float(int(amount * factor)) / factor
    return amount


@dataclass
class PriceRequest:
    """Price request."""

    symbol: str
    exchange_id: str
    ledger_id: str
    ticker_dialogue: Any
    request_timestamp: datetime


ALLOWED_ASSETS: dict[str, list[dict[str, str]]] = {
    "base": [
        # BASE-chain Uniswap tokens
        {
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "symbol": "USDC",
            "coingecko_id": "usd-coin",
        },
        {
            "address": "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
            "symbol": "cbETH",
            "coingecko_id": "coinbase-wrapped-staked-eth",
        },
        {
            "address": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
            "symbol": "cbBTC",
            "coingecko_id": "coinbase-wrapped-btc",
        },
        {
            "address": "0x4200000000000000000000000000000000000006",
            "symbol": "WETH",
            "coingecko_id": "l2-standard-bridged-weth-base",
        },
    ]
}


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko, Trendmoon
    from packages.eightballer.protocols.tickers.custom_types import Ticker


class MindshareabciappEvents(Enum):
    """Events for the fsm."""

    REJECTED = "REJECTED"
    FAILED = "FAILED"
    NO_SIGNAL = "NO_SIGNAL"
    RESET = "RESET"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    EXIT_SIGNAL = "EXIT_SIGNAL"
    APPROVED = "APPROVED"
    DONE = "DONE"
    POSITIONS_CHECKED = "POSITIONS_CHECKED"
    RESUME = "RESUME"
    ERROR = "ERROR"
    RETRIES_EXCEEDED = "RETRIES_EXCEEDED"
    AT_LIMIT = "AT_LIMIT"
    EXECUTED = "EXECUTED"
    RETRY = "RETRY"
    CAN_TRADE = "CAN_TRADE"


class MindshareabciappStates(Enum):
    """States for the fsm."""

    DATACOLLECTIONROUND = "datacollectionround"
    PORTFOLIOVALIDATIONROUND = "portfoliovalidationround"
    RISKEVALUATIONROUND = "riskevaluationround"
    PAUSEDROUND = "pausedround"
    CHECKSTAKINGKPIROUND = "checkstakingkpiround"
    SIGNALAGGREGATIONROUND = "signalaggregationround"
    HANDLEERRORROUND = "handleerrorround"
    POSITIONMONITORINGROUND = "positionmonitoringround"
    ANALYSISROUND = "analysisround"
    TRADECONSTRUCTIONROUND = "tradeconstructionround"
    SETUPROUND = "setupround"
    EXECUTIONROUND = "executionround"


class BaseState(State, ABC):
    """Base class for states."""

    _state: MindshareabciappStates = None

    supported_protocols: dict[PublicId, list] = {}

    def setup(self) -> None:
        """Perform the setup."""
        self.started = False
        self._is_done = False
        self._message = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._event = None
        self._is_done = False  # Initially, the state is not done
        self._message = None
        self._performative_to_dialogue_class = None  # Will be initialized lazily

    @property
    def performative_to_dialogue_class(self) -> dict:
        """Get the performative to dialogue class mapping, initializing if needed."""
        if self._performative_to_dialogue_class is None:
            self._performative_to_dialogue_class = {
                OrdersMessage.Performative.CREATE_ORDER: self.context.orders_dialogues,
                OrdersMessage.Performative.GET_ORDERS: self.context.orders_dialogues,
                ContractApiMessage.Performative.GET_STATE: self.context.contract_api_dialogues,
                ContractApiMessage.Performative.GET_RAW_TRANSACTION: self.context.contract_api_dialogues,
                TickersMessage.Performative.GET_TICKER: self.context.tickers_dialogues,
                SigningMessage.Performative.SIGN_TRANSACTION: self.context.signing_dialogues,
                LedgerApiMessage.Performative.SEND_SIGNED_TRANSACTION: self.context.ledger_api_dialogues,
                LedgerApiMessage.Performative.GET_TRANSACTION_RECEIPT: self.context.ledger_api_dialogues,
            }
        return self._performative_to_dialogue_class

    @property
    def coingecko(self) -> "Coingecko":
        """Get the CoinGecko API client."""
        return cast("Coingecko", self.context.coingecko)

    @property
    def trendmoon(self) -> "Trendmoon":
        """Get the Trendmoon API client."""
        return cast("Trendmoon", self.context.trendmoon)

    def act(self) -> None:
        """Perform the act."""
        self._is_done = True
        self._event = MindshareabciappEvents.DONE

    def is_done(self) -> bool:
        """Is done."""
        return self._is_done

    @property
    def event(self) -> str | None:
        """Current event."""
        return self._event

    @classmethod
    def _get_request_nonce_from_dialogue(cls, dialogue: BaseDialogue) -> str:
        """Get the request nonce for the request, from the protocol's dialogue."""
        return dialogue.dialogue_label.dialogue_reference[0]

    def get_dialogue_callback_request(self):
        """Get callback request for dialogue handling."""

        def callback_request(message: Message, dialogue: BaseDialogue) -> None:
            """The callback request."""
            if message.protocol_id in self.supported_protocols:
                self.context.logger.debug(f"Message: {message} {dialogue}")
                self._message = message
                self.supported_protocols.get(message.protocol_id).append(message)

                # Check if this dialogue has a validation function
                if hasattr(dialogue, "validation_func") and callable(dialogue.validation_func):
                    try:
                        # Call the validation function
                        is_valid = dialogue.validation_func(message, dialogue)
                        if is_valid:
                            self.context.logger.debug(
                                "Message validated successfully for dialogue "
                                f"{dialogue.dialogue_label.dialogue_reference[0]}"
                            )
                        else:
                            self.context.logger.warning(
                                "Message validation failed for dialogue "
                                f"{dialogue.dialogue_label.dialogue_reference[0]}"
                            )
                    except Exception as e:
                        self.context.logger.exception(f"Error in validation function: {e}")

                return

            self.context.logger.warning(
                f"Message not supported: {message.protocol_id}. Supported protocols: {self.supported_protocols}"
            )

        return callback_request

    def submit_msg(
        self,
        performative: Message.Performative,
        connection_id: str,
        **kwargs: Any,
    ) -> BaseDialogue:
        """Submit a message and return the dialogue."""

        dialogue_class: BaseDialogue = self.performative_to_dialogue_class[performative]

        # Create message and dialogue
        msg, dialogue = dialogue_class.create(counterparty=connection_id, performative=performative, **kwargs)
        msg._sender = str(self.context.skill_id)  # noqa: SLF001

        request_nonce = self._get_request_nonce_from_dialogue(dialogue)
        self.context.requests.request_id_to_callback[request_nonce] = self.get_dialogue_callback_request()
        self.context.outbox.put_message(message=msg)
        return dialogue


# Define states


class SetupRound(BaseState):
    """This class implements the behaviour of the state SetupRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.SETUPROUND
        self.setup_success: bool = False
        self.setup_data: dict[str, Any] = {}
        self.started: bool = False

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False

    def _initialize_state(self) -> None:
        """Initialize persistent storage for the agent."""
        self.context.logger.info("Initializing persistent storage...")

        store_path = self.context.params.store_path

        if not store_path:
            store_path = "./persistent_data"
            self.setup_data["store_path"] = store_path

        store_path = Path(store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage files
        files_to_initialize = {
            "positions.json": {"positions": [], "last_updated": None},
            "signals.json": {"signals": [], "last_signal": None},
            "performance.json": {"trades": [], "metrics": {}},
            "state.json": {"last_round": None, "error_count": 0},
        }

        for filename, default_content in files_to_initialize.items():
            file_path = store_path / filename
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(default_content, f, indent=2)

        self.context.store_path = store_path
        self.context.logger.info(f"Persistent storage initialized at: {store_path}")

        coingecko_api_key = self.context.params.coingecko_api_key
        self.context.logger.info(f"Setting CoinGecko API key: {coingecko_api_key}")

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")
        if self.started:
            return
        self.started = True

        try:
            self._initialize_state()
            self.setup_success = True
            self._event = MindshareabciappEvents.DONE

        except Exception as e:
            self.context.logger.exception(f"Setup failed. {e!s}")
            self.context.error_context = {
                "error_type": "setup_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR

        finally:
            self._is_done = True


class DataCollectionRound(BaseState):
    """This class implements the behaviour of the state DataCollectionRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.DATACOLLECTIONROUND
        self.collected_data: dict[str, Any] = {}
        self.started_at: datetime | None = None
        self.collection_initialized = False
        self.pending_tokens: list[dict[str, str]] = []
        self.completed_tokens: list[dict[str, str]] = []
        self.failed_tokens: list[dict[str, str]] = []

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False

    def _initialize_collection(self) -> None:
        """Initialize data collection on first run."""
        if self.collection_initialized:
            return

        self.context.logger.info("Initializing data collection...")

        self.pending_tokens = ALLOWED_ASSETS["base"].copy()
        self.completed_tokens = []
        self.failed_tokens = []

        self.collected_data = {
            "ohlcv": {},
            "current_prices": {},
            "market_data": {},
            "social_data": {},
            "technical_data": {},
            "fundamental_data": {},
            "onchain_data": {},
            "collection_timestamp": datetime.now(UTC).isoformat(),
            "errors": [],
            "cache_hits": 0,
            "api_calls": 0,
        }

        self.started_at = datetime.now(UTC)
        self.collection_initialized = True
        self.context.logger.info(f"Initialized batch collection for {len(self.pending_tokens)} tokens")

    def _collect_token_data(self, token_info: dict[str, str]) -> None:
        """Collect data for a single token."""
        symbol = token_info["symbol"]
        address = token_info["address"]

        self.context.logger.debug(f"Collecting data for {symbol} ({address})")

        try:
            ohlcv_data, price_data = self._fetch_coingecko_data(token_info)

            if ohlcv_data:
                self.collected_data["ohlcv"][symbol] = ohlcv_data
                ma_length = self.context.params.moving_average_length
                technical_data = self._get_technical_data(ohlcv_data, ma_length)
                self.collected_data["technical_data"][symbol] = technical_data
                self.context.logger.debug(f"Collected technical data for {symbol}: {technical_data}")
            if price_data:
                self.collected_data["current_prices"][symbol] = price_data

            self.completed_tokens.append(token_info)
            self.context.logger.debug(f"Successfully collected data for {symbol}")

        except (ValueError, KeyError, ConnectionError, TimeoutError) as e:
            self.context.logger.warning(f"Failed to collect data for {symbol}: {e}")
            self.failed_tokens.append(token_info)
            self.collected_data["errors"].append(f"Error processing {symbol} ({address}): {e}")

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")

        try:
            self._initialize_collection()

            while self.pending_tokens:
                token_info = self.pending_tokens.pop(0)
                self._collect_token_data(token_info)

            self._finalize_collection()
            self._event = MindshareabciappEvents.DONE if self._is_data_sufficient() else MindshareabciappEvents.ERROR
            self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Data collection failed: {e}")
            self.context.error_context = {
                "error_type": "data_collection_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _store_collected_data(self) -> None:
        """Store collected data to persistent storage."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available, skipping data storage")
            return

        try:
            data_file = self.context.store_path / "collected_data.json"
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(self.collected_data, f, indent=2)
            self.context.logger.info(f"Collected data stored to {data_file}")
        except Exception as e:
            self.context.logger.exception(f"Failed to store collected data: {e}")

    def _log_collection_summary(self) -> None:
        """Log collection summary."""
        completed = len(self.completed_tokens)
        failed = len(self.failed_tokens)
        total = completed + failed
        collection_time = self.collected_data.get("collection_time", 0)
        api_calls = self.collected_data.get("api_calls", 0)

        self.context.logger.info(
            f"Batch collection summary: {completed}/{total} successful, "
            f"{failed} failed, {collection_time:.1f}s, {api_calls} API calls"
        )

    def _is_data_sufficient(self) -> bool:
        """Check if collected data is sufficient for analysis."""
        min_required = max(1, len(ALLOWED_ASSETS["base"]) * self.context.params.data_sufficiency_threshold)
        successful_count = len(self.completed_tokens)
        return successful_count >= min_required

    def _finalize_collection(self) -> None:
        """Finalize data collection and store results."""
        if self.started_at:
            collection_time = (datetime.now(UTC) - self.started_at).total_seconds()
            self.collected_data["collection_time"] = collection_time

        self._store_collected_data()
        self._log_collection_summary()

    def _fetch_coingecko_data(self, token_info: dict[str, str]) -> tuple[dict | None, dict | None]:
        """Fetch OHLCV and price data from CoinGecko API.

        Args:
        ----
            token_info: Dict containing symbol, address, and coingecko_id

        Returns:
        -------
            Tuple of (ohlcv_data, price_data) or (None, None) on error

        Expected data structure:
        - price_data: Current market data with price, volume, market cap, 24h change
        - ohlcv_data: Historical OHLCV data for the last 30 days

        """
        symbol = token_info["symbol"]
        coingecko_id = token_info["coingecko_id"]

        try:
            # Fetch current price and market data for the token
            self.collected_data["api_calls"] += 1
            price_data = self._get_current_price_data(coingecko_id)

            self.context.logger.debug(f"Collected price data for {symbol}: {price_data}")

            # Fetch historical OHLCV data for the token
            self.collected_data["api_calls"] += 1
            ohlcv_data = self._get_historical_ohlcv_data(coingecko_id)

            self.context.logger.debug(f"Collected OHLCV data for {symbol}: {ohlcv_data}")

            if not ohlcv_data or not price_data:
                error_msg = "No data returned from CoinGecko API"
                raise ValueError(error_msg)

            return ohlcv_data, price_data

        except (ValueError, KeyError, ConnectionError, TimeoutError) as e:
            self.context.logger.warning(f"CoinGecko API error for {symbol}: {e}")
            return None, None

    def _get_current_price_data(self, coingecko_id: str) -> dict | None:
        """Get current price and market data for a token."""

        query_params = {
            "ids": coingecko_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        price_data = self.context.coingecko.coin_price_by_id(query_params)

        if price_data:
            return price_data[coingecko_id]

        return None

    def _get_historical_ohlcv_data(self, coingecko_id: str) -> list[list[Any]] | None:
        """Get historical OHLCV data for a token."""

        path_params = {"id": coingecko_id}
        query_params = {"vs_currency": "usd", "days": 30}

        return self.context.coingecko.get_ohlcv_data(path_params, query_params)

    # TODO: @xiuxiuxar: implement validation as part of the data collection round.
    def _validate_ohlcv_data(self, ohlcv_data: list[list[Any]]) -> None:
        """Validate OHLCV data format."""
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) == 0:
            msg = "ohlcv_data must be a non-empty list of lists."
            raise ValueError(msg)

        for row in ohlcv_data:
            if not (isinstance(row, list) and len(row) >= 6):
                msg = "Each row must be a list with at least 6 elements: timestamp, open, high, low, close, volume."
                raise ValueError(msg)


class PausedRound(BaseState):
    """This class implements the behaviour of the state PausedRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_done = False
        self.started = False
        self._state = MindshareabciappStates.PAUSEDROUND

    sleep_until: datetime | None = None

    def act(self) -> None:
        """Perform the action of the state."""

        if not self.started:
            self._is_done = False
            self.started = True
            cool_down = timedelta(seconds=self.context.params.reset_pause_duration)
            self.started_at = datetime.now(tz=UTC)
            self.sleep_until = self.started_at + cool_down
            self.context.logger.info(f"Cool down for {cool_down}s")
            return

        now = datetime.now(tz=UTC)
        if now < self.sleep_until:
            remaining = (self.sleep_until - now).total_seconds()
            self.context.logger.debug(f"Cooling down remaining: {remaining}s")
            return
        self.context.logger.info(f"Cool down finished. at {now}")
        self._is_done = True
        self._event = MindshareabciappEvents.RESUME
        self.started = False


class CheckStakingKPIRound(BaseState):
    """This class implements the behaviour of the state CheckStakingKPIRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.CHECKSTAKINGKPIROUND

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")
        self._is_done = True
        self._event = MindshareabciappEvents.DONE


class HandleErrorRound(BaseState):
    """This class implements the behaviour of the state HandleErrorRound."""

    RETRYABLE_ERRORS = {
        "ConnectionError": True,
    }

    NON_RETRYABLE_ERRORS = {
        "configuration_error": False,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.HANDLEERRORROUND
        self._retry_states = {}

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")
        self._is_done = True
        self._event = MindshareabciappEvents.DONE


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


class PortfolioValidationRound(BaseState):
    """This class implements the behaviour of the state PortfolioValidationRound."""

    supported_protocols = {
        ContractApiMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.PORTFOLIOVALIDATIONROUND
        self.validation_initialized: bool = False
        self.portfolio_metrics: dict[str, Any] = {}
        self.open_positions: list[dict[str, Any]] = []
        self.validation_result: str = ""
        self.pending_contract_calls: list[ContractApiDialogue] = []
        self.contract_responses: dict[str, Any] = {}
        self.capital_loading_complete: bool = False
        self.contract_call_submitted: bool = False

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False
        self.validation_initialized = False
        self.portfolio_metrics = {}
        self.open_positions = []
        self.validation_result = ""
        self.pending_contract_calls = []
        self.contract_responses = {}
        self.capital_loading_complete = False
        self.contract_call_submitted = False
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def act(self) -> None:
        """Perform the act using simple state-based flow."""
        try:
            # Initialize validation on first call
            if not self.validation_initialized:
                self.context.logger.info(f"Entering {self._state} state.")
                self._initialize_validation()

                if not self._load_portfolio_data():
                    self.context.logger.error("Failed to load portfolio data")
                    self._event = MindshareabciappEvents.ERROR
                    self._is_done = True
                    return

            # Submit contract call if not already submitted
            if not self.contract_call_submitted:
                self.context.logger.info("Submitting USDC balance contract call")
                self._load_available_capital_async()
                return  # Exit early, let FSM cycle for async response

            # Check for contract responses if call submitted but not complete
            if not self.capital_loading_complete:
                self._check_contract_responses()
                if not self.capital_loading_complete:
                    return  # Still waiting for response, let FSM cycle

            # Proceed with validation once capital loading is complete
            validation_checks = [
                self._check_position_limits(),
                self._check_available_capital(),
                self._check_exposure_limits(),
            ]

            can_trade = all(validation_checks)
            self._log_validation_summary()

            if can_trade:
                self.context.logger.info("Portfolio validation passed - can proceed with new trades")
                # Set available trading capital on context for use by trade construction
                self.context.available_trading_capital = self.available_trading_capital
                self.context.logger.info(
                    f"Available trading capital set to: ${self.context.available_trading_capital:.2f}"
                )
                self._event = MindshareabciappEvents.CAN_TRADE
            else:
                self.context.logger.info(f"Portfolio validation failed: {self.validation_result}")
                self._event = MindshareabciappEvents.AT_LIMIT

            self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Portfolio validation failed: {e}")
            self.context.error_context = {
                "error_type": "portfolio_validation_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _initialize_validation(self) -> None:
        """Initialize portfolio validation on first call."""
        if self.validation_initialized:
            return

        self.context.logger.info("Initializing portfolio validation...")

        self.portfolio_metrics = {
            "max_positions": self.context.params.max_positions,
            "current_positions": 0,
            "available_capital_usdc": 0.0,
            "total_portfolio_value": 0.0,
            "total_exposure": 0.0,
            "max_exposure_per_position": self.context.params.max_exposure_per_position,
            "max_total_exposure": self.context.params.max_total_exposure,
            "min_capital_buffer": self.context.params.min_capital_buffer,
        }

        self.validation_initialized = True

    def _load_portfolio_data(self) -> bool:
        """Load portfolio data from persistent storage."""
        try:
            if not self.context.store_path:
                self.context.logger.warning("No store path available")
                return False

            positions_file = self.context.store_path / "positions.json"
            if positions_file.exists():
                with open(positions_file, encoding="utf-8") as f:
                    positions_data = json.load(f)

                self.open_positions = [
                    pos for pos in positions_data.get("positions", []) if pos.get("status") == "open"
                ]

                self.portfolio_metrics["current_positions"] = len(self.open_positions)
                self.portfolio_metrics["total_portfolio_value"] = positions_data.get("total_portfolio_value", 0.0)

                total_exposure = sum(pos.get("entry_value_usdc", 0.0) for pos in self.open_positions)
                self.portfolio_metrics["total_exposure"] = total_exposure

            else:
                self.context.logger.info("No open positions found - assuming empty portfolio")
                self.open_positions = []

            return True

        except (FileNotFoundError, PermissionError, OSError) as e:
            self.context.logger.exception(f"Failed to access portfolio files: {e}")
            return False
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.context.logger.exception(f"Failed to parse portfolio data: {e}")
            return False
        except Exception as e:
            self.context.logger.exception(f"Unexpected error loading portfolio data: {e}")
            return False

    def _load_available_capital_async(self) -> None:
        """Load available capital from USDC balance in the agent's SAFE asynchronously."""
        try:
            target_chains = ["base"]
            chain = target_chains[0] if target_chains else "base"

            safe_addresses = self.context.params.safe_contract_addresses

            # Handle case where safe_addresses might be a string instead of dict
            if isinstance(safe_addresses, str):
                self.context.logger.info(f"Safe addresses is a string: {safe_addresses}")
                try:
                    safe_addresses_dict = json.loads(safe_addresses)
                    safe_address = safe_addresses_dict.get(chain)
                except json.JSONDecodeError:
                    self.context.logger.warning(f"Failed to parse safe_addresses JSON string: {safe_addresses}")
                    safe_address = None
            elif isinstance(safe_addresses, dict):
                safe_address = safe_addresses.get(chain)
            else:
                self.context.logger.warning(f"Unexpected safe_addresses type: {type(safe_addresses)}")
                safe_address = None

            if not safe_address:
                self.context.logger.warning(f"No SAFE address found for chain {chain}")
                self.portfolio_metrics["available_capital_usdc"] = 0.0
                self.capital_loading_complete = True
                return

            usdc_addresses = {
                "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            }

            usdc_address = usdc_addresses.get(chain)
            if not usdc_address:
                self.context.logger.warning(f"No USDC address found for chain {chain}")
                self.portfolio_metrics["available_capital_usdc"] = 0.0
                self.capital_loading_complete = True
                return

            # Submit async contract call for USDC balance
            self._submit_usdc_balance_request(chain, safe_address, usdc_address)
            self.contract_call_submitted = True

        except Exception as e:
            self.context.logger.exception(f"Failed to load available capital: {e}")
            self.portfolio_metrics["available_capital_usdc"] = 0.0
            self.capital_loading_complete = True

    def _submit_usdc_balance_request(self, chain: str, safe_address: str, usdc_address: str) -> None:
        """Submit a contract call request for USDC balance."""
        try:
            self.context.logger.info(f"Requesting USDC balance for {safe_address} on {chain}")

            # Submit the contract call
            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id="valory/ledger:0.19.0",
                contract_address=usdc_address,
                contract_id=str(Erc20.contract_id),
                callable="balance_of",
                ledger_id="ethereum",  # Use ethereum ledger for Base chain (L2)
                kwargs=ContractApiMessage.Kwargs({"account": safe_address, "chain_id": "base"}),
            )

            # Add validation function and metadata
            dialogue.validation_func = self._validate_usdc_balance_response
            dialogue.chain = chain
            dialogue.safe_address = safe_address
            dialogue.usdc_address = usdc_address

            request_nonce = self._get_request_nonce_from_dialogue(dialogue)
            self.context.requests.request_id_to_callback[request_nonce] = self.get_dialogue_callback_request()

            # Track the pending call
            self.pending_contract_calls.append(dialogue)

            self.context.logger.debug(f"Submitted USDC balance request for {safe_address}")

        except Exception as e:
            self.context.logger.exception(f"Failed to submit USDC balance request: {e}")
            self.portfolio_metrics["available_capital_usdc"] = 0.0
            self.capital_loading_complete = True

    def _validate_usdc_balance_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate USDC balance response message."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                if hasattr(message, "raw_transaction") and message.raw_transaction:
                    balance = message.raw_transaction.body.get("int")
                    if balance is not None:
                        # Store the response
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "balance": int(balance),
                            "chain": dialogue.chain,
                            "safe_address": dialogue.safe_address,
                        }
                        self.context.logger.info(f"Received USDC balance: {balance}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating USDC balance response: {e}")
            return False

    def _check_contract_responses(self) -> None:
        """Check if contract responses have arrived and process them."""
        try:
            # Process any received responses
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    # Process USDC balance response
                    if "balance" in response:
                        balance_raw = response["balance"]
                        usdc_balance = float(balance_raw) / (10**6)
                        self.portfolio_metrics["available_capital_usdc"] = usdc_balance
                        self.context.logger.info(f"Available USDC capital: ${usdc_balance:.2f}")
                        self.capital_loading_complete = True

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # If we still have pending calls, keep waiting (no timeout logic)
            if self.pending_contract_calls and not self.capital_loading_complete:
                return  # Keep waiting for responses

            # If no pending calls or we have responses, mark as complete
            if not self.pending_contract_calls and not self.capital_loading_complete:
                # No responses received, use fallback
                self.portfolio_metrics["available_capital_usdc"] = 1000.0
                self.capital_loading_complete = True
                self.context.logger.info("No contract responses received, using fallback capital value")

        except Exception as e:
            self.context.logger.exception(f"Error checking contract responses: {e}")
            self.portfolio_metrics["available_capital_usdc"] = 1000.0
            self.capital_loading_complete = True

    def _check_position_limits(self) -> bool:
        """Check if the number of open positions exceeds the limit."""
        max_positions = self.portfolio_metrics["max_positions"]
        current_positions = self.portfolio_metrics["current_positions"]

        if current_positions >= max_positions:
            self.validation_result = f"Max positions limit reached ({current_positions}/{max_positions})"
            self.context.logger.info(self.validation_result)
            return False

        self.context.logger.info(f"Position limit check: PASSED - {current_positions}/{max_positions} positions")
        return True

    def _check_available_capital(self) -> bool:
        """Check if there is enough available capital to open a new position."""
        available_capital = self.portfolio_metrics["available_capital_usdc"]
        min_capital_buffer = self.portfolio_metrics["min_capital_buffer"]

        if available_capital <= min_capital_buffer:
            self.validation_result = (
                f"Insufficient available capital (${available_capital:.2f} <= ${min_capital_buffer:.2f})"
            )
            self.context.logger.info(self.validation_result)
            return False

        min_position_size = self.context.params.min_position_size_usdc
        available_for_trading = available_capital - min_capital_buffer

        if available_for_trading < min_position_size:
            self.validation_result = (
                f"Insufficient available capital for trading (${available_for_trading:.2f} < ${min_position_size:.2f})"
            )
            self.context.logger.info(self.validation_result)
            return False

        self.context.logger.info(
            f"Available capital check: PASSED - {available_capital:.2f} > {min_capital_buffer:.2f}"
        )
        return True

    def _check_exposure_limits(self) -> bool:
        """Check if the total exposure exceeds the limit."""
        total_exposure = self.portfolio_metrics["total_exposure"]
        max_total_exposure = self.portfolio_metrics["max_total_exposure"]
        available_capital = self.portfolio_metrics["available_capital_usdc"]

        total_portfolio_value = total_exposure + available_capital

        if total_portfolio_value > 0:
            exposure_percentage = (total_exposure / total_portfolio_value) * 100

            if exposure_percentage >= max_total_exposure:
                self.validation_result = (
                    f"Total exposure exceeds maximum limit ({exposure_percentage:.1f}% > {max_total_exposure:.1f}%)"
                )
                self.context.logger.info(self.validation_result)
                return False

        max_exposure_per_position = self.portfolio_metrics["max_exposure_per_position"]
        max_new_position_value = (total_portfolio_value * max_exposure_per_position) / 100
        available_for_trading = available_capital - self.portfolio_metrics["min_capital_buffer"]
        if max_new_position_value > available_for_trading:
            self.context.logger.info(
                f"Position size will be limited by available capital (${available_for_trading:.2f}) "
                f"rather than exposure limit (${max_new_position_value:.2f})"
            )

        exposure_percentage = (total_exposure / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        self.context.logger.info(
            f"Exposure limit check: PASSED - {exposure_percentage:.1f}% exposure < {max_total_exposure:.1f}% limit"
        )
        return True

    def _log_validation_summary(self) -> None:
        """Log the validation summary."""
        metrics = self.portfolio_metrics

        self.context.logger.info("=== Portfolio Validation Summary ===")
        self.context.logger.info(f"Open Positions: {metrics['current_positions']}/{metrics['max_positions']}")
        self.context.logger.info(f"Available Capital: ${metrics['available_capital_usdc']:.2f}")
        self.context.logger.info(f"Total Exposure: ${metrics['total_exposure']:.2f}")

        if metrics["total_exposure"] + metrics["available_capital_usdc"] > 0:
            total_value = metrics["total_exposure"] + metrics["available_capital_usdc"]
            exposure_pct = (metrics["total_exposure"] / total_value) * 100
            self.context.logger.info(f"Portfolio Exposure: {exposure_pct:.1f}%")

        if self.open_positions:
            self.context.logger.info("Current Positions:")
            for pos in self.open_positions:
                symbol = pos.get("symbol", "Unknown")
                value = pos.get("entry_value_usdc", 0)
                pnl = pos.get("unrealized_pnl", 0)
                self.context.logger.info(f"  {symbol}: ${value:.2f} (P&L: ${pnl:.2f})")

        self.context.logger.info("====================================")

    @property
    def can_add_position(self) -> bool:
        """Check if we can add a new position based on current constraints."""
        return (
            self.portfolio_metrics["current_positions"] < self.portfolio_metrics["max_positions"]
            and self.portfolio_metrics["available_capital_usdc"] > self.portfolio_metrics["min_capital_buffer"]
        )

    @property
    def available_trading_capital(self) -> float:
        """Get the amount of capital available for new trades."""
        return max(0, self.portfolio_metrics["available_capital_usdc"] - self.portfolio_metrics["min_capital_buffer"])


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

            with open(data_file, encoding="utf-8") as f:
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

        except Exception as e:
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
            "sma_20": None,
            "ema_20": None,
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "adx": None,
            "obv": None,
            "p_technical": 0.5,
            "technical_traits": {},
            "price_above_ma20": False,
            "rsi_in_range": False,
            "macd_bullish": False,
            "adx_strong_trend": False,
            "obv_increasing": False,
        }

        try:
            ohlcv_data = self.collected_data.get("ohlcv_data", {}).get(symbol)
            if not ohlcv_data:
                self.context.logger.warning(f"No OHLCV data found for {symbol}")
                return technical_scores

            current_prices = self.collected_data.get("current_price", {}).get(symbol)
            current_price = current_prices.get("usd", 0)

            technical_indicators = self._get_technical_data(ohlcv_data)

            indicators_dict = {}
            for indicator_name, value in technical_indicators:
                indicators_dict[indicator_name] = value

            technical_scores.update({
                "sma_20": indicators_dict.get("SMA"),
                "ema_20": indicators_dict.get("EMA"),
                "rsi": indicators_dict.get("RSI"),
                "adx": indicators_dict.get("ADX"),
                "obv": indicators_dict.get("OBV"),
            })

            # Extract MACD Components
            macd_data = indicators_dict.get("MACD", {})
            if isinstance(macd_data, dict):
                technical_scores["macd"] = macd_data.get("MACD")
                technical_scores["macd_signal"] = macd_data.get("MACDs")

            p_technical, technical_traits = self._calculate_technical_probability_score(
                current_price=current_price,
                indicators_dict=indicators_dict
            )

            technical_scores["p_technical"] = p_technical
            technical_scores["technical_traits"] = technical_traits

            technical_scores.update({
                "price_above_ma20": technical_traits.get("price_above_ma20", {}).get("condition", False),
                "rsi_in_range": technical_traits.get("rsi_in_range", {}).get("condition", False),
                "macd_bullish": technical_traits.get("macd_bullish", {}).get("condition", False),
                "adx_strong_trend": technical_traits.get("adx_strong_trend", {}).get("condition", False),
                "obv_increasing": technical_traits.get("obv_increasing", {}).get("condition", False),
            })

            self.context.logger.info(f"Technical analysis for {symbol}: p_technical={p_technical:.3f}")

        except Exception as e:
            self.context.logger.warning(f"Technical analysis failed for {symbol}: {e}")

        return technical_scores

    def _calculate_social_probability_score(self, mentions: int, dominance: int) -> float:
        """Calculate the social probability score."""
        try:
            mentions_score = min(mentions / 100.0, 1.0) if mentions > 0 else 0.0
            dominance_score = min(dominance / 10.0, 1.0) if dominance > 0 else 0.0

            social_weight_mentions = 0.4
            social_weight_dominance = 0.6

            p_social = (mentions_score * social_weight_mentions + 
                       dominance_score * social_weight_dominance)

            return max(0.0, min(1.0, p_social))  # Clamp to [0, 1]

        except Exception as e:
            self.context.logger.warning(f"Failed to calculate social score: {e}")
            return 0.5

    def _calculate_technical_probability_score(
        self,
        current_price: float,
        indicators: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Calculate the technical probability score."""

        sma_20 = indicators.get("SMA", current_price)
        rsi = indicators.get("RSI", 50.0)
        adx = indicators.get("ADX", 20.0)
        obv = indicators.get("OBV", 0)

        macd_data = indicators.get("MACD", {})
        if isinstance(macd_data, dict):
            macd_line = macd_data.get("MACD", 0)
            macd_signal = macd_data.get("MACDs", 0)
        else:
            macd = macd_signal = 0

        # Calculate OBV slope (simplified)
        obv_slope = 1 if obv > 0 else -1 if obv < 0 else 0

        # Define technical analysis traits (based on notebook strategy)
        traits = {
            "price_above_ma20": {
                "condition": current_price > sma_20 if sma_20 else False,
                "weight": 0.2,
                "description": "Price is above MA20",
                "value": current_price - sma_20 if sma_20 else 0
            },
            "rsi_in_range": {
                "condition": 30 < rsi < 70,  # Use reasonable RSI range
                "weight": 0.2,
                "description": "RSI is in healthy range",
                "value": rsi
            },
            "macd_bullish_cross": {
                "condition": macd > macd_signal,
                "weight": 0.2,
                "description": "MACD is above its signal line",
                "value": macd - macd_signal
            },
            "adx_strong_trend": {
                "condition": adx > 25,  # Strong trend threshold
                "weight": 0.2,
                "description": "ADX indicates a strong trend",
                "value": adx
            },
            "obv_increasing": {
                "condition": obv_slope > 0,
                "weight": 0.2,
                "description": "OBV is increasing (positive slope)",
                "value": obv_slope
            }
        }

        # Calculate probability score
        p_technical = sum(
            trait["weight"] for trait in traits.values() 
            if trait["condition"]
        )

        return p_technical, traits

    def _combine_analysis_results(
        self,
        token_info: dict[str, str],
        social_scores: dict[str, Any],
        technical_scores: dict[str, Any]) -> dict[str, Any]:
        """Combine the analysis results."""
        symbol = token_info["symbol"]

        try:
            # Extract probability scores
            p_social = social_scores.get("p_social", 0.5)
            p_technical = technical_scores.get("p_technical", 0.5)

            # Default weights for Balanced strategy (from TDD)
            # [social, fundamental, onchain, technical] = [0.2, 0.3, 0.25, 0.25]
            # Since we don't have fundamental/onchain yet, redistribute weights
            social_weight = 0.4
            technical_weight = 0.6

            # Calculate combined probability score
            p_combined = (p_social * social_weight + p_technical * technical_weight)

            # Risk assessment (basic)
            risk_level = self._assess_risk_level(p_combined, social_scores, technical_scores)

            # Trading signal strength
            signal_strength = self._calculate_signal_strength(p_combined, social_scores, technical_scores)

            combined_analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(UTC).isoformat(),
                "p_social": round(p_social, 3),
                "p_technical": round(p_technical, 3),
                "p_combined": round(p_combined, 3),
                "risk_level": risk_level,
                "signal_strength": signal_strength,
                "trading_recommendation": self._get_trading_recommendation(p_combined),
                "confidence": self._calculate_confidence(social_scores, technical_scores),
                "analysis_quality": self._assess_analysis_quality(social_scores, technical_scores),
                "weights_used": {
                    "social_weight": social_weight,
                    "technical_weight": technical_weight,
                },
            }

            return combined_analysis

        except Exception as e:
            self.context.logger.exception(f"Failed to combine analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "p_social": 0.5,
                "p_technical": 0.5,
                "p_combined": 0.5,
            }

    
    def _assess_risk_level(self, p_combined: float, social_scores: dict[str, Any], technical_scores: dict[str, Any]) -> str:
        """Assess the risk level based on the analysis score.."""
        try:
            if p_combined > 0.7:
                return "low"
            elif p_combined > 0.5:
                return "medium"
            else:
                return "high"

        except Exception:
            # self.context.logger.warning(f"Failed to assess risk level: {e}")
            return "medium"

    def _calculate_signal_strength(self, p_combined: float, social_scores: dict, technical_scores: dict) -> str:
        """Calculate trading signal strength."""
        try:
            if p_combined > 0.75:
                return "strong_buy"
            elif p_combined > 0.6:
                return "buy"
            elif p_combined > 0.4:
                return "neutral"
            elif p_combined > 0.25:
                return "sell"
            else:
                return "strong_sell"
        except Exception:
            return "neutral"

    def _get_trading_recommendation(self, p_combined: float) -> str:
        """Get trading recommendation based on combined score."""
        try:
            if p_combined > 0.6:
                return "BUY"
            elif p_combined < 0.4:
                return "SELL"
            else:
                return "HOLD"
        except Exception:
            return "HOLD"

    def _calculate_confidence(self, social_scores: dict, technical_scores: dict) -> float:
        """Calculate confidence in the analysis."""
        try:
            # Confidence based on data availability and score consistency
            social_available = social_scores.get("social_metrics_available", False)
            technical_quality = 1.0 if technical_scores.get("rsi") is not None else 0.5

            confidence = 0.5  # Base confidence
            if social_available:
                confidence += 0.25
            confidence += technical_quality * 0.25

            return round(confidence, 3)
        except Exception:
            return 0.5

    def _assess_analysis_quality(self, social_scores: dict, technical_scores: dict) -> str:
        """Assess the quality of analysis data."""
        try:
            score = 0
            max_score = 2

            # Check social data quality
            if social_scores.get("social_metrics_available", False):
                score += 1

            # Check technical data quality
            if technical_scores.get("rsi") is not None:
                score += 1

            quality_pct = score / max_score
            if quality_pct > 0.8:
                return "high"
            elif quality_pct > 0.5:
                return "medium"
            else:
                return "low"
        except Exception:
            return "low"

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
                    strong_signals.append({
                        "symbol": symbol,
                        "p_combined": analysis.get("p_combined"),
                        "signal_strength": analysis.get("signal_strength"),
                        "recommendation": analysis.get("trading_recommendation"),
                    })

            # Sort by combined score
            strong_signals.sort(key=lambda x: x["p_combined"], reverse=True)
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
            analysis_file = self.context.store_path / "analysis_results.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(self.analysis_results, f, indent=2)

            self.context.logger.info(f"Analysis results stored to {analysis_file}")

        except Exception as e:
            self.context.logger.exception(f"Failed to store analysis results: {e}")

    def _get_technical_data(self, ohlcv_data: list[list[Any]], moving_average_length: int = 20) -> list:
        """Calculate core technical indicators for a coin using pandas-ta with validation."""
        try:
            data = self._preprocess_ohlcv_data(ohlcv_data)

            # Calculate different groups of indicators
            self._calculate_trend_indicators(data, moving_average_length)
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
        data["RSI"] = ta.rsi(data["close"], length=14)
        data["RSI"] = data["RSI"].clip(0, 100)

        # MACD
        self._process_macd_indicator(data)

        # ADX
        self._process_adx_indicator(data)

    def _process_macd_indicator(self, data: pd.DataFrame) -> None:
        """Process MACD indicator and handle different column naming conventions."""
        macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
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
        adx = ta.adx(data["high"], data["low"], data["close"], length=14)
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

    def _build_technical_result(self, data: pd.DataFrame) -> list:
        """Build the final result list from calculated indicators."""
        latest_data = data.iloc[-1]
        result = []

        # Add basic indicators
        for indicator in ["SMA", "EMA", "RSI", "ADX"]:
            if indicator in data.columns:
                result.append((indicator, latest_data[indicator]))

        # Add MACD (only if all components are available)
        if all(col in data.columns for col in ["MACD", "MACDh", "MACDs"]):
            result.append(
                ("MACD", {"MACD": latest_data["MACD"], "MACDh": latest_data["MACDh"], "MACDs": latest_data["MACDs"]})
            )

        # Add Bollinger Bands (only if all components are available)
        if all(col in data.columns for col in ["BBL", "BBM", "BBU"]):
            result.append(
                ("BB", {"Lower": latest_data["BBL"], "Middle": latest_data["BBM"], "Upper": latest_data["BBU"]})
            )

        # Add volume indicators
        for indicator in ["OBV", "CMF"]:
            if indicator in data.columns:
                result.append((indicator, latest_data[indicator]))

        return result


class SignalAggregationRound(BaseState):
    """This class implements the behaviour of the state SignalAggregationRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.SIGNALAGGREGATIONROUND

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")
        self._is_done = True
        self._event = MindshareabciappEvents.SIGNAL_GENERATED


class RiskEvaluationRound(BaseState):
    """This class implements the behaviour of the state RiskEvaluationRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.RISKEVALUATIONROUND
        self.evaluation_intitalized: bool = False
        self.trade_signal: dict[str, Any] = {}
        self.risk_assessment: dict[str, Any] = {}
        self.approved_trade_proposal: dict[str, Any] = {}
        self.open_positions: list[dict[str, Any]] = []

    def setup(self) -> None:
        """Setup the state."""
        self._is_done = False
        self.evaluation_intitalized = False
        self.trade_signal = {}
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

            evaluation_steps = [
                (self._check_trading_pair_duplication, "Failed trading pair duplication check"),
                (self._calculate_volatility_metrics, "Failed to calculate volatility metrics"),
                (self._apply_risk_vetoes, "Failed to apply risk vetoes"),
                # (self._determine_position_size, "Failed to determine position size"),
                (self._calculate_risk_levels, "Failed to calculate risk levels"),
                (self._validate_risk_parameters, "Failed to validate risk parameters"),
            ]

            for step_func, error_msg in evaluation_steps:
                if not step_func():
                    self.context.logger.error(error_msg)
                    self._event = MindshareabciappEvents.REJECTED
                    self._is_done = True
                    return

            self._create_approved_trade_proposal()
            self._store_approved_trade_proposal()

            self.context.logger.info("Risk evaluation passed - trade approved")
            self._event = MindshareabciappEvents.APPROVED
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

    def _initialize_risk_evaluation(self) -> None:
        """Initialize the risk evaluation."""
        self.evaluation_intitalized = True

        self.risk_assessment = {
            "volatility_atr": 0.0,
            "volatility_tier": "unknown",
            "onchain_risk_score": 0.0,
            "risk_vetoes": [],
            "position_size_usdc": 0.0,
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
                with open(positions_file, encoding="utf-8") as f:
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
            elif "/" not in pos_symbol and pos_symbol:
                # Assume USDC quote for single symbols
                if pos_symbol == target_symbol:
                    existing_positions.append(pos)

        if existing_positions:
            self.context.logger.warning(
                f"Trading pair duplication check: FAILED - Already have open position(s) in {target_symbol}: {len(existing_positions)} position(s)"
            )

            # Log details of existing positions
            for i, pos in enumerate(existing_positions):
                entry_value = pos.get("entry_value_usdc", 0)
                pnl = pos.get("unrealized_pnl", 0)
                side = pos.get("side", "unknown")
                self.context.logger.info(
                    f"  Existing position {i+1}: {side} ${entry_value:.2f} (P&L: ${pnl:.2f})"
                )

            return False

        self.context.logger.info(f"Trading pair duplication check: PASSED - no existing {target_symbol} positions")
        return True

    def _get_trading_pair_from_position(self, position: dict) -> str:
        """Extract standardized trading pair identifier from position."""
        symbol = position.get("symbol", "").upper()

        if "/" in symbol:
            return symbol  # Already in pair format like "OLAS/USDC"
        if symbol:
            return f"{symbol}/USDC"  # Assume USDC quote
        return "UNKNOWN/USDC"

    def _check_conflicting_positions(self, target_symbol: str) -> list[dict]:
        """Get list of positions that would conflict with the target trading pair."""
        conflicting_positions = []
        target_pair = f"{target_symbol.upper()}/USDC"

        for pos in self.open_positions:
            pos_pair = self._get_trading_pair_from_position(pos)
            if pos_pair == target_pair:
                conflicting_positions.append(pos)

        return conflicting_positions

    def _load_trade_signal(self) -> bool:
        """Load the trade signal from the previous round."""
        try:
            if hasattr(self.context, "aggregated_trade_signal") and self.context.aggregated_trade_signal:
                self.trade_signal = self.context.aggregated_trade_signal
                self.context.logger.info(
                    f"Loaded aggregated trade signal for {self.trade_signal.get('symbol', 'Unknown')}"
                )
                return True

            if not self.context.store_path:
                self.context.logger.warning("No store path available for loading trade signal.")
                return False

            signals_file = self.context.store_path / "signals.json"
            if not signals_file.exists():
                self.context.logger.warning("No signals file found")
                return False

            with open(signals_file, encoding="utf-8") as f:
                signals_data = json.load(f)

            latest_signal = signals_data.get("last_signal")

            if not latest_signal or latest_signal.get("status") != "generated":
                self.context.logger.warning("No valid generated signal found")
                return False

            self.trade_signal = latest_signal
            self.context.logger.info(f"Loaded aggregated trade signal for {self.trade_signal.get('symbol', 'Unknown')}")
            return True

        except Exception as e:
            self.context.logger.exception(f"Error loading trade signal: {e}")
            return False

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
                vetoes.append({
                    "type": "volatility",
                    "reason": "Volatility tier is very high - too risky",
                    "severity": "medium"
                })

            p_trade = self.trade_signal.get("p_trade", 0.0)
            min_signal_threshold = 0.6  # Minimum 60% confidence
            if p_trade < min_signal_threshold:
                vetoes.append({
                    "type": "signal_strength",
                    "reason": f"Trade signal too weak: {p_trade:.3f} < {min_signal_threshold}",
                    "severity": "high"
                })

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

    def _determine_position_size(self) -> bool:
        """Determine the position size based on risk assessment."""
        try:
            available_capital = getattr(self.context, "available_trading_capital", None)
            if available_capital is None:
                self.context.logger.warning("No available trading capital found, using fallback")
                available_capital = 1000.0

            strategy = self.context.params.trading_stategy
            volatility_tier = self.risk_assessment.get("volatility_tier")
            signal_strength = self.trade_signal.get("p_trade", 0.5)

            base_position_pct = self._get_base_position_percentage(strategy)

            volatility_multiplier = self._get_volatility_multiplier(volatility_tier)

            signal_multiplier = min(signal_strength * 2.0, 1.5)

            final_position_pct = base_position_pct * volatility_multiplier * signal_multiplier

            min_position_pct = 0.02
            max_position_pct = 0.20
            final_position_pct = max(min_position_pct, min(final_position_pct, max_position_pct))

            position_size_usdc = available_capital * final_position_pct

            min_position_size = self.context.params.min_position_size_usdc
            max_position_size = self.context.params.max_position_size_usdc
            position_size_usdc = max(min_position_size, min(position_size_usdc, max_position_size))

            self.risk_assessment["position_size_usdc"] = position_size_usdc
            self.risk_assessment["position_percentage"] = (position_size_usdc / available_capital) * 100

            self.context.logger.info(
                f"Position sizing: {position_size_usdc:.2f} USDC "
                f"({(position_size_usdc / available_capital) * 100:.1f}% of capital)"
            )

            return True

        except Exception as e:
            self.context.logger.exception(f"Failed to determine position size: {e}")
            return False

    def _calculate_risk_levels(self) -> bool:
        """Calculate the risk levels based on position size and volatility."""
        try:
            symbol = self.trade_signal.get("symbol")
            current_price = self._get_current_price(symbol)
            if not current_price or current_price <= 0.0:
                self.context.logger.warning(f"No valid current price found for {symbol}")
                return False

            stop_loss_pct = 0.84
            tailing_stop_loss_pct = 0.97
            trailing_activation = 1.25

            stop_loss_price = current_price * stop_loss_pct

            risk_amount = current_price - stop_loss_price
            take_profit_price = current_price + (risk_amount * 2)

            self.risk_assessment.update({
                "current_price": current_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "tailing_stop_loss_pct": tailing_stop_loss_pct,
                "trailing_activation_level": trailing_activation,
            })

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
            # # Check position size is reasonable
            # position_size = self.risk_assessment["position_size_usdc"]
            # if position_size <= 0:
            #     self.context.logger.error("Position size is zero or negative")
            #     return False

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
            # "position_size_usdc": self.risk_assessment["position_size_usdc"],
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
                    with open(signals_file, encoding="utf-8") as f:
                        signals_data = json.load(f)

                # Update the last signal with approval
                signals_data["last_signal"] = self.approved_trade_proposal
                signals_data["last_updated"] = datetime.now(UTC).isoformat()

                # Save updated signals
                with open(signals_file, "w", encoding="utf-8") as f:
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

            with open(data_file, encoding="utf-8") as f:
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

            with open(data_file, encoding="utf-8") as f:
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

    def _get_base_position_percentage(self, strategy: str) -> float:
        """Get base position percentage based on strategy."""
        strategy_percentages = {
            "aggressive": 0.15,  # 15% of capital
            "balanced": 0.10,    # 10% of capital
            "conservative": 0.05  # 5% of capital
        }
        return strategy_percentages.get(strategy, 0.10)

    def _get_volatility_multiplier(self, volatility_tier: str) -> float:
        """Get position size multiplier based on volatility."""
        multipliers = {
            "low": 1.2,        # Larger positions for low volatility
            "medium": 1.0,     # Base size for medium volatility
            "high": 0.7,       # Smaller positions for high volatility
            "very_high": 0.5,  # Much smaller positions for very high volatility
        }
        return multipliers.get(volatility_tier, 0.8)


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

            with open(signals_file, encoding="utf-8") as f:
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
            params = []

            def encode_dict(d: dict) -> bytes:
                """Encode a dictionary to a hex string."""
                return json.dumps(d).encode(DEFAULT_ENCODING)

            # Use a placeholder amount for ticker requests since we need price to calculate position size
            ticker_amount = 1.0  # Use 1 USDC as standard amount for price discovery

            params.append({"symbol": trading_pair, "params": encode_dict({"amount": ticker_amount})})
            self.context.logger.debug(f"Ticker request params: symbol={trading_pair}, amount={ticker_amount}")

            for param in params:
                ticker_dialogue = self.submit_msg(
                    TickersMessage.Performative.GET_TICKER,
                    connection_id=str(DCXT_PUBLIC_ID),
                    exchange_id="balancer",
                    ledger_id="base",
                    **param,
                )

            ticker_dialogue.validation_func = self._validate_ticker_msg
            ticker_dialogue.exchange_id = "balancer"
            ticker_dialogue.ledger_id = "base"
            ticker_dialogue.symbol = symbol
            ticker_dialogue.trading_pair = trading_pair

            price_request = PriceRequest(
                symbol=symbol,
                exchange_id="balancer",
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
                return False

            slippage_tolerance = self._calculate_slippage_tolerance()
            stop_loss_price = self._calculate_stop_loss_price(current_price, trade_direction)
            take_profit_price = self._calculate_take_profit_price(current_price, trade_direction, stop_loss_price)

            token_quantity = position_size_usdc / current_price

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

    def _calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk parameters."""
        try:
            # Get available trading capital from portfolio validation
            # Check if we have portfolio metrics from previous validation
            available_capital = getattr(self.context, "available_trading_capital", None)
            if available_capital is None:
                # Fallback to a smaller default amount for safety
                available_capital = 100.0  # Much smaller default to avoid insufficient balance
                self.context.logger.warning(f"No available trading capital found, using fallback: ${available_capital}")
            else:
                self.context.logger.info(
                    f"Using available trading capital from portfolio validation: ${available_capital:.2f}"
                )

            base_position_pct = 0.10  # 10% base allocation

            # Adjust based on signal strength (0.5 = neutral, 1.0 = very bullish, 0.0 = very bearish)
            strength_multiplier = signal_strength if signal_strength > 0.5 else (1 - signal_strength)
            position_pct = base_position_pct * (0.5 + strength_multiplier)  # Range: 5% - 15%

            # Calculate position size
            position_size = available_capital * position_pct

            # Apply min/max constraints
            min_size = self.risk_parameters["min_position_size"]
            max_size = min(self.risk_parameters["max_position_size"], available_capital * 0.20)  # Max 20% of capital

            position_size = max(min_size, min(position_size, max_size))

            self.context.logger.info(f"Calculated position size: ${position_size:.2f} (signal: {signal_strength:.2f})")
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
                    with open(trades_file, encoding="utf-8") as f:
                        pending_trades = json.load(f)

                # Add new trade
                pending_trades["trades"].append(self.constructed_trade)
                pending_trades["last_updated"] = datetime.now(UTC).isoformat()

                # Save updated trades
                with open(trades_file, "w", encoding="utf-8") as f:
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


class ExecutionRound(BaseState):
    """This class implements the behaviour of the state ExecutionRound."""

    supported_protocols = {
        OrdersMessage.protocol_id: [],
        ContractApiMessage.protocol_id: [],
        LedgerApiMessage.protocol_id: [],
        SigningMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.EXECUTIONROUND

        # Order management
        self.pending_orders: list[Order] = []
        self.submitted_orders: list[Order] = []
        self.completed_orders: list[Order] = []
        self.failed_orders: list[Order] = []

        # Execution context
        self.execution_type: str = ""  # "entry" or "exit"
        self.execution_initialized: bool = False
        self.execution_started_at: datetime | None = None

        # Multisend operation tracking
        self.active_operation: dict[str, Any] | None = None
        self.operation_queue: list[dict[str, Any]] = []

        # Transaction tracking
        self.pending_dialogues: dict[str, str] = {}
        self.dialogue_responses: dict[str, dict] = {}

    def setup(self) -> None:
        """Setup the execution round."""
        super().setup()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all state variables for a fresh execution round."""
        self._is_done = False
        self.execution_initialized = False
        self.execution_type = ""
        self.execution_started_at = None

        self.pending_orders = []
        self.submitted_orders = []
        self.completed_orders = []
        self.failed_orders = []

        self.active_operation = None
        self.operation_queue = []
        self.pending_dialogues = {}
        self.dialogue_responses = {}

        for protocol in self.supported_protocols:
            self.supported_protocols[protocol] = []

    def act(self) -> None:
        """Perform the act."""
        try:
            # Initialize if needed
            if not self.execution_initialized:
                self._initialize_execution()
                if self._is_done:
                    return

            # Process any incoming messages
            if self._has_pending_responses():
                self._process_responses()
                return

            # Submit new orders if available
            if self.pending_orders and not self.active_operation:
                self._start_next_order()
                return

            # Continue active operation
            if self.active_operation:
                self._continue_operation()
                return

            # Check timeout
            if self._is_timeout():
                self._handle_timeout()
                return

            # Check completion
            if self._is_complete():
                self._finalize()
                return

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            self._handle_error(e)

    def _initialize_execution(self) -> None:
        """Initialize execution round by determining what to execute."""
        self.context.logger.info(f"Entering {self._state} state.")

        # Check if we have positions to exit (priority over new trades)
        if hasattr(self.context, "positions_to_exit") and self.context.positions_to_exit:
            self._setup_exit_execution()
        elif hasattr(self.context, "constructed_trade") and self.context.constructed_trade:
            self._setup_entry_execution()
        else:
            self.context.logger.warning("No trades to execute")
            self._complete(MindshareabciappEvents.EXECUTED)
            return

        self.execution_initialized = True
        self.execution_started_at = datetime.now(UTC)
        self.context.logger.info(
            f"Initialized {self.execution_type} execution with {len(self.pending_orders)} order(s)"
        )

    def _setup_exit_execution(self) -> None:
        """Prepare exit orders for open positions."""
        self.execution_type = "exit"

        for position in self.context.positions_to_exit:
            order = self._create_exit_order(position)
            if order:
                self.pending_orders.append(order)

    def _setup_entry_execution(self) -> None:
        """Prepare entry orders for new position."""
        self.execution_type = "entry"

        order = self._create_entry_order(self.context.constructed_trade)
        if order:
            self.pending_orders.append(order)

    # =========== ORDER CREATION ===========

    def _create_exit_order(self, position: dict[str, Any]) -> Order | None:
        """Create an exit order from position data."""
        symbol = position.get("symbol")
        quantity = position.get("token_quantity", 0)

        if quantity <= 0:
            self.context.logger.warning(f"Invalid token quantity for {symbol}: {quantity}")
            return None

        order = Order(
            id=f"exit_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            symbol=f"{symbol}/USDC",
            side=OrderSide.SELL,
            amount=quantity,  # Amount in token units (human readable)
            price=position.get("exit_price", position.get("current_price", 0)),
            type=OrderType.MARKET,
            exchange_id="balancer",
            ledger_id="base",
            status=OrderStatus.NEW,
            timestamp=datetime.now(UTC).timestamp(),
        )

        # Add metadata
        order.position_id = position.get("position_id")
        order.exit_reason = position.get("exit_reason", "unknown")

        return order

    def _create_entry_order(self, trade: dict[str, Any]) -> Order | None:
        """Create an entry order from trade construction."""
        symbol = trade.get("symbol")
        quantity = truncate_to_decimals(trade.get("token_quantity"))

        order = Order(
            id=trade.get("trade_id"),
            symbol=f"{symbol}/USDC",
            asset_a=trade.get("buy_token"),
            asset_b=trade.get("sell_token"),
            side=OrderSide.BUY,
            amount=quantity,  # Amount in BUY tokens
            price=trade.get("entry_price"),
            type=OrderType.MARKET,
            exchange_id="balancer",
            ledger_id="base",  # Use "base" for Base chain
            status=OrderStatus.NEW,
            timestamp=datetime.now(UTC).timestamp(),
        )

        # Attach risk parameters
        order.stop_loss_price = trade.get("stop_loss_price")
        order.take_profit_price = trade.get("take_profit_price")

        return order

    # =========== ORDER PROCESSING ===========

    def _start_next_order(self) -> None:
        """Begin processing the next pending order."""
        if not self.pending_orders:
            return

        order = self.pending_orders.pop(0)
        self.submitted_orders.append(order)

        self.context.logger.info(f"Processing order: {order.id} - {order.side} {order.amount} {order.symbol}")

        # Create multisend operation
        self.active_operation = self._create_operation(order)
        self._continue_operation()

    def _create_operation(self, order: Order) -> dict[str, Any]:
        """Create a new multisend operation."""
        return {
            "order": order,
            "state": "approve_pending",
            "approve_data": None,
            "swap_data": None,
            "multisend_data": None,
            "safe_hash": None,
            "created_at": datetime.now(UTC),
        }

    def _continue_operation(self) -> None:
        """Continue processing the active multisendoperation."""
        if not self.active_operation:
            return

        state = self.active_operation["state"]

        if state == "approve_pending":
            self._request_approve_data()
        elif state == "swap_pending":
            self._request_swap_data()
        elif state == "multisend_pending":
            self._build_multisend()
        elif state == "safe_hash_pending":
            self._request_safe_hash()
        elif state == "execution_pending":
            self._execute_safe_tx()
        elif state == "signing_pending":
            self._sign_transaction()
        elif state == "broadcast_pending":
            self._broadcast_transaction()
        elif state == "receipt_pending":
            self._request_receipt(self.active_operation["tx_digest"])

    # =========== ERC20 approval ===========

    def _request_approve_data(self) -> None:
        """Request ERC20 approve call data."""

        if any(dialogue_type == "approve" for dialogue_type in self.pending_dialogues.values()):
            self.context.logger.info("Approval dialogue already in progress, skipping approval request")
            return

        order = self.active_operation["order"]
        amount = self._calculate_approval_amount(order)

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id="valory/ledger:0.19.0",
            contract_address=order.asset_b,
            contract_id=str(ERC20.contract_id),
            ledger_id="ethereum",
            callable="build_approval_tx",
            kwargs=ContractApiMessage.Kwargs(
                {
                    "spender": BALANCER_VAULT,
                    "amount": amount,
                }
            ),
        )

        dialogue.validation_func = self._validate_approve_response
        self._track_dialogue(dialogue, "approve")

        self.context.logger.info(f"Requested ERC20 approval for {order.asset_b}")

    def _validate_approve_response(self, message: Message, dialogue: BaseDialogue) -> bool:
        """Process ERC20 approval response."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                data = message.raw_transaction.body.get("data")
                if data:
                    self.active_operation["approve_data"] = "0x" + data.hex()
                    self.active_operation["state"] = "swap_pending"
                    self._clear_dialogue(dialogue)

                    self._auto_continue()
                    return True

            self.context.logger.error(f"Invalid approve response: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing approve response: {e}")
            return False

    # =========== DCXT Swap ===========

    def _request_swap_data(self) -> None:
        """Request DCXT swap call data."""
        order = self.active_operation["order"]

        safe_address = self._get_safe_address()
        if safe_address:
            order.info = json.dumps({"safe_contract_address": safe_address})

        dialogue = self.submit_msg(
            performative=OrdersMessage.Performative.CREATE_ORDER,
            connection_id=str(DCXT_PUBLIC_ID),
            order=order,
            exchange_id="balancer",
            ledger_id="base",
        )

        dialogue.validation_func = self._validate_swap_response
        self._track_dialogue(dialogue, "swap")

        self.context.logger.info(f"Requested DCXT swap data for {order.id}")

    def _validate_swap_response(self, message: OrdersMessage, dialogue: BaseDialogue) -> bool:
        """Process DCXT swap response."""
        try:
            if message.performative in {OrdersMessage.Performative.ORDER, OrdersMessage.Performative.ORDER_CREATED}:
                order = message.order
                if order and hasattr(order, "info"):
                    info = json.loads(order.info)
                    swap_data = info.get("data")
                    if swap_data:
                        self.active_operation["swap_data"] = swap_data
                        self.active_operation["state"] = "multisend_pending"
                        self._clear_dialogue(dialogue)

                        self._auto_continue()
                        return True

            self.context.logger.error(f"Invalid swap response: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing swap response: {e}")
            return False

    # =========== Multisend construction ===========

    def _build_multisend(self) -> None:
        """Build multisend transaction data."""
        approve_data = self.active_operation["approve_data"]
        swap_data = self.active_operation["swap_data"]
        order = self.active_operation["order"]

        if not approve_data.startswith("0x"):
            approve_data = "0x" + approve_data
        if not swap_data.startswith("0x"):
            swap_data = "0x" + swap_data

        payload = [
            {"operation": MultiSendOperation.CALL, "to": order.asset_b, "value": 0, "data": approve_data},
            {"operation": MultiSendOperation.CALL, "to": BALANCER_VAULT, "value": 0, "data": swap_data},
        ]

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id="valory/ledger:0.19.0",
            contract_address=MULTISEND_ADDRESS,
            contract_id=str(MultiSendContract.contract_id),
            ledger_id="ethereum",
            callable="get_tx_data",
            kwargs=ContractApiMessage.Kwargs({"multi_send_txs": payload}),
        )

        dialogue.validation_func = self._validate_multisend_response
        self._track_dialogue(dialogue, "multisend")

        self.context.logger.info("Building multisend transaction")

    def _validate_multisend_response(self, message: ContractApiMessage, dialogue: BaseDialogue) -> bool:
        """Process multisend data response."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                self.active_operation["multisend_data"] = message.raw_transaction
                self.active_operation["state"] = "safe_hash_pending"
                self._clear_dialogue(dialogue)

                self._auto_continue()
                return True

            self.context.logger.error(f"Invalid multisend response: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing multisend response: {e}")
            return False

    # =========== Safe transaction ===========

    def _request_safe_hash(self) -> None:
        """Request Safe transaction hash."""
        multisend_data = self.active_operation["multisend_data"]
        safe_address = self._get_safe_address()

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id="valory/ledger:0.19.0",
            contract_address=safe_address,
            contract_id=str(GnosisSafeContract.contract_id),
            callable="get_raw_safe_transaction_hash",
            ledger_id="ethereum",
            kwargs=ContractApiMessage.Kwargs(
                {
                    "to_address": MULTISEND_ADDRESS,
                    "value": 0,
                    "data": bytes.fromhex(multisend_data.body["data"][2:]),
                    "operation": SafeOperation.DELEGATE_CALL.value,
                    "safe_tx_gas": 0,
                    "gas_price": 0,
                }
            ),
        )

        dialogue.validation_func = self._validate_safe_hash_response
        dialogue.original_data = multisend_data.body["data"]
        self._track_dialogue(dialogue, "safe_hash")

        self.context.logger.info("Requesting Safe transaction hash")

    def _validate_safe_hash_response(self, message: ContractApiMessage, dialogue: BaseDialogue) -> bool:
        """Process Safe hash response."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                self.active_operation["safe_hash"] = message.raw_transaction
                self.active_operation["state"] = "execution_pending"
                self._clear_dialogue(dialogue)

                self.active_operation["original_data"] = dialogue.original_data

                self._auto_continue()
                return True

            self.context.logger.error(f"Invalid safe hash response: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing safe hash response: {e}")
            return False

    def _execute_safe_tx(self) -> None:
        """Execute Safe transaction with pre-approved signature."""
        safe_address = self._get_safe_address()
        call_data = self.active_operation["original_data"]

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id="valory/ledger:0.19.0",
            contract_address=safe_address,
            contract_id=str(GnosisSafeContract.contract_id),
            callable="get_raw_safe_transaction",
            ledger_id="ethereum",
            kwargs=ContractApiMessage.Kwargs(
                {
                    "sender_address": self.context.agent_address,
                    "owners": (self.context.agent_address,),
                    "to_address": MULTISEND_ADDRESS,
                    "value": 0,
                    "data": bytes.fromhex(call_data.removeprefix("0x")),
                    "signatures_by_owner": {self.context.agent_address: self._get_preapproved_signature()},
                    "operation": SafeOperation.DELEGATE_CALL.value,
                    "safe_tx_gas": 0,
                    "base_gas": 0,
                    "gas_price": 0,
                    "gas_token": NULL_ADDRESS,
                    "refund_receiver": NULL_ADDRESS,
                }
            ),
        )

        # Set up validation for the execution response
        dialogue.validation_func = self._validate_execution_response
        self._track_dialogue(dialogue, "execution")

        self.context.logger.info("Executing Safe transaction")

    def _validate_execution_response(self, message: ContractApiMessage, dialogue: BaseDialogue) -> bool:
        """Process Safe execution response."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                self.active_operation["safe_hash"] = message.raw_transaction
                self.active_operation["state"] = "signing_pending"
                self._clear_dialogue(dialogue)

                self._auto_continue()
                return True

            self.context.logger.error(f"Invalid execution response: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing execution response: {e}")
            return False

    # =========== Transaction signing ===========

    def _sign_transaction(self) -> None:
        """Sign the Safe execution transaction."""
        raw_tx = self.active_operation["safe_hash"]

        terms = Terms(
            ledger_id="ethereum",
            sender_address=self.context.agent_address,
            counterparty_address="",
            amount_by_currency_id={},
            quantities_by_good_id={},
            is_sender_payable_tx_fee=True,
            nonce="",
            fee_by_currency_id={},
        )

        # Create signing dialogue
        signing_msg, signing_dialogue = self.context.signing_dialogues.create(
            counterparty=self.context.decision_maker_address,
            performative=SigningMessage.Performative.SIGN_TRANSACTION,
            raw_transaction=raw_tx,
            terms=terms,
        )

        # Set validation
        signing_dialogue.validation_func = self._validate_signing_response

        # Register callback
        request_nonce = signing_dialogue.dialogue_label.dialogue_reference[0]
        self.context.requests.request_id_to_callback[request_nonce] = self.get_dialogue_callback_request()

        # Send to decision maker
        self.context.decision_maker_message_queue.put_nowait(signing_msg)

        self.active_operation["state"] = "signing_pending"
        self.context.logger.info("Transaction sent for signing")

    def _validate_signing_response(self, message: SigningMessage, _dialogue: BaseDialogue) -> bool:
        """Process signing response."""
        try:
            if message.performative == SigningMessage.Performative.SIGNED_TRANSACTION:
                self.active_operation["signed_tx"] = message.signed_transaction
                self.active_operation["state"] = "broadcast_pending"

                self._auto_continue()
                return True

            self.context.logger.error(f"Signing failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing signing response: {e}")
            return False

    # =========== Transaction broadcasting ===========

    def _broadcast_transaction(self) -> None:
        """Broadcast the signed transaction."""
        signed_tx = self.active_operation["signed_tx"]

        dialogue = self.submit_msg(
            performative=LedgerApiMessage.Performative.SEND_SIGNED_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            signed_transaction=signed_tx,
            kwargs=LedgerApiMessage.Kwargs({"ledger_id": "ethereum"}),
        )

        dialogue.validation_func = self._validate_broadcast_response
        self._track_dialogue(dialogue, "broadcast")

        self.context.logger.info("Broadcasting transaction to chain")

    def _validate_broadcast_response(self, message: LedgerApiMessage, dialogue: BaseDialogue) -> bool:
        """Process broadcast response."""
        try:
            if message.performative == LedgerApiMessage.Performative.TRANSACTION_DIGEST:
                tx_hash = message.transaction_digest.body
                self.active_operation["tx_hash"] = tx_hash
                self.active_operation["tx_digest"] = message.transaction_digest
                self.active_operation["state"] = "receipt_pending"

                self.context.logger.info(f"Transaction broadcast successful: {tx_hash}")
                self._clear_dialogue(dialogue)

                self._auto_continue()
                return True

            self.context.logger.error(f"Broadcast failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing broadcast response: {e}")
            return False

    # =========== Receipt verification ===========

    def _request_receipt(self, tx_digest: TransactionDigest) -> None:
        """Request transaction receipt."""
        dialogue = self.submit_msg(
            performative=LedgerApiMessage.Performative.GET_TRANSACTION_RECEIPT,
            connection_id=LEDGER_API_ADDRESS,
            transaction_digest=tx_digest,
        )

        dialogue.validation_func = self._validate_receipt_response
        self._track_dialogue(dialogue, "receipt")

        self.context.logger.info("Requesting transaction receipt")

    def _validate_receipt_response(self, message: LedgerApiMessage, dialogue: BaseDialogue) -> bool:
        """Process receipt response."""
        try:
            if message.performative == LedgerApiMessage.Performative.TRANSACTION_RECEIPT:
                receipt = message.transaction_receipt.receipt

                if receipt.get("status") == 1:
                    self.context.logger.info("✅ Transaction successful on chain!")
                    self._clear_dialogue(dialogue)
                    self._complete_operation()
                    return True
                self.context.logger.error(f"Transaction failed on chain: {receipt}")
                self._fail_operation("Transaction reverted on chain")
                return False

        except Exception as e:
            self.context.logger.exception(f"Error processing receipt: {e}")
            return False

    # =========== Operation completion ===========

    def _complete_operation(self) -> None:
        """Mark operation as complete and update positions."""
        order = self.active_operation["order"]
        order.status = OrderStatus.FILLED
        order.filled = order.amount

        # Move from submitted to completed
        if order in self.submitted_orders:
            self.submitted_orders.remove(order)
        self.completed_orders.append(order)

        # Update position tracking
        if self.execution_type == "exit":
            self._finalize_exit(order)
        else:
            self._create_position(order)

        self.context.logger.info(f"Order {order.id} completed successfully")

        # Clear operation
        self.active_operation = None

        # Check if more orders to process
        if not self.pending_orders and not self.submitted_orders:
            self._finalize()

    def _fail_operation(self, reason: str) -> None:
        """Mark operation as failed."""
        order = self.active_operation["order"]
        order.status = OrderStatus.FAILED
        order.info = reason

        if order in self.submitted_orders:
            self.submitted_orders.remove(order)
        self.failed_orders.append(order)

        self.context.logger.error(f"Order {order.id} failed: {reason}")

        # CLear operation
        self.active_operation = None

        # Check completion
        if not self.pending_orders and not self.submitted_orders:
            self._finalize()

    # =========== Position management ===========

    def _finalize_exit(self, order: Order, partial_fill: bool = False) -> None:
        """Update position after exit."""
        try:
            # Find the original position
            position_id = getattr(order, "position_id", None)
            if not position_id:
                self.context.logger.warning(f"No position_id found for exit order {order.id}")
                return

            original_position = self._find_position_by_id(position_id)
            if not original_position:
                self.context.logger.warning(f"Original position not found for ID: {position_id}")
                return

            # Calculate execution details
            executed_price = order.price or order.average_price or 0
            executed_quantity = order.filled or order.amount
            entry_price = original_position.get("entry_price", 0)

            # Calculate P&L
            realized_pnl = (executed_price - entry_price) * executed_quantity
            realized_pnl_percentage = ((executed_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

            # Create closed position record
            closed_position = {
                **original_position,
                "status": "closed" if not partial_fill else "partially_closed",
                "exit_price": executed_price,
                "exit_time": datetime.fromtimestamp(order.timestamp, UTC).isoformat(),
                "exit_reason": getattr(order, "exit_reason", "manual"),
                "executed_quantity": executed_quantity,
                "realized_pnl": realized_pnl,
                "realized_pnl_percentage": realized_pnl_percentage,
                "order_id": order.id,
                "closed_at": datetime.now(UTC).isoformat(),
                "partial_fill": partial_fill,
            }

            # Update positions storage
            self._update_position_in_storage(closed_position)

            self.context.logger.info(
                f"{'Partially ' if partial_fill else ''}Closed position {original_position.get('symbol')} with "
                f"P&L: ${realized_pnl:.2f} ({realized_pnl_percentage:.2f}%)"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to finalize position exit: {e}")

    def _create_position(self, order: Order, partial_fill: bool = False) -> None:
        try:
            # Extract symbol from order symbol (e.g., "OLAS/USDC" -> "OLAS")
            symbol = order.symbol.split("/")[0] if "/" in order.symbol else order.symbol

            # Calculate execution details
            executed_price = order.price or order.average_price or 0
            executed_usdc_amount = order.filled or order.amount

            # For buy orders, calculate token quantity from USDC amount and price
            if order.side == "buy":
                token_quantity = executed_usdc_amount / executed_price if executed_price > 0 else 0
                position_size_usdc = executed_usdc_amount
            else:
                # For sell orders (shouldn't happen in entry, but handle anyway)
                token_quantity = executed_usdc_amount
                position_size_usdc = executed_usdc_amount * executed_price

            # Create new position record
            position_id = f"pos_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

            new_position = {
                "position_id": position_id,
                "symbol": symbol,
                "contract_address": getattr(order, "contract_address", ""),
                "direction": "long",  # Assuming all entries are long positions for now
                "status": "open",
                "entry_price": executed_price,
                "entry_time": datetime.fromtimestamp(order.timestamp, UTC).isoformat(),
                "token_quantity": token_quantity,
                "position_size_usdc": position_size_usdc,
                "stop_loss_price": getattr(order, "stop_loss_price", 0),
                "take_profit_price": getattr(order, "take_profit_price", 0),
                "current_price": executed_price,
                "unrealized_pnl": 0.0,
                "pnl_percentage": 0.0,
                "order_id": order.id,
                "created_at": datetime.now(UTC).isoformat(),
                "last_updated": datetime.now(UTC).isoformat(),
                "trailing_stop_enabled": False,
                "trailing_distance": 0.05,
                "partial_fill": partial_fill,
            }

            # Add position to storage
            self._add_position_to_storage(new_position)

            self.context.logger.info(
                f"Created {'partial ' if partial_fill else ''}new position {symbol} "
                f"at ${executed_price:.6f}, quantity: {token_quantity:.6f}"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to create new position: {e}")

    def _add_position_to_storage(self, position: dict[str, Any]) -> None:
        """Add a new position to persistent storage."""
        if not self.context.store_path:
            return

        try:
            positions_file = self.context.store_path / "positions.json"

            # Load existing data
            existing_data = {"positions": []}
            if positions_file.exists():
                with open(positions_file, encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Add new position
            positions = existing_data.get("positions", [])
            positions.append(position)

            # Calculate summary statistics
            open_positions = [pos for pos in positions if pos.get("status") == "open"]
            total_portfolio_value = sum(pos.get("position_size_usdc", 0) for pos in open_positions)

            # Save updated data
            updated_data = {
                "positions": positions,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_positions": len(positions),
                "open_positions": len(open_positions),
                "total_portfolio_value": round(total_portfolio_value, 2),
            }

            with open(positions_file, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, indent=2)

            self.context.logger.info(f"Added new position {position['position_id']} to storage")

        except Exception as e:
            self.context.logger.exception(f"Failed to add position to storage: {e}")

    def _update_position_in_storage(self, position: dict[str, Any]) -> None:
        """Update a position in persistent storage."""
        if not self.context.store_path:
            return

        try:
            positions_file = self.context.store_path / "positions.json"

            # Load existing data
            existing_data = {"positions": []}
            if positions_file.exists():
                with open(positions_file, encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Update the specific position
            positions = existing_data.get("positions", [])
            position_id = position["position_id"]

            # Find and update existing position
            updated = False
            for i, pos in enumerate(positions):
                if pos.get("position_id") == position_id:
                    positions[i] = position
                    updated = True
                    break

            if not updated:
                positions.append(position)

            # Save updated data
            existing_data["positions"] = positions
            existing_data["last_updated"] = datetime.now(UTC).isoformat()

            with open(positions_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)

            self.context.logger.info(f"Updated position {position_id} in storage")

        except Exception as e:
            self.context.logger.exception(f"Failed to update position in storage: {e}")

    # =========== Helper methods ===========

    def _auto_continue(self) -> None:
        """Automatically continue the operation."""
        if self.active_operation:
            try:
                self._continue_operation()
            except Exception as e:
                self.context.logger.exception(f"Failed to continue operation: {e}")
                self._fail_operation(f"Auto-continue error: {e!s}")

    def _track_dialogue(self, dialogue: BaseDialogue, operation_type: str) -> None:
        """Track dialogue reference."""
        ref = dialogue.dialogue_label.dialogue_reference[0]
        self.pending_dialogues[ref] = operation_type
        self.context.logger.debug(f"Tracking {operation_type} dialogue: {ref}")

    def _clear_dialogue(self, dialogue: BaseDialogue) -> None:
        """Clear a completed dialogue."""
        ref = dialogue.dialogue_label.dialogue_reference[0]
        if ref in self.pending_dialogues:
            del self.pending_dialogues[ref]
            self.context.logger.debug(f"cleared dialogue: {ref}")

    def _get_safe_address(self) -> str | None:
        """Get the Safe address for Base chain."""
        safe_addresses = self.context.params.safe_contract_addresses

        if isinstance(safe_addresses, str):
            try:
                safe_dict = json.loads(safe_addresses)
                return safe_dict.get("base")
            except json.JSONDecodeError:
                return None
        elif isinstance(safe_addresses, dict):
            return safe_addresses.get("base")

        return None

    def _calculate_approval_amount(self, order: Order) -> int | None:
        """Calculate ERC20 approval amount in wei."""
        if order.side == OrderSide.BUY:
            usdc_amount = order.amount * order.price

            buffer_multiplier = 1.005  # Buffer for slippage and fees
            return int(usdc_amount * buffer_multiplier * 10**6)  # USDC has 6 decimals
        if order.side == OrderSide.SELL:
            msg = "Sell orders are not supported yet"
            raise NotImplementedError(msg)
        return None

    def _get_preapproved_signature(self) -> str:
        """Generate pre-approved signature format."""
        owner = self.context.agent_address

        # Convert address to bytes (32 bytes, left-padded)
        r_bytes = to_bytes(hexstr=owner[2:].rjust(64, "0"))
        s_bytes = b"\x00" * 32  # 32 zero bytes
        v_bytes = to_bytes(1)  # Single byte with value 1

        return (r_bytes + s_bytes + v_bytes).hex()

    def _has_pending_responses(self) -> bool:
        """Check if there are any pending responses to process."""
        return any(len(messages) > 0 for messages in self.supported_protocols.values())

    def _process_responses(self) -> None:
        """Process any pending protocol responses."""
        order_messages = self.supported_protocols.get(OrdersMessage.protocol_id, [])
        if order_messages:
            self._process_order_response(order_messages.pop(0))

    def _process_order_response(self, message: OrdersMessage) -> None:
        """Process an order response message."""
        if message.performative == OrdersMessage.Performative.ERROR:
            self.context.logger.error(f"Order error: {message}")
            if self.active_operation:
                self._fail_operation("Order execution error")
        elif message.performative in {OrdersMessage.Performative.ORDER, OrdersMessage.Performative.ORDER_CREATED}:
            # For DCXT, this is handled through validation functions
            pass

    def _is_timeout(self) -> bool:
        """Check if execution has timed out."""
        if not self.execution_started_at:
            return False

        elapsed = (datetime.now(UTC) - self.execution_started_at).total_seconds()
        return elapsed > ORDER_PLACEMENT_TIMEOUT_SECONDS

    def _handle_timeout(self) -> None:
        """Handle execution timeout."""
        self.context.logger.warning("Execution timeout reached")

        # Fail all pending operations
        for order in self.submitted_orders:
            order.status = OrderStatus.FAILED
            order.info = "Timeout"
            self.failed_orders.append(order)

        self.submitted_orders.clear()
        self.pending_orders.clear()
        self.active_operation = None

        self._finalize()

    def _is_complete(self) -> bool:
        """Check if execution is complete."""
        return (
            len(self.pending_orders) == 0
            and len(self.submitted_orders) == 0
            and self.active_operation is None
            and len(self.pending_dialogues) == 0
        )

    def _finalize(self) -> None:
        """Finalize execution round."""
        successful = len(self.completed_orders)
        failed = len(self.failed_orders)
        total = successful + failed

        self.context.logger.info(f"Execution complete: {successful}/{total} successful")

        if self.completed_orders:
            self.context.logger.info("Successful orders:")
            for order in self.completed_orders:
                self.context.logger.info(f"  {order.side.name} {order.amount} {order.symbol} at ${order.price:.6f}")

        if self.failed_orders:
            self.context.logger.info("Failed orders:")
            for order in self.failed_orders:
                self.context.logger.info(f"  {order.symbol}: {order.info}")

        # Store execution summary
        self._store_execution_summary()

        # Clean up context
        self._cleanup_context()

        # Set completion event
        if successful > 0:
            self._complete(MindshareabciappEvents.EXECUTED)
        else:
            self._complete(MindshareabciappEvents.FAILED)

    def _store_execution_summary(self) -> None:
        """Store execution summary for analysis and reporting."""
        if not self.context.store_path:
            return

        try:
            summary_file = self.context.store_path / "execution_history.json"

            # Load existing history
            history = {"executions": []}
            if summary_file.exists():
                with open(summary_file, encoding="utf-8") as f:
                    history = json.load(f)

            # Create execution summary
            execution_summary = {
                "timestamp": datetime.now(UTC).isoformat(),
                "execution_type": self.execution_type,
                "total_orders": len(self.completed_orders) + len(self.failed_orders),
                "successful_orders": len(self.completed_orders),
                "failed_orders": len(self.failed_orders),
                "completed": [
                    {
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.name,
                        "amount": order.filled or order.amount,
                        "price": order.price,
                        "timestamp": datetime.fromtimestamp(order.timestamp, UTC).isoformat(),
                    }
                    for order in self.completed_orders
                ],
                "failed": [
                    {
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.name,
                        "error_message": getattr(order, "error_message", "Unknown error"),
                    }
                    for order in self.failed_orders
                ],
            }

            # Add to history (keep last 100 executions)
            history["executions"].append(execution_summary)
            if len(history["executions"]) > 100:
                history["executions"] = history["executions"][-100:]

            history["last_updated"] = datetime.now(UTC).isoformat()

            # Save updated history
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            self.context.logger.exception(f"Failed to store execution summary: {e}")

    def _cleanup_context(self) -> None:
        """Cleanup context after execution."""
        if hasattr(self.context, "constructed_trade"):
            delattr(self.context, "constructed_trade")
        if hasattr(self.context, "positions_to_exit"):
            delattr(self.context, "positions_to_exit")

    def _complete(self, event: MindshareabciappEvents) -> None:
        """Mark round as complete with event."""
        self._event = event
        self._is_done = True

    def _handle_error(self, error: Exception) -> None:
        """Handle execution error."""
        self.context.logger.error(f"Execution error: {error!s}")

        self.context.error_context = {
            "error_type": "execution_error",
            "error_message": str(error),
            "originating_round": str(self._state),
        }

        self._complete(MindshareabciappEvents.FAILED)


class MindshareabciappFsmBehaviour(FSMBehaviour):
    """This class implements a simple Finite State Machine behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_transition_timestamp: datetime | None = None
        self._previous_rounds: list[str] = []
        self._period_count: int = 0

        self.register_state(MindshareabciappStates.SETUPROUND.value, SetupRound(**kwargs), True)
        self.register_state(MindshareabciappStates.HANDLEERRORROUND.value, HandleErrorRound(**kwargs))
        self.register_state(MindshareabciappStates.DATACOLLECTIONROUND.value, DataCollectionRound(**kwargs))
        self.register_state(MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value, PortfolioValidationRound(**kwargs))
        self.register_state(MindshareabciappStates.RISKEVALUATIONROUND.value, RiskEvaluationRound(**kwargs))
        self.register_state(MindshareabciappStates.PAUSEDROUND.value, PausedRound(**kwargs))
        self.register_state(MindshareabciappStates.CHECKSTAKINGKPIROUND.value, CheckStakingKPIRound(**kwargs))
        self.register_state(MindshareabciappStates.SIGNALAGGREGATIONROUND.value, SignalAggregationRound(**kwargs))
        self.register_state(MindshareabciappStates.POSITIONMONITORINGROUND.value, PositionMonitoringRound(**kwargs))
        self.register_state(MindshareabciappStates.ANALYSISROUND.value, AnalysisRound(**kwargs))
        self.register_state(MindshareabciappStates.TRADECONSTRUCTIONROUND.value, TradeConstructionRound(**kwargs))
        self.register_state(MindshareabciappStates.EXECUTIONROUND.value, ExecutionRound(**kwargs))

        self.register_transition(
            source=MindshareabciappStates.ANALYSISROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.ANALYSISROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.DATACOLLECTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.DATACOLLECTIONROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.POSITIONMONITORINGROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.DATACOLLECTIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.EXECUTIONROUND.value,
            event=MindshareabciappEvents.EXECUTED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.EXECUTIONROUND.value,
            event=MindshareabciappEvents.FAILED,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RESET,
            destination=MindshareabciappStates.SETUPROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RETRIES_EXCEEDED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RETRY,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PAUSEDROUND.value,
            event=MindshareabciappEvents.RESET,
            destination=MindshareabciappStates.SETUPROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PAUSEDROUND.value,
            event=MindshareabciappEvents.RESUME,
            destination=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.AT_LIMIT,
            destination=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.CAN_TRADE,
            destination=MindshareabciappStates.ANALYSISROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.EXIT_SIGNAL,
            destination=MindshareabciappStates.EXECUTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.POSITIONS_CHECKED,
            destination=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.APPROVED,
            destination=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.REJECTED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SETUPROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SETUPROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.NO_SIGNAL,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.SIGNAL_GENERATED,
            destination=MindshareabciappStates.RISKEVALUATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.EXECUTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )

    def act(self) -> None:
        """Override act to track transitions."""
        if self.current is None:
            super().act()
            return

        # Store the current state before potential transition
        previous_state = self.current

        # Call parent act which handles the FSM logic
        super().act()

        # Check if a transition occurred
        if self.current != previous_state and self.current is not None:
            # A transition occurred - track it
            current_time = datetime.now(UTC)
            self._last_transition_timestamp = current_time

            # Track round history (keep last 25 rounds)
            if previous_state and previous_state not in self._previous_rounds[-1:]:  # Avoid duplicates
                self._previous_rounds.append(previous_state)
                if len(self._previous_rounds) > 25:
                    self._previous_rounds = self._previous_rounds[-25:]

            # Check if we transitioned back to data collection round (indicates a new operational period/cycle)
            if (
                self.current == MindshareabciappStates.DATACOLLECTIONROUND.value
                and previous_state != MindshareabciappStates.DATACOLLECTIONROUND.value
            ):
                self._period_count += 1
                self.context.logger.info(f"FSM started new operational period: {self._period_count}")

            self.context.logger.info(f"FSM transitioned from {previous_state} to {self.current}")

    @property
    def last_transition_timestamp(self) -> datetime | None:
        """Get the timestamp of the last transition."""
        return self._last_transition_timestamp

    @property
    def previous_rounds(self) -> list[str]:
        """Get the history of previous rounds."""
        return self._previous_rounds.copy()

    @property
    def period_count(self) -> int:
        """Get the current period count."""
        return self._period_count

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up Mindshareabciapp FSM behaviour.")
        self._last_transition_timestamp = datetime.now(UTC)

    def teardown(self) -> None:
        """Implement the teardown."""
        self.context.logger.info("Tearing down Mindshareabciapp FSM behaviour.")

    def terminate(self) -> None:
        """Implement the termination."""
        os._exit(0)
