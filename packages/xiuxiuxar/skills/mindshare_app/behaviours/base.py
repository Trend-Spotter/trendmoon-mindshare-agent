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
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from datetime import UTC, datetime
from dataclasses import dataclass

from eth_utils import to_bytes
from aea.protocols.base import Message
from aea.skills.behaviours import State
from aea.configurations.base import PublicId
from autonomy.deploy.constants import DEFAULT_ENCODING
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.eightballer.connections.dcxt import PUBLIC_ID as DCXT_PUBLIC_ID
from packages.eightballer.protocols.orders import OrdersMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.open_aea.protocols.signing.message import SigningMessage
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.eightballer.protocols.tickers.message import TickersMessage
from packages.eightballer.protocols.orders.custom_types import OrderStatus


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.mindshare_app.models import Coingecko, Trendmoon

LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)
ETHER_VALUE = 0
NULL_ADDRESS = "0x" + "0" * 40
SAFE_TX_GAS = 300_000  # Non-zero value to prevent Safe revert during gas estimation

ALLOWED_ASSETS: dict[str, list[dict[str, str]]] = {
    "base": [
        # BASE-chain Uniswap tokens
        {
            "address": "0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b",
            "symbol": "VIRTUAL",
            "coingecko_id": "virtual-protocol",
        },
        {
            "address": "0x532f27101965dd16442E59d40670FaF5eBB142E4",
            "symbol": "BRETT",
            "coingecko_id": "based-brett",
        },
        {
            "address": "0x4F9Fd6Be4a90f2620860d680c0d4d5Fb53d1A825",
            "symbol": "AIXBT",
            "coingecko_id": "aixbt",
        },
        {
            "address": "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed",
            "symbol": "DEGEN",
            "coingecko_id": "degen-base",
        },
        {
            "address": "0x54330d28ca3357f294334bdc454a032e7f353416",
            "symbol": "OLAS",
            "coingecko_id": "autonolas",
        },
        {
            "address": "0xAC1Bd2486aAf3B5C0fc3Fd868558b082a531B2B4",
            "symbol": "TOSHI",
            "coingecko_id": "toshi",
        },
        {
            "address": "0x1111111111166b7fe7bd91427724b487980afc69",
            "symbol": "ZORA",
            "coingecko_id": "zora",
        },
        {
            "address": "0x940181a94A35A4569E4529A3CDfB74e38FD98631",
            "symbol": "AERO",
            "coingecko_id": "aerodrome-finance",
        },
        {
            "address": "0x768BE13e1680b5ebE0024C42c896E3dB59ec0149",
            "symbol": "SKI",
            "coingecko_id": "ski-mask-dog",
        },
        {
            "address": "0x9E6A46f294bB67c20F1D1E7AfB0bBEf614403B55",
            "symbol": "MAG7.ssi",
            "coingecko_id": "mag7-ssi",
            "decimals": 8,
        },
        {
            "address": "0xB1a03EdA10342529bBF8EB700a06C60441fEf25d",
            "symbol": "MIGGLES",
            "coingecko_id": "mister-miggles",
        },
        {
            "address": "0x226a2fa2556c48245e57cd1cba4c6c9e67077dd2",
            "symbol": "BIO",
            "coingecko_id": "bio-protocol",
        },
        {
            "address": "0x2f299be3b081e8cd47dc56c1932fcae7a91b5dcd",
            "symbol": "XTTA",
            "coingecko_id": "xtta",
        },
        {
            "address": "0xd5C3a723e63a0ECaB81081c26c6A3c4b2634Bf85",
            "symbol": "WOJAK",
            "coingecko_id": "based-wojak",
        },
        {
            "address": "0xA4A2E2ca3fBfE21aed83471D28b6f65A233C6e00",
            "symbol": "TIBBIR",
            "coingecko_id": "ribbita-by-virtuals",
        },
    ]
}


@dataclass
class TradingStrategy:
    """Consolidated trading strategy for buy signal generation."""

    def __init__(self, strategy_type: str = "balanced", context=None):
        self.strategy_type = strategy_type.lower()
        self.context = context
        self._initialize_strategy_params()

    def _initialize_strategy_params(self) -> None:
        """Initialize strategy parameters based on strategy type."""
        # Use context params if available, otherwise fallback to defaults

        base_params = {
            "min_conditions_met": 4,  # All 4 core conditions must be met
            "rsi_lower_limit": self.context.params.rsi_lower_limit,
            "rsi_upper_limit": self.context.params.rsi_upper_limit,
            "adx_threshold": self.context.params.adx_threshold,
            "social_threshold": 0.6,
        }

        # Core trading conditions (all must be met for a trade)
        core_conditions = ["price_above_ma", "rsi_in_range", "macd_bullish", "adx_strong"]

        # Strategy-specific weights (for probability scoring only, not trading decisions)
        if self.strategy_type == "conservative":
            self.params = {
                **base_params,
                "core_conditions": core_conditions,
                "weights": {
                    "price_above_ma": 0.25,
                    "rsi_in_range": 0.25,
                    "macd_bullish": 0.25,
                    "adx_strong": 0.25,
                    "obv_increasing": 0.0,  # Data only
                    "social_bullish": 0.0,  # Data only
                },
                "social_weight": 0.3,
                "technical_weight": 0.7,
            }
        elif self.strategy_type == "aggressive":
            self.params = {
                **base_params,
                "social_threshold": 0.5,
                "core_conditions": core_conditions,
                "weights": {
                    "price_above_ma": 0.25,
                    "rsi_in_range": 0.25,
                    "macd_bullish": 0.25,
                    "adx_strong": 0.25,
                    "obv_increasing": 0.0,  # Data only
                    "social_bullish": 0.0,  # Data only
                },
                "social_weight": 0.5,
                "technical_weight": 0.5,
            }
        else:  # balanced (default)
            self.params = {
                **base_params,
                "core_conditions": core_conditions,
                "weights": {
                    "price_above_ma": 0.25,
                    "rsi_in_range": 0.25,
                    "macd_bullish": 0.25,
                    "adx_strong": 0.25,
                    "obv_increasing": 0.0,  # Data only
                    "social_bullish": 0.0,  # Data only
                },
                "social_weight": 0.4,
                "technical_weight": 0.6,
            }

    def evaluate_conditions(self, current_price: float, social_scores: dict, technical_scores: dict) -> dict:
        """Evaluate trading conditions and return results."""
        conditions_met = {}

        # Extract technical indicators
        sma = technical_scores.get("sma")
        rsi = technical_scores.get("rsi")
        macd = technical_scores.get("macd")
        macd_signal = technical_scores.get("macd_signal")
        adx = technical_scores.get("adx")
        _p_social = social_scores.get("p_social", 0.5)

        # 1. Price above 20MA
        conditions_met["price_above_ma"] = {
            "condition": current_price > sma if sma and sma > 0 else False,
            "weight": self.params["weights"]["price_above_ma"],
            "value": current_price - sma if sma else 0,
            "description": "Price is above MA",
        }

        # 2. RSI in healthy range
        conditions_met["rsi_in_range"] = {
            "condition": (self.params["rsi_lower_limit"] < rsi < self.params["rsi_upper_limit"])
            if rsi is not None
            else False,
            "weight": self.params["weights"]["rsi_in_range"],
            "value": rsi,
            "description": f"RSI in range ({self.params['rsi_lower_limit']}-{self.params['rsi_upper_limit']})",
        }

        # 3. MACD bullish cross
        conditions_met["macd_bullish"] = {
            "condition": macd > macd_signal if macd is not None and macd_signal is not None else False,
            "weight": self.params["weights"]["macd_bullish"],
            "value": macd - macd_signal if macd is not None and macd_signal is not None else 0,
            "description": "MACD above signal line",
        }

        # 4. ADX strong trend
        conditions_met["adx_strong"] = {
            "condition": adx > self.params["adx_threshold"] if adx is not None else False,
            "weight": self.params["weights"]["adx_strong"],
            "value": adx,
            "description": f"ADX indicates strong trend (>{self.params['adx_threshold']})",
        }

        return conditions_met

    def calculate_signal_strength(self, conditions_met: dict) -> tuple[float, int]:
        """Calculate weighted signal strength and count met conditions."""
        signal_strength = 0.0
        num_conditions_met = 0

        for condition_data in conditions_met.values():
            if condition_data["condition"]:
                signal_strength += condition_data["weight"]
                num_conditions_met += 1

        return signal_strength, num_conditions_met

    def should_generate_signal(self, conditions_met: dict) -> bool:
        """Determine if signal should be generated based on core conditions only."""
        # All 4 core conditions must be met for a trade signal
        core_conditions = self.params["core_conditions"]

        for condition_name in core_conditions:
            if condition_name not in conditions_met or not conditions_met[condition_name]["condition"]:
                return False

        return True

    def calculate_combined_probability(self, p_social: float, p_technical: float) -> float:
        """Calculate combined probability score using strategy weights."""
        return p_social * self.params["social_weight"] + p_technical * self.params["technical_weight"]


class StakingState(Enum):
    """Staking state enumeration for the staking."""

    UNSTAKED = 0
    STAKED = 1
    EVICTED = 2


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
    ORDER_PLACED = "ORDER_PLACED"
    RETRY = "RETRY"
    CAN_TRADE = "CAN_TRADE"
    SERVICE_NOT_STAKED = "SERVICE_NOT_STAKED"
    SERVICE_EVICTED = "SERVICE_EVICTED"
    CHECKPOINT_PREPARED = "CHECKPOINT_PREPARED"
    NEXT_CHECKPOINT_NOT_REACHED_YET = "NEXT_CHECKPOINT_NOT_REACHED_YET"


class MindshareabciappStates(Enum):
    """States for the fsm."""

    DATACOLLECTIONROUND = "datacollectionround"
    PORTFOLIOVALIDATIONROUND = "portfoliovalidationround"
    RISKEVALUATIONROUND = "riskevaluationround"
    PAUSEDROUND = "pausedround"
    CALLCHECKPOINTROUND = "callcheckpointround"
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

    def _get_preapproved_signature(self) -> str:
        """Generate pre-approved signature format."""
        owner = self.context.agent_address

        # Convert address to bytes (32 bytes, left-padded)
        r_bytes = to_bytes(hexstr=owner[2:].rjust(64, "0"))
        s_bytes = b"\x00" * 32  # 32 zero bytes
        v_bytes = to_bytes(1)  # Single byte with value 1

        return (r_bytes + s_bytes + v_bytes).hex()

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

    def _load_pending_trades(self) -> list[dict[str, Any]]:
        """Load pending trades from storage."""
        if not self.context.store_path:
            return []

        trades_file = self.context.store_path / "pending_trades.json"
        if not trades_file.exists():
            return []

        try:
            with open(trades_file, encoding=DEFAULT_ENCODING) as f:
                data = json.load(f)
                return data.get("trades", [])
        except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as e:
            self.context.logger.warning(f"Failed to load pending trades: {e}")
            return []

    def _update_pending_trades(self, trades: list[dict[str, Any]]) -> None:
        """Update pending trades in storage."""
        if not self.context.store_path:
            return

        try:
            trades_file = self.context.store_path / "pending_trades.json"

            updated_data = {
                "trades": trades,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_pending": len(trades),
            }

            with open(trades_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(updated_data, f, indent=2)

            self.context.logger.info(f"Updated pending trades: {len(trades)} trades")

        except Exception as e:
            self.context.logger.exception(f"Failed to update pending trades: {e}")

    def _find_position_by_id(self, position_id: str) -> dict[str, Any] | None:
        """Find position by ID in storage."""
        if not self.context.store_path:
            return None

        positions_file = self.context.store_path / "positions.json"
        if not positions_file.exists():
            return None

        try:
            with open(positions_file, encoding=DEFAULT_ENCODING) as f:
                data = json.load(f)
                positions = data.get("positions", [])

            for position in positions:
                if position.get("position_id") == position_id:
                    return position

            return None

        except json.JSONDecodeError as e:
            self.context.logger.warning(f"Failed to find position {position_id}: {e}")
            return None

    def monitor_cowswap_orders(self, order_ids: list[str]) -> None:
        """Monitor multiple CoWSwap orders."""
        if not order_ids:
            return

        self.context.logger.info(f"Monitoring CoWSwap orders: {order_ids}")

        safe_address = self._get_safe_address()
        dialogue = self.submit_msg(
            performative=OrdersMessage.Performative.GET_ORDERS,
            connection_id=str(DCXT_PUBLIC_ID),
            exchange_id="cowswap",
            ledger_id="base",
            account=safe_address or self.context.agent_address,
        )

        dialogue.validation_func = self._validate_cowswap_monitoring_response
        dialogue.monitored_order_ids = order_ids

    def _validate_cowswap_monitoring_response(self, message: OrdersMessage, dialogue: BaseDialogue) -> bool:
        """Validate CoWSwap order monitoring response."""
        try:
            if message.performative == OrdersMessage.Performative.ORDERS:
                return self._process_orders_response(message, dialogue)

            if message.performative == OrdersMessage.Performative.ERROR:
                self.context.logger.error(f"Error monitoring CoW orders: {message.error_msg}")
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating CoW monitoring response: {e}")
            return False

    def _process_orders_response(self, message: OrdersMessage, dialogue: BaseDialogue) -> bool:
        """Process orders response from CoWSwap monitoring."""
        orders = message.orders.orders
        monitored_order_ids = getattr(dialogue, "monitored_order_ids", [])

        self.context.logger.info(f"Checking {len(monitored_order_ids)} orders against {len(orders)} returned orders")

        order_updates = {}
        for target_order_id in monitored_order_ids:
            target_order = self._find_order_by_id(orders, target_order_id)
            order_updates[target_order_id] = self._determine_order_status(target_order_id, target_order)

        self._process_order_updates(order_updates)
        return True

    def _find_order_by_id(self, orders: list, target_order_id: str):
        """Find order by ID in the orders list."""
        for order in orders:
            if str(order.id) == str(target_order_id):
                return order
        return None

    def _determine_order_status(self, target_order_id: str, target_order) -> dict[str, Any]:
        """Determine the status of an order and log accordingly."""
        if target_order is None:
            self.context.logger.info(f"CoW order {target_order_id} no longer in open orders - assuming filled")
            return {"status": "filled", "order": None}

        if target_order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
            self.context.logger.info(f"CoW order {target_order_id} executed with status: {target_order.status}")
            return {"status": "filled", "order": target_order}

        if target_order.status in {OrderStatus.CANCELLED, OrderStatus.EXPIRED}:
            status_name = "cancelled" if target_order.status == OrderStatus.CANCELLED else "expired"
            self.context.logger.warning(f"CoW order {target_order_id} was {status_name}")
            return {"status": status_name, "order": target_order}

        self.context.logger.info(f"CoW order {target_order_id} still open with status: {target_order.status}")
        return {"status": "open", "order": target_order}

    def _process_order_updates(self, order_updates: dict[str, dict[str, Any]]) -> None:
        """Process order updates - to be implemented by subclasses."""
