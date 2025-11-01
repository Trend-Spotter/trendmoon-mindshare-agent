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

from aea.protocols.base import Message
from autonomy.deploy.constants import DEFAULT_ENCODING
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
from packages.valory.contracts.gnosis_safe.contract import SafeOperation, GnosisSafeContract
from packages.valory.protocols.ledger_api.custom_types import Terms, TransactionDigest
from packages.eightballer.protocols.orders.custom_types import Order, OrderSide, OrderType, OrderStatus
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    ALLOWED_ASSETS,
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


ORDER_PLACEMENT_TIMEOUT_SECONDS = 30
LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)
SAFE_TX_GAS = 300_000  # Non-zero value to prevent Safe revert during gas estimation
NULL_ADDRESS = "0x" + "0" * 40
BALANCER_VAULT = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"  # Balancer V2 Vault on Base
COW_VAULT_RELAYER = "0xC92E8bdf79f0507f65a392b0ab4667716BFE0110"  # CoW Protocol GPv2VaultRelayer on Base
MULTISEND_ADDRESS = "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D"  # Base chain multisend


def truncate_to_decimals(amount: float, decimals: int = 4) -> float:
    """Truncate amount to specified number of decimal places to avoid precision issues."""
    if isinstance(amount, int | float):
        # Multiply by 10^decimals, truncate to int, then divide back
        factor = 10**decimals
        return float(int(amount * factor)) / factor
    return amount


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

        # Order metadata storage (position_id, exit_reason, etc.)
        self.order_metadata: dict[str, dict[str, Any]] = {}

        # Execution context
        self.execution_type: str = ""  # "entry" or "exit"
        self.execution_initialized: bool = False
        self.execution_started_at: datetime | None = None

        # Operation tracking
        self.active_operation: dict[str, Any] | None = None

        # Dialogue tracking
        self.pending_dialogues: dict[str, str] = {}

        # Balance tracking for exit orders
        self.pending_balance_queries: dict[str, str] = {}  # token_address -> dialogue_ref
        self.token_balances: dict[str, float] = {}  # token_address -> balance (human-readable)
        self.balance_queries_complete: bool = False

        # Failure tracking flags
        self.approve_request_failed: bool = False
        self.swap_request_failed: bool = False
        self.cow_approval_failed: bool = False
        self.cow_order_failed: bool = False
        self.cow_monitoring_failed: bool = False
        self.multisend_failed: bool = False
        self.safe_hash_failed: bool = False
        self.execution_failed: bool = False
        self.signing_failed: bool = False
        self.broadcast_failed: bool = False
        self.receipt_failed: bool = False

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
        self.order_metadata = {}

        self.active_operation = None
        self.pending_dialogues = {}

        # Reset balance tracking
        self.pending_balance_queries = {}
        self.token_balances = {}
        self.balance_queries_complete = False

        # Reset failure tracking flags
        self.approve_request_failed = False
        self.swap_request_failed = False
        self.cow_approval_failed = False
        self.cow_order_failed = False
        self.cow_monitoring_failed = False
        self.multisend_failed = False
        self.safe_hash_failed = False
        self.execution_failed = False
        self.signing_failed = False
        self.broadcast_failed = False
        self.receipt_failed = False

        for protocol in self.supported_protocols:
            self.supported_protocols[protocol] = []

    def act(self) -> None:  # noqa: PLR0911
        """Perform the act."""
        try:
            # Initialize if needed
            if not self.execution_initialized:
                self._initialize_execution()
                if self._is_done:
                    return

            # For exit orders, wait for balance queries to complete before creating orders
            if self.execution_type == "exit" and not self.pending_orders:
                # If balance queries not complete yet, wait for them
                if not self.balance_queries_complete:
                    # Process any incoming messages (including balance responses)
                    if self._has_pending_responses():
                        self._process_responses()
                        return
                    # Still waiting for balance responses
                    return

                # Balance queries complete and no orders created yet - create them now
                self._create_exit_orders_from_balances()
                # Orders created, will be submitted in next act() cycle
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

        # First, query on-chain balances for all tokens we need to exit
        unique_tokens = set()
        for position in self.context.positions_to_exit:
            contract_address = position.get("contract_address")
            if contract_address:
                unique_tokens.add(contract_address)

        # If no valid tokens found, mark as complete immediately
        if not unique_tokens:
            self.context.logger.warning("No valid tokens found for exit (missing contract addresses)")
            self.balance_queries_complete = True
            return

        # Submit balance queries for all unique tokens
        for token_address in unique_tokens:
            self._query_token_balance(token_address)

        self.context.logger.info(f"Submitted balance queries for {len(unique_tokens)} tokens")
        # Don't create orders yet - wait for balance queries to complete

    def _create_exit_orders_from_balances(self) -> None:
        """Create exit orders using actual on-chain balances."""
        self.context.logger.info("Creating exit orders using actual on-chain balances")

        for position in self.context.positions_to_exit:
            order = self._create_exit_order(position)
            if order:
                self.pending_orders.append(order)

        self.context.logger.info(f"Created {len(self.pending_orders)} exit orders")

    def _setup_entry_execution(self) -> None:
        """Prepare entry orders for new position."""
        self.execution_type = "entry"

        order = self._create_entry_order(self.context.constructed_trade)
        if order:
            self.pending_orders.append(order)

    # =========== ORDER CREATION ===========

    def _create_exit_order(self, position: dict[str, Any]) -> Order | None:
        """Create an exit order from position data using actual on-chain balance."""
        symbol = position.get("symbol")
        contract_address = position.get("contract_address")
        stored_quantity = position.get("token_quantity", 0)

        if not contract_address:
            self.context.logger.warning(f"Missing contract_address for {symbol}")
            return None

        # Get actual on-chain balance
        actual_balance = self.token_balances.get(contract_address, 0)

        if actual_balance <= 0:
            self.context.logger.warning(
                f"No balance found for {symbol} ({contract_address}). "
                f"Stored: {stored_quantity}, Actual: {actual_balance}"
            )
            return None

        # Use actual balance and truncate to 4 decimals for precision
        # Sell the entire balance to fully close the position
        quantity = truncate_to_decimals(actual_balance, 4)

        # Log comparison between stored and actual balance
        balance_diff = stored_quantity - actual_balance
        balance_diff_pct = (balance_diff / stored_quantity * 100) if stored_quantity > 0 else 0
        self.context.logger.info(
            f"Exit {symbol}: stored={stored_quantity:.6f}, actual={actual_balance:.6f}, "
            f"diff={balance_diff:.6f} ({balance_diff_pct:.2f}%), selling={quantity:.4f}"
        )

        # Select best exchange for this exit trade
        exchange_id = self._select_exchange_for_trade("exit", symbol, quantity)

        order = Order(
            id=f"exit_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            symbol=f"{symbol}/USDC",
            asset_a=contract_address,  # Token being sold
            asset_b=self.context.params.base_usdc_address,  # USDC being bought
            side=OrderSide.SELL,
            amount=quantity,  # Amount in token units (human readable) - actual balance
            price=position.get("exit_price", position.get("current_price", 0)),
            type=OrderType.MARKET,
            exchange_id=exchange_id,
            ledger_id="base",
            status=OrderStatus.NEW,
            timestamp=datetime.now(UTC).timestamp(),
        )

        # Store metadata separately (Order model doesn't have these fields)
        self.order_metadata[order.id] = {
            "position_id": position.get("position_id"),
            "exit_reason": position.get("exit_reason", "unknown"),
        }

        return order

    def _create_entry_order(self, trade: dict[str, Any]) -> Order | None:
        """Create an entry order from trade construction."""
        symbol = trade.get("symbol")
        quantity = truncate_to_decimals(trade.get("token_quantity"))

        # Select best exchange for this entry trade
        exchange_id = self._select_exchange_for_trade("entry", symbol, quantity)

        order = Order(
            id=trade.get("trade_id"),
            symbol=f"{symbol}/USDC",
            asset_a=trade.get("buy_token"),
            asset_b=trade.get("sell_token"),
            side=OrderSide.BUY,
            amount=quantity,  # Amount in BUY tokens
            price=trade.get("entry_price"),
            type=OrderType.MARKET,
            exchange_id=exchange_id,
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
        """Create a new operation (multisend for Balancer, direct for CoWSwap)."""
        # Determine exchange from order or use default
        exchange_id = getattr(order, "exchange_id", "balancer")

        return {
            "order": order,
            "exchange_id": exchange_id,
            "state": "approve_pending",
            "approve_data": None,
            "swap_data": None,
            "multisend_data": None,
            "safe_hash": None,
            "created_at": datetime.now(UTC),
        }

    def _continue_operation(self) -> None:
        """Continue processing the active operation (multisend for Balancer, direct for CoWSwap)."""
        if not self.active_operation:
            return

        exchange_id = self.active_operation.get("exchange_id", "balancer")
        state = self.active_operation["state"]

        if exchange_id == "cowswap":
            self._continue_cowswap_operation(state)
        else:
            self._continue_balancer_operation(state)

    def _check_balancer_failures(self) -> bool:
        """Check for Balancer operation failures and handle them. Returns True if failure detected."""
        failures = {
            self.approve_request_failed: "ERC20 approval request failed",
            self.swap_request_failed: "Swap request failed",
            self.multisend_failed: "Multisend construction failed",
            self.safe_hash_failed: "Safe transaction hash request failed",
            self.execution_failed: "Safe transaction execution failed",
            self.signing_failed: "Transaction signing failed",
            self.broadcast_failed: "Transaction broadcast failed",
            self.receipt_failed: "Transaction failed on chain",
        }

        for failed, reason in failures.items():
            if failed:
                self._fail_operation(reason)
                return True
        return False

    def _continue_balancer_operation(self, state: str) -> None:
        """Continue processing Balancer multisend operation."""
        if self._check_balancer_failures():
            return

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

    def _check_cowswap_failures(self) -> bool:
        """Check for CoWSwap operation failures and handle them. Returns True if failure detected."""
        failures = {
            self.cow_approval_failed: "CoW approval request failed",
            self.cow_order_failed: "CoW order submission failed",
            self.cow_monitoring_failed: "CoW order monitoring failed",
            self.safe_hash_failed: "Safe transaction hash request failed",
            self.execution_failed: "Safe transaction execution failed",
            self.signing_failed: "Transaction signing failed",
            self.broadcast_failed: "Transaction broadcast failed",
            self.receipt_failed: "Transaction failed on chain",
        }

        for failed, reason in failures.items():
            if failed:
                self._fail_operation(reason)
                return True
        return False

    def _continue_cowswap_operation(self, state: str) -> None:
        """Continue processing CoWSwap operation."""
        if self._check_cowswap_failures():
            return

        if state == "approve_pending":
            self._request_cow_approval_transaction()
        elif state == "approval_confirmed":
            self._submit_cow_order()
        elif state == "monitoring":
            self._monitor_cow_execution()
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
        exchange_id = self.active_operation.get("exchange_id", "balancer")
        spender = self._get_spender_for_exchange(exchange_id)

        # Determine which token needs approval based on order side
        # BUY orders: approve asset_b (USDC being spent)
        # SELL orders: approve asset_a (token being sold)
        token_to_approve = order.asset_b if order.side == OrderSide.BUY else order.asset_a

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            contract_address=token_to_approve,
            contract_id=str(ERC20.contract_id),
            ledger_id="ethereum",
            callable="build_approval_tx",
            kwargs=ContractApiMessage.Kwargs(
                {
                    "spender": spender,
                    "amount": amount,
                }
            ),
        )

        dialogue.validation_func = self._validate_approve_response
        self._track_dialogue(dialogue, "approve")

        self.context.logger.info(f"Requested ERC20 approval for {token_to_approve}")

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
            self.approve_request_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing approve response: {e}")
            self.approve_request_failed = True
            return True

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
            ledger_id="ethereum",
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
            self.swap_request_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing swap response: {e}")
            self.swap_request_failed = True
            return True

    # =========== Exchange Selection ===========

    def _select_exchange_for_trade(self, _trade_type: str, _symbol: str, _amount: float) -> str:
        """Select the best exchange for a trade based on various factors."""
        # Placeholder

        return "cowswap"

    def _get_spender_for_exchange(self, exchange_id: str) -> str:
        """Get the appropriate spender address for the exchange."""
        if exchange_id == "cowswap":
            return COW_VAULT_RELAYER
        return BALANCER_VAULT

    # =========== CoWSwap Methods ===========

    def _request_cow_approval_transaction(self) -> None:
        """Request ERC20 approval transaction for CoWSwap (direct, no multisend)."""
        if any(dialogue_type == "approve" for dialogue_type in self.pending_dialogues.values()):
            self.context.logger.info("CoW approval dialogue already in progress, skipping approval request")
            return

        order = self.active_operation["order"]
        amount = self._calculate_approval_amount(order)

        token_address = order.asset_b if order.side == OrderSide.BUY else order.asset_a

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            contract_address=token_address,
            contract_id=str(ERC20.contract_id),
            ledger_id="ethereum",
            callable="build_approval_tx",
            kwargs=ContractApiMessage.Kwargs(
                {
                    "spender": COW_VAULT_RELAYER,
                    "amount": amount,
                }
            ),
        )

        dialogue.validation_func = self._validate_cow_approval_response
        self._track_dialogue(dialogue, "approve")

        self.context.logger.info(f"Requested CoW ERC20 approval for {token_address}")

    def _validate_cow_approval_response(self, message: Message, dialogue: BaseDialogue) -> bool:
        """Process CoW ERC20 approval response and prepare for Safe execution."""
        try:
            self.context.logger.info(f"Validating CoW approval response: {message}")
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                data = message.raw_transaction.body.get("data")
                if data:
                    self.active_operation["approve_data"] = "0x" + data.hex()
                    # For CoWSwap, we need to execute the approval as a single Safe transaction
                    self.active_operation["state"] = "safe_hash_pending"
                    self._clear_dialogue(dialogue)

                    self._auto_continue()
                    return True

            self.context.logger.error(f"Invalid CoW approval response: {message.performative}")
            self.cow_approval_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing CoW approval response: {e}")
            self.cow_approval_failed = True
            return True

    def _submit_cow_order(self) -> None:
        """Submit order to CoW Protocol via API after approval confirmed."""
        order = self.active_operation["order"]

        safe_address = self._get_safe_address()
        if safe_address:
            order.info = json.dumps({"safe_contract_address": safe_address})

        dialogue = self.submit_msg(
            performative=OrdersMessage.Performative.CREATE_ORDER,
            connection_id=str(DCXT_PUBLIC_ID),
            order=order,
            exchange_id="cowswap",
            ledger_id="ethereum",
        )

        dialogue.validation_func = self._validate_cow_order_response
        self._track_dialogue(dialogue, "cow_order")

        self.context.logger.info(f"Submitted CoW order: {order.id}")

    def _validate_cow_order_response(self, message: OrdersMessage, dialogue: BaseDialogue) -> bool:
        """Process CoW order submission response."""
        try:
            if message.performative in {OrdersMessage.Performative.ORDER, OrdersMessage.Performative.ORDER_CREATED}:
                response_order = message.order
                if response_order:
                    # Store original order ID for tracking
                    original_order_id = self.active_operation["order"].id

                    # Update the order object with the actual CoW Protocol order ID
                    self.active_operation["order"].id = response_order.id

                    # Store the original ID in the operation metadata for reference
                    self.active_operation["original_order_id"] = original_order_id

                    # Append order ID to pending_trades.json for easy reference
                    self._append_order_id_to_pending_trades(original_order_id, response_order.id)

                    self.context.logger.info(
                        f"CoW order submitted successfully. Original ID: {original_order_id}, "
                        f"CoW Protocol ID: {response_order.id}"
                    )

                    self.active_operation["state"] = "monitoring"
                    self._clear_dialogue(dialogue)

                    self._auto_continue()
                    return True

            self.context.logger.error(f"Invalid CoW order response: {message.performative}")
            self.cow_order_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing CoW order response: {e}")
            self.cow_order_failed = True
            return True

    def _append_order_id_to_pending_trades(self, original_order_id: str, cowswap_order_id: str) -> None:
        """Append the CoWSwap order ID to pending_trades.json for easy reference in position monitoring."""
        try:
            if not self.context.store_path:
                self.context.logger.warning("No store_path available, cannot update pending_trades.json")
                return

            trades_file = self.context.store_path / "pending_trades.json"

            if not trades_file.exists():
                self.context.logger.warning(f"pending_trades.json not found at {trades_file}")
                return

            # Load existing pending trades
            with open(trades_file, encoding=DEFAULT_ENCODING) as f:
                pending_trades = json.load(f)

            # Find the trade with matching original order ID and add CoWSwap order ID
            trades_updated = False
            for trade in pending_trades.get("trades", []):
                if trade.get("trade_id") == original_order_id:
                    trade["cowswap_order_id"] = cowswap_order_id
                    trade["order_submitted_at"] = datetime.now(UTC).isoformat()
                    trades_updated = True
                    self.context.logger.info(f"Added CoWSwap order ID {cowswap_order_id} to trade {original_order_id}")
                    break

            if trades_updated:
                # Update timestamp and save
                pending_trades["last_updated"] = datetime.now(UTC).isoformat()

                with open(trades_file, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(pending_trades, f, indent=2)

                self.context.logger.info("Updated pending_trades.json with CoWSwap order ID")
            else:
                self.context.logger.warning(f"Could not find trade with ID {original_order_id} in pending_trades.json")

        except Exception as e:
            self.context.logger.exception(f"Failed to append order ID to pending_trades.json: {e}")

    def _monitor_cow_execution(self) -> None:
        """Monitor CoW order execution status."""
        order = self.active_operation["order"]
        self.context.logger.info(f"Monitoring CoW order execution: {order.id}")

        # Submit GET_ORDERS request to check order status
        dialogue = self.submit_msg(
            performative=OrdersMessage.Performative.GET_ORDERS,
            connection_id=str(DCXT_PUBLIC_ID),
            exchange_id="cowswap",
            ledger_id="base",  # here ledger_id defines DCXT chain for the exchange.
            account=self._get_safe_address() or self.context.agent_address,
        )

        # Set validation function for the response
        dialogue.validation_func = self._validate_cow_monitoring_response
        self.active_operation["monitoring_dialogue"] = dialogue

    def _validate_cow_monitoring_response(self, message: OrdersMessage, _dialogue: BaseDialogue) -> bool:
        """Validate CoW order monitoring response and finalize if needed."""
        try:
            if message.performative == OrdersMessage.Performative.ORDERS:
                orders = message.orders.orders
                target_order_id = self.active_operation["order"].id

                self.context.logger.info(f"Checking CoW order {target_order_id} against {len(orders)} returned orders")

                # Debug: log all order IDs for comparison
                if orders:
                    order_ids = [order.id for order in orders]
                    self.context.logger.info(f"Returned order IDs: {order_ids}")
                else:
                    self.context.logger.info("No orders returned from CoW API")

                # Look for our specific order in the returned orders
                target_order = None
                for order in orders:
                    self.context.logger.debug(f"Comparing order ID: '{order.id}' with target: '{target_order_id}'")
                    if str(order.id) == str(target_order_id):
                        target_order = order
                        self.context.logger.info(f"Found matching order: {order.id} with status: {order.status}")
                        break

                if target_order is None:
                    # Order not found in open orders - it has been filled or cancelled
                    self.context.logger.info(f"CoW order {target_order_id} no longer in open orders - assuming filled")
                    self._finalize_cow_order()
                elif target_order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                    self.context.logger.info(f"CoW order {target_order_id} executed with status: {target_order.status}")
                    self.active_operation["order"] = target_order  # Update with latest status
                    self._finalize_cow_order()
                elif target_order.status in {OrderStatus.CANCELLED, OrderStatus.EXPIRED}:
                    status_msg = "cancelled" if target_order.status == OrderStatus.CANCELLED else "expired"
                    self.context.logger.warning(f"CoW order {target_order_id} was {status_msg}")
                    self.active_operation["state"] = "failed"
                else:
                    # Order still open, transition to next round with ORDER_PLACED event
                    self.context.logger.info(
                        f"CoW order {target_order_id} still open with status: {target_order.status}"
                    )
                    self._complete(MindshareabciappEvents.ORDER_PLACED)
                return True

            if message.performative == OrdersMessage.Performative.ERROR:
                self.context.logger.error(f"Error monitoring CoW order: {message.error_msg}")
                # Continue monitoring on error - don't fail the operation
                return True

            self.context.logger.error(f"Unexpected CoW monitoring performative: {message.performative}")
            self.cow_monitoring_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error validating CoW monitoring response: {e}")
            self.cow_monitoring_failed = True
            return True

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

        # Determine which token was approved based on order side
        # BUY orders: approve asset_b (USDC being spent)
        # SELL orders: approve asset_a (token being sold)
        token_approved = order.asset_b if order.side == OrderSide.BUY else order.asset_a

        payload = [
            {"operation": MultiSendOperation.CALL, "to": token_approved, "value": 0, "data": approve_data},
            {"operation": MultiSendOperation.CALL, "to": BALANCER_VAULT, "value": 0, "data": swap_data},
        ]

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
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
            self.multisend_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing multisend response: {e}")
            self.multisend_failed = True
            return True

    # =========== Safe transaction ===========

    def _request_safe_hash(self) -> None:
        """Request Safe transaction hash."""
        safe_address = self._get_safe_address()
        exchange_id = self.active_operation.get("exchange_id", "balancer")

        if exchange_id == "cowswap":
            # For CoWSwap, create direct approval transaction
            order = self.active_operation["order"]
            approve_data = self.active_operation["approve_data"]

            token_address = order.asset_b if order.side == OrderSide.BUY else order.asset_a

            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=safe_address,
                contract_id=str(GnosisSafeContract.contract_id),
                callable="get_raw_safe_transaction_hash",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs(
                    {
                        "to_address": token_address,  # Token contract
                        "value": 0,
                        "data": bytes.fromhex(approve_data[2:]),
                        "operation": SafeOperation.CALL.value,
                        "safe_tx_gas": SAFE_TX_GAS,  # Use defined gas amount
                        "gas_price": 0,
                    }
                ),
            )
            dialogue.original_data = approve_data
            transaction_type = "CoWSwap approval"
        else:
            # For Balancer, use multisend transaction
            multisend_data = self.active_operation["multisend_data"]

            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id=LEDGER_API_ADDRESS,
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
            dialogue.original_data = multisend_data.body["data"]
            transaction_type = "Balancer multisend"

        dialogue.validation_func = self._validate_safe_hash_response
        self._track_dialogue(dialogue, "safe_hash")

        self.context.logger.info(f"Requesting Safe transaction hash for {transaction_type}")

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
            self.safe_hash_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing safe hash response: {e}")
            self.safe_hash_failed = True
            return True

    def _execute_safe_tx(self) -> None:
        """Execute Safe transaction with pre-approved signature."""
        safe_address = self._get_safe_address()
        call_data = self.active_operation["original_data"]
        exchange_id = self.active_operation.get("exchange_id", "balancer")

        if exchange_id == "cowswap":
            # For CoWSwap: direct CALL to token contract
            order = self.active_operation["order"]

            to_address = order.asset_b if order.side == OrderSide.BUY else order.asset_a
            operation = SafeOperation.CALL.value
        else:
            # For Balancer: DELEGATE_CALL to MultiSend
            to_address = MULTISEND_ADDRESS
            operation = SafeOperation.DELEGATE_CALL.value

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            contract_address=safe_address,
            contract_id=str(GnosisSafeContract.contract_id),
            callable="get_raw_safe_transaction",
            ledger_id="ethereum",  # Use Base chain
            kwargs=ContractApiMessage.Kwargs(
                {
                    "sender_address": self.context.agent_address,
                    "owners": (self.context.agent_address,),
                    "to_address": to_address,
                    "value": 0,
                    "data": bytes.fromhex(call_data.removeprefix("0x")),
                    "signatures_by_owner": {self.context.agent_address: self._get_preapproved_signature()},
                    "operation": operation,
                    "safe_tx_gas": SAFE_TX_GAS,
                    "base_gas": 0,
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
            self.execution_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing execution response: {e}")
            self.execution_failed = True
            return True

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
            self.signing_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing signing response: {e}")
            self.signing_failed = True
            return True

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
            self.broadcast_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing broadcast response: {e}")
            self.broadcast_failed = True
            return True

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
                    self.context.logger.info("Transaction successful on chain!")
                    self._clear_dialogue(dialogue)
                    self._complete_operation()
                    return True
                self.context.logger.error(f"Transaction failed on chain: {receipt}")
                self.receipt_failed = True
                return True

            self.context.logger.error(f"Invalid receipt response: {message.performative}")
            self.receipt_failed = True
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing receipt: {e}")
            self.receipt_failed = True
            return True

    # =========== Operation completion ===========

    def _complete_operation(self) -> None:
        """Mark operation as complete and update positions."""
        exchange_id = self.active_operation.get("exchange_id", "balancer")

        # For CoWSwap, completion depends on the current state
        if exchange_id == "cowswap":
            current_state = self.active_operation.get("state", "")
            if current_state == "receipt_pending":
                # This means the approval transaction was successful
                # Transition to submit the CoW order
                self.active_operation["state"] = "approval_confirmed"
                self._auto_continue()
                return
            if current_state == "monitoring":
                # This is the final completion of the CoW order
                self._finalize_cow_order()
                return

        # For Balancer or final CoW completion
        self._finalize_order()

    def _finalize_order(self) -> None:
        """Complete the order and update positions."""
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

    def _finalize_cow_order(self) -> None:
        """Complete the CoW order execution."""
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

        self.context.logger.info(f"CoW order {order.id} completed successfully")

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

        # Clean up metadata for failed order
        if order.id in self.order_metadata:
            del self.order_metadata[order.id]

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
            # Get metadata from order_metadata dictionary
            metadata = self.order_metadata.get(order.id, {})
            position_id = metadata.get("position_id")
            exit_reason = metadata.get("exit_reason", "manual")

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
                "exit_reason": exit_reason,
                "executed_quantity": executed_quantity,
                "realized_pnl": realized_pnl,
                "realized_pnl_percentage": realized_pnl_percentage,
                "order_id": order.id,
                "closed_at": datetime.now(UTC).isoformat(),
                "partial_fill": partial_fill,
            }

            # Update positions storage
            self._update_position_in_storage(closed_position)

            # Clean up metadata to prevent memory leaks
            if order.id in self.order_metadata:
                del self.order_metadata[order.id]

            self.context.logger.info(
                f"{'Partially ' if partial_fill else ''}Closed position {original_position.get('symbol')} with "
                f"P&L: ${realized_pnl:.2f} ({realized_pnl_percentage:.2f}%)"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to finalize position exit: {e}")
            # Clean up metadata even on error
            if order.id in self.order_metadata:
                del self.order_metadata[order.id]

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

            # Extract contract address based on order side
            # For BUY orders: asset_a is the token being bought
            # For SELL orders: asset_b is the token being bought (but this shouldn't happen for entries)
            contract_address = order.asset_a if order.side == OrderSide.BUY else order.asset_b

            new_position = {
                "position_id": position_id,
                "symbol": symbol,
                "contract_address": contract_address,
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
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
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

            with open(positions_file, "w", encoding=DEFAULT_ENCODING) as f:
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
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
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

            with open(positions_file, "w", encoding=DEFAULT_ENCODING) as f:
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

    def _calculate_approval_amount(self, order: Order) -> int | None:
        """Calculate ERC20 approval amount in wei."""
        if order.side == OrderSide.BUY:
            usdc_amount = order.amount * order.price

            buffer_multiplier = 1.005  # Buffer for slippage and fees
            return int(usdc_amount * buffer_multiplier * 10**6)  # USDC has 6 decimals
        if order.side == OrderSide.SELL:
            # For sell orders, approve the token amount being sold
            token_address = order.asset_a
            token_amount = order.amount
            buffer_multiplier = 1.005

            # Look up token decimals from ALLOWED_ASSETS
            token_decimals = 18  # Default to standard ERC20 decimals
            for token in ALLOWED_ASSETS["base"]:
                if token.get("address") == token_address:
                    token_decimals = token.get("decimals", 18)
                    break

            return int(token_amount * buffer_multiplier * 10**token_decimals)
        return None

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
            # Clean up metadata for timed out orders
            if order.id in self.order_metadata:
                del self.order_metadata[order.id]

        # Clean up metadata for any pending orders that won't be executed
        for order in self.pending_orders:
            if order.id in self.order_metadata:
                del self.order_metadata[order.id]

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
                with open(summary_file, encoding=DEFAULT_ENCODING) as f:
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
            with open(summary_file, "w", encoding=DEFAULT_ENCODING) as f:
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

    # =========== Balance Query Methods ===========

    def _query_token_balance(self, token_address: str) -> None:
        """Query ERC20 token balance for the Safe address."""
        safe_address = self._get_safe_address()
        if not safe_address:
            self.context.logger.error("No Safe address available for balance query")
            return

        self.context.logger.info(f"Querying balance for token {token_address}")

        dialogue = self.submit_msg(
            performative=ContractApiMessage.Performative.GET_STATE,
            connection_id=LEDGER_API_ADDRESS,
            contract_address=token_address,
            contract_id=str(Erc20.contract_id),
            ledger_id="ethereum",
            callable="balance_of",
            kwargs=ContractApiMessage.Kwargs({"account": safe_address, "chain_id": "base"}),
        )

        dialogue.validation_func = self._validate_balance_response
        dialogue.token_address = token_address  # Store for reference
        self._track_dialogue(dialogue, "balance")

        # Track this balance query
        ref = dialogue.dialogue_label.dialogue_reference[0]
        self.pending_balance_queries[token_address] = ref

    def _validate_balance_response(self, message: ContractApiMessage, dialogue: BaseDialogue) -> bool:
        """Process ERC20 balance query response."""
        try:
            token_address = dialogue.token_address

            if (
                message.performative == ContractApiMessage.Performative.RAW_TRANSACTION
                and hasattr(message, "raw_transaction")
                and message.raw_transaction
            ):
                # balance_of returns {"int": balance}
                balance_wei = message.raw_transaction.body.get("int", 0)

                # Get token decimals from ALLOWED_ASSETS
                decimals = self._get_token_decimals(token_address)

                # Convert from wei to human-readable
                balance_human = balance_wei / (10**decimals)

                self.token_balances[token_address] = balance_human
                self.context.logger.info(
                    f"Retrieved balance for {token_address}: {balance_human} "
                    f"(wei: {balance_wei}, decimals: {decimals})"
                )

                # Remove from pending queries
                if token_address in self.pending_balance_queries:
                    del self.pending_balance_queries[token_address]

                # Check if all balance queries are complete
                if not self.pending_balance_queries:
                    self.balance_queries_complete = True
                    self.context.logger.info("All balance queries complete")

                self._clear_dialogue(dialogue)
                return True

            # Invalid response - store 0 balance and remove from pending
            self.context.logger.error(f"Invalid balance response for {token_address}: {message.performative}")
            self.token_balances[token_address] = 0
            if token_address in self.pending_balance_queries:
                del self.pending_balance_queries[token_address]

            # Check if all queries complete (even with failures)
            if not self.pending_balance_queries:
                self.balance_queries_complete = True
                self.context.logger.warning("All balance queries complete (some may have failed)")

            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing balance response: {e}")
            # Store 0 balance for failed query and remove from pending
            token_address = getattr(dialogue, "token_address", None)
            if token_address:
                self.token_balances[token_address] = 0
                if token_address in self.pending_balance_queries:
                    del self.pending_balance_queries[token_address]

                # Check if all queries complete (even with failures)
                if not self.pending_balance_queries:
                    self.balance_queries_complete = True
                    self.context.logger.warning("All balance queries complete (some may have failed)")

            return True

    def _get_token_decimals(self, token_address: str) -> int:
        """Get token decimals from ALLOWED_ASSETS configuration."""
        for token in ALLOWED_ASSETS["base"]:
            if token.get("address", "").lower() == token_address.lower():
                return token.get("decimals", 18)
        # Default to 18 if not found
        self.context.logger.warning(f"Token {token_address} not found in ALLOWED_ASSETS, using default 18 decimals")
        return 18

    def _find_position_by_id(self, position_id: str) -> dict[str, Any] | None:
        """Find a position by its ID from persistent storage."""
        if not self.context.store_path:
            return None

        positions_file = self.context.store_path / "positions.json"
        if not positions_file.exists():
            return None

        try:
            with open(positions_file, encoding=DEFAULT_ENCODING) as f:
                data = json.load(f)

            for position in data.get("positions", []):
                if position.get("position_id") == position_id:
                    return position

            return None
        except Exception as e:
            self.context.logger.exception(f"Failed to find position {position_id}: {e}")
            return None
