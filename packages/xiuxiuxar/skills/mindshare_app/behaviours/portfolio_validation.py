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

from aea.protocols.base import Message
from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.connections.ledger.connection import (
    PUBLIC_ID as LEDGER_CONNECTION_PUBLIC_ID,
)
from packages.eightballer.contracts.erc_20.contract import Erc20
from packages.xiuxiuxar.skills.mindshare_app.dialogues import ContractApiDialogue
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


LEDGER_API_ADDRESS = str(LEDGER_CONNECTION_PUBLIC_ID)


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
                with open(positions_file, encoding=DEFAULT_ENCODING) as f:
                    positions_data = json.load(f)

                self.open_positions = [
                    pos for pos in positions_data.get("positions", []) if pos.get("status") == "open"
                ]

                self.portfolio_metrics["current_positions"] = len(self.open_positions)
                self.portfolio_metrics["total_portfolio_value"] = positions_data.get("total_portfolio_value", 0.0)

                # Use current_value_usdc if available (updated by position monitor), fallback to entry_value_usdc
                total_exposure = sum(
                    pos.get("current_value_usdc", pos.get("entry_value_usdc", 0.0)) for pos in self.open_positions
                )
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
                connection_id=LEDGER_API_ADDRESS,
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
        total_exposure = self.portfolio_metrics["total_exposure"]

        # Calculate NAV = Available Cash + Existing Positions
        nav = available_capital + total_exposure

        # Calculate dynamic buffer: buffer = NAV * BUFFER_RATIO (5%)
        dynamic_buffer = nav * self.context.params.buffer_ratio

        if available_capital <= dynamic_buffer:
            self.validation_result = (
                f"Insufficient available capital (${available_capital:.2f} <= ${dynamic_buffer:.2f} buffer)"
            )
            self.context.logger.info(self.validation_result)
            return False

        min_position_size = self.context.params.min_position_size_usdc
        available_for_trading = available_capital - dynamic_buffer

        if available_for_trading < min_position_size:
            self.validation_result = (
                f"Insufficient available capital for trading (${available_for_trading:.2f} < ${min_position_size:.2f})"
            )
            self.context.logger.info(self.validation_result)
            return False

        self.context.logger.info(
            "Available capital check: "
            f"PASSED - ${available_capital:.2f} > ${dynamic_buffer:.2f} buffer (5% of ${nav:.2f} NAV)"
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

        # Calculate dynamic buffer
        dynamic_buffer = total_portfolio_value * self.context.params.buffer_ratio
        available_for_trading = available_capital - dynamic_buffer

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
        available_capital = self.portfolio_metrics["available_capital_usdc"]
        total_exposure = self.portfolio_metrics["total_exposure"]
        nav = available_capital + total_exposure
        dynamic_buffer = nav * self.context.params.buffer_ratio

        return (
            self.portfolio_metrics["current_positions"] < self.portfolio_metrics["max_positions"]
            and available_capital > dynamic_buffer
        )

    @property
    def available_trading_capital(self) -> float:
        """Get the amount of capital available for new trades.

        This returns the TOTAL available capital WITHOUT subtracting the buffer.
        The buffer will be calculated dynamically in trade_construction based on NAV.
        """
        return self.portfolio_metrics["available_capital_usdc"]
