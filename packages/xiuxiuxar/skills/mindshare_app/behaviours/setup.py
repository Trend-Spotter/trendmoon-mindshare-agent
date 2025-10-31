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

"""This package contains the Mindshare App behaviour for the setup round."""

import json
from typing import Any
from pathlib import Path
from datetime import UTC, datetime

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


class SetupRound(BaseState):
    """This class implements the behaviour of the state SetupRound."""

    supported_protocols = {
        ContractApiMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.SETUPROUND
        self.setup_success: bool = False
        self.setup_data: dict[str, Any] = {}
        self.started: bool = False
        self.balance_check_submitted: bool = False
        self.balance_received: bool = False
        self.usdc_balance: float | None = None
        self.pending_contract_calls: list[ContractApiDialogue] = []
        self.contract_responses: dict[str, Any] = {}

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False
        self.balance_check_submitted = False
        self.balance_received = False
        self.usdc_balance = None
        self.pending_contract_calls = []
        self.contract_responses = {}
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def _initialize_state(self) -> None:
        """Initialize persistent storage for the agent."""
        self.context.logger.info("Initializing persistent storage...")

        store_path = self.context.params.store_path

        safe_contract_addresses = self.context.params.safe_contract_addresses
        if safe_contract_addresses is None or safe_contract_addresses == "":
            self.context.logger.error(
                f"No safe contract addresses found: {self.context.params.safe_contract_addresses}"
            )
            msg = "No safe contract addresses found"
            raise ValueError(msg)
        self.context.logger.info(f"Safe contract addresses: {safe_contract_addresses}")

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
            "portfolio_snapshot.json": {
                "timestamp": None,
                "available_capital_usdc": 0.0,
                "total_exposure": 0.0,
                "current_positions": 0,
                "total_positions": 0,
                "total_unrealized_pnl": 0.0,
                "current_portfolio_value": 0.0,
            },
        }

        for filename, default_content in files_to_initialize.items():
            file_path = store_path / filename
            if not file_path.exists():
                with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(default_content, f, indent=2)

        self.context.store_path = store_path
        self.context.logger.info(f"Persistent storage initialized at: {store_path}")

        coingecko_api_key = self.context.params.coingecko_api_key
        self.context.logger.info(f"Setting CoinGecko API key: {coingecko_api_key}")

    def _fetch_initial_portfolio_value_async(self) -> None:
        """Fetch initial portfolio value from USDC balance asynchronously."""
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
                self.usdc_balance = 0.0
                self.balance_received = True
                return

            usdc_addresses = {
                "base": self.context.params.base_usdc_address,
            }

            usdc_address = usdc_addresses.get(chain)
            if not usdc_address:
                self.context.logger.warning(f"No USDC address found for chain {chain}")
                self.usdc_balance = 0.0
                self.balance_received = True
                return

            # Submit async contract call for USDC balance
            self._submit_usdc_balance_request(chain, safe_address, usdc_address)
            self.balance_check_submitted = True

        except Exception as e:
            self.context.logger.exception(f"Failed to fetch initial portfolio value: {e}")
            self.usdc_balance = 0.0
            self.balance_received = True

    def _submit_usdc_balance_request(self, chain: str, safe_address: str, usdc_address: str) -> None:
        """Submit a contract call request for USDC balance."""
        try:
            self.context.logger.info(f"Requesting initial USDC balance for {safe_address} on {chain}")

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

            self.context.logger.debug(f"Submitted initial USDC balance request for {safe_address}")

        except Exception as e:
            self.context.logger.exception(f"Failed to submit USDC balance request: {e}")
            self.usdc_balance = 0.0
            self.balance_received = True

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
                        self.context.logger.info(f"Received initial USDC balance: {balance}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error during setup: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating USDC balance response during setup: {e}")
            return False

    def _check_balance_response(self) -> None:
        """Check if balance response has arrived and process it."""
        try:
            # Process any received responses
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    # Process USDC balance response
                    if "balance" in response:
                        balance_raw = response["balance"]
                        self.usdc_balance = float(balance_raw) / (10**6)
                        self.context.logger.info(f"Initial portfolio value: ${self.usdc_balance:.2f}")
                        self.balance_received = True

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # If we still have pending calls, keep waiting
            if self.pending_contract_calls and not self.balance_received:
                return  # Keep waiting for responses

            # If no pending calls or we have responses, mark as received
            if not self.pending_contract_calls and not self.balance_received:
                # No responses received, use fallback
                self.usdc_balance = 0.0
                self.balance_received = True
                self.context.logger.info("No balance response received, using fallback value of $0")

        except Exception as e:
            self.context.logger.exception(f"Error checking balance response: {e}")
            self.usdc_balance = 0.0
            self.balance_received = True

    def _save_initial_portfolio_snapshot(self) -> None:
        """Save initial portfolio snapshot to persistent storage."""
        if not self.context.store_path:
            return

        try:
            snapshot_file = self.context.store_path / "portfolio_snapshot.json"

            # Use the fetched balance, or default to 0 if not available
            portfolio_value = self.usdc_balance if self.usdc_balance is not None else 0.0

            snapshot_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "available_capital_usdc": portfolio_value,
                "total_exposure": 0.0,
                "current_positions": 0,
                "total_positions": 0,
                "total_unrealized_pnl": 0.0,
                "current_portfolio_value": portfolio_value,
            }

            with open(snapshot_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(snapshot_data, f, indent=2)

            self.context.logger.info(
                f"Saved initial portfolio snapshot: ${snapshot_data['current_portfolio_value']:.2f}"
            )

        except Exception as e:
            self.context.logger.exception(f"Failed to save initial portfolio snapshot: {e}")

    def act(self) -> None:
        """Perform the act."""
        try:
            # Initialize state on first call
            if not self.started:
                self.context.logger.info(f"Entering {self._state} state.")
                self._initialize_state()
                self.started = True

            # Submit balance check if not already submitted
            if not self.balance_check_submitted:
                self.context.logger.info("Fetching initial portfolio value...")
                self._fetch_initial_portfolio_value_async()
                return  # Exit early, let FSM cycle for async response

            # Check for balance response if submitted but not received
            if not self.balance_received:
                self._check_balance_response()
                if not self.balance_received:
                    return  # Still waiting for response, let FSM cycle

            # Save initial portfolio snapshot once balance is received
            self._save_initial_portfolio_snapshot()

            # Setup complete
            self.setup_success = True
            self._event = MindshareabciappEvents.DONE
            self._is_done = True

        except Exception as e:
            self.context.logger.exception(f"Setup failed. {e!s}")
            self.context.error_context = {
                "error_type": "setup_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True
