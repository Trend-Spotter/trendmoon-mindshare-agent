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

from eth_utils import to_bytes
from aea.protocols.base import Message
from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.open_aea.protocols.signing.message import SigningMessage
from packages.valory.contracts.gnosis_safe.contract import SafeOperation, GnosisSafeContract
from packages.valory.protocols.ledger_api.custom_types import Terms
from packages.xiuxiuxar.skills.mindshare_app.dialogues import LedgerApiDialogue, ContractApiDialogue
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    ETHER_VALUE,
    SAFE_TX_GAS,
    NULL_ADDRESS,
    LEDGER_API_ADDRESS,
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class CheckStakingKPIRound(BaseState):
    """This class implements the behaviour of the state CheckStakingKPIRound."""

    supported_protocols = {
        ContractApiMessage.protocol_id: [],
        LedgerApiMessage.protocol_id: [],
        SigningMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.CHECKSTAKINGKPIROUND
        self.started: bool = False
        self.kpi_check_initialized: bool = False
        self.vanity_tx_hex: str | None = None
        self.is_staking_kpi_met: bool | None = None
        self.pending_contract_calls: list[ContractApiDialogue] = []
        self.contract_responses: dict[str, Any] = {}
        self.staking_kpi_check_complete: bool = False
        self.vanity_tx_prepared: bool = False
        self.vanity_tx_request_submitted: bool = False
        self.vanity_tx_execution_submitted: bool = False
        self.vanity_tx_executed: bool = False
        self.vanity_tx_signing_submitted: bool = False
        self.vanity_tx_broadcast_submitted: bool = False
        self.vanity_tx_raw_data: bytes | None = None
        self.vanity_tx_signed_data: bytes | None = None
        self.vanity_tx_final_hash: str | None = None
        self.agent_balance_check_submitted: bool = False
        self.agent_balance: int | None = None
        self.has_required_funds: bool = False
        self.vanity_tx_broadcast_retries: int = 0
        self.vanity_tx_broadcast_failed: bool = False
        self.nonce_check_failed: bool = False
        self.vanity_tx_hash_failed: bool = False
        self.vanity_tx_execution_failed: bool = False
        self.vanity_tx_signing_failed: bool = False

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False
        self.started = False
        self.kpi_check_initialized = False
        self.vanity_tx_hex = None
        self.is_staking_kpi_met = None
        self.pending_contract_calls = []
        self.contract_responses = {}
        self.staking_kpi_check_complete = False
        self.vanity_tx_prepared = False
        self.vanity_tx_request_submitted = False
        self.vanity_tx_execution_submitted = False
        self.vanity_tx_executed = False
        self.vanity_tx_signing_submitted = False
        self.vanity_tx_broadcast_submitted = False
        self.vanity_tx_raw_data = None
        self.vanity_tx_signed_data = None
        self.vanity_tx_final_hash = None
        self.agent_balance_check_submitted = False
        self.agent_balance = None
        self.has_required_funds = False
        self.vanity_tx_broadcast_retries = 0
        self.vanity_tx_broadcast_failed = False
        self.vanity_tx_retry_needed = False
        self.nonce_check_failed = False
        self.vanity_tx_hash_failed = False
        self.vanity_tx_execution_failed = False
        self.vanity_tx_signing_failed = False
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def act(self) -> None:
        """Perform the act."""
        try:
            self._handle_startup()

            if self._handle_kpi_initialization():
                return

            if self._handle_staking_kpi_check():
                return

            if self._handle_balance_check():
                return

            if self._handle_vanity_tx_preparation():
                return

            if self._handle_vanity_tx_execution():
                return

            # All checks complete, finalize
            self._finalize_kpi_check()

        except Exception as e:
            self.context.logger.exception(f"CheckStakingKPIRound failed: {e}")
            self.context.error_context = {
                "error_type": "staking_kpi_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _handle_startup(self) -> None:
        """Handle startup logging."""
        if not self.started:
            self.context.logger.info(f"Entering {self._state} state.")
            self.started = True

    def _handle_kpi_initialization(self) -> bool:
        """Handle KPI initialization. Returns True if should exit early."""
        if not self.kpi_check_initialized:
            self._initialize_kpi_check()
            return True
        return False

    def _handle_staking_kpi_check(self) -> bool:
        """Handle staking KPI check. Returns True if should exit early."""
        if not self.staking_kpi_check_complete:
            self._check_contract_responses()
            return not self.staking_kpi_check_complete
        return False

    def _handle_balance_check(self) -> bool:
        """Handle agent balance check. Returns True if should exit early."""
        if not self.agent_balance_check_submitted:
            self._check_agent_balance_async()
            return True

        if self.agent_balance_check_submitted and self.agent_balance is None:
            self._check_balance_response()
            return self.agent_balance is None

        return False

    def _handle_vanity_tx_preparation(self) -> bool:
        """Handle vanity transaction preparation. Returns True if should exit early."""
        if not self.is_staking_kpi_met and not self.vanity_tx_request_submitted and self._should_prepare_vanity_tx():
            self._prepare_vanity_tx_async()
            return True

        if self.vanity_tx_request_submitted and not self.vanity_tx_prepared:
            self._check_vanity_tx_responses()
            return not self.vanity_tx_prepared

        return False

    def _handle_vanity_tx_execution(self) -> bool:  # noqa: PLR0911
        """Handle vanity transaction execution. Returns True if should exit early."""
        # Check if broadcast failed after max retries
        if self.vanity_tx_broadcast_failed:
            self.context.logger.error("Vanity transaction broadcast failed after max retries")
            self.context.error_context = {
                "error_type": "vanity_tx_broadcast_error",
                "error_message": (
                    f"Failed to broadcast vanity transaction after {self.vanity_tx_broadcast_retries} attempts"
                ),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True
            return True

        # Check if signing failed
        if self.vanity_tx_signing_failed:
            self.context.logger.error("Vanity transaction signing failed")
            self.context.error_context = {
                "error_type": "vanity_tx_signing_error",
                "error_message": "Failed to sign vanity transaction",
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True
            return True

        # Step 1: Submit the raw transaction request
        if self.vanity_tx_prepared and self.vanity_tx_hex and not self.vanity_tx_execution_submitted:
            self._execute_vanity_tx_async()
            return True

        # Step 2: Wait for raw transaction response and submit for signing
        if self.vanity_tx_execution_submitted and not self.vanity_tx_signing_submitted:
            self._check_vanity_tx_execution_responses()
            # Keep waiting - don't proceed until signing is submitted
            return True

        # Step 3: Wait for signing to complete (validation function handles this)
        if self.vanity_tx_signing_submitted and not self.vanity_tx_broadcast_submitted:
            # Signing validation function will trigger broadcast
            return True

        # Step 3.5: Handle broadcast retry after failure
        if self.vanity_tx_retry_needed and self.vanity_tx_signed_data:
            self.vanity_tx_retry_needed = False
            self._broadcast_vanity_transaction()
            return True

        # Step 4: Wait for broadcast to complete (validation function sets vanity_tx_executed)
        # Broadcast validation function will set vanity_tx_executed = True
        return self.vanity_tx_broadcast_submitted and not self.vanity_tx_executed

    def _initialize_kpi_check(self) -> None:
        """Initialize staking KPI check."""
        if self.kpi_check_initialized:
            return

        self.context.logger.info("Initializing staking KPI check...")

        # Submit async request to check if staking KPI is met
        self._check_staking_kpi_async()
        self.kpi_check_initialized = True

    def _check_staking_kpi_async(self) -> None:
        """Check if staking KPI is met asynchronously."""
        try:
            staking_chain = self.context.params.staking_chain
            safe_addresses = self.context.params.safe_contract_addresses

            if isinstance(safe_addresses, str):
                try:
                    safe_addresses = json.loads(safe_addresses)
                except json.JSONDecodeError:
                    self.context.logger.warning("Failed to parse safe_addresses")
                    safe_addresses = {}

            safe_address = safe_addresses.get(staking_chain)
            if not safe_address:
                self.context.logger.warning(f"No safe address found for staking chain {staking_chain}")
                self.is_staking_kpi_met = False
                self.staking_kpi_check_complete = True
                return

            # Submit contract call to get current nonce
            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_STATE,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=safe_address,
                contract_id=str(GnosisSafeContract.contract_id),
                callable="get_safe_nonce",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs({"chain_id": staking_chain}),
            )

            # Add validation function and metadata
            dialogue.validation_func = self._validate_nonce_response
            dialogue.safe_address = safe_address
            dialogue.chain = staking_chain

            # Track the pending call
            self.pending_contract_calls.append(dialogue)

            self.context.logger.info(f"Submitted nonce request for safe {safe_address}")

        except Exception as e:
            self.context.logger.exception(f"Failed to check staking KPI: {e}")
            self.is_staking_kpi_met = False
            self.staking_kpi_check_complete = True

    def _validate_nonce_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate nonce response message."""
        try:
            if message.performative == ContractApiMessage.Performative.STATE:
                if hasattr(message, "state") and message.state:
                    nonce = message.state.body.get("safe_nonce")
                    if nonce is None:
                        nonce = message.state.body.get("nonce")
                    if nonce is not None:
                        # Store the response
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "current_nonce": int(nonce),
                            "chain": dialogue.chain,
                            "safe_address": dialogue.safe_address,
                        }
                        self.context.logger.info(f"Received current nonce: {nonce}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.error(f"Nonce check failed - Contract API error: {message.message}")
                self.nonce_check_failed = True
                self.staking_kpi_check_complete = True
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating nonce response: {e}")
            self.nonce_check_failed = True
            self.staking_kpi_check_complete = True
            return False

    def _check_contract_responses(self) -> None:
        """Check if contract responses have arrived and process them."""
        try:
            # Process any received responses
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    # Process nonce response
                    if "current_nonce" in response:
                        current_nonce = response["current_nonce"]
                        self._evaluate_staking_kpi(current_nonce)
                        # Clear the response after processing
                        del self.contract_responses[request_nonce]

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # Check if nonce check failed
            if self.nonce_check_failed:
                self.context.logger.error("Nonce check failed, transitioning to error handling")
                self.context.error_context = {
                    "error_type": "nonce_check_error",
                    "error_message": "Failed to retrieve safe nonce from contract API",
                    "originating_round": str(self._state),
                }
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

            # If no pending calls, mark as complete
            if not self.pending_contract_calls and not self.staking_kpi_check_complete:
                # No responses received, use fallback
                self.is_staking_kpi_met = False
                self.staking_kpi_check_complete = True
                self.context.logger.info("No nonce response received, assuming KPI not met")

        except Exception as e:
            self.context.logger.exception(f"Error checking contract responses: {e}")
            self.is_staking_kpi_met = False
            self.staking_kpi_check_complete = True

    def _evaluate_staking_kpi(self, current_nonce: int) -> None:
        """Evaluate if staking KPI is met based on current nonce using period-based tracking."""
        try:
            # Load state first
            kpi_state = self._load_kpi_state()

            # Get parameters (includes contract-calculated min_tx from state)
            params = self._get_kpi_parameters(kpi_state)

            # Handle migration if needed
            kpi_state = self._migrate_kpi_state_if_needed(kpi_state, current_nonce, params)

            # Get checkpoint data and validate
            checkpoint_data = self._get_validated_checkpoint_data(kpi_state, current_nonce)

            # Evaluate KPI based on grace period and transactions
            self._perform_kpi_evaluation(checkpoint_data, current_nonce, params)

            # Save updated state
            self._save_evaluation_result(checkpoint_data, current_nonce)
            self.staking_kpi_check_complete = True

        except Exception as e:
            self.context.logger.exception(f"Error evaluating staking KPI: {e}")
            self.is_staking_kpi_met = False
            self.staking_kpi_check_complete = True

    def _get_kpi_parameters(self, kpi_state: dict[str, Any]) -> dict[str, int]:
        """Get KPI evaluation parameters.

        Args:
        ----
            kpi_state: Loaded KPI state from state.json

        Returns:
        -------
            Dict with staking_threshold_period and min_num_of_safe_tx_required

        """
        # Get staking_threshold_period from config
        # This is a grace period (in FSM cycles) to allow organic transactions before
        # we start checking KPI and potentially sending vanity transactions.
        # After the grace period expires, we check KPI every cycle and send vanity tx if needed.
        staking_threshold_period = getattr(self.context.params, "staking_threshold_period", 22)

        # Try to get min_tx_required from state.json (contract-calculated value)
        min_tx_required = kpi_state.get("min_num_of_safe_tx_required")

        if min_tx_required is None:
            # Fallback to hardcoded config param
            min_tx_required = getattr(self.context.params, "min_num_of_safe_tx_required", 5)
            self.context.logger.warning(
                f"Using hardcoded min_num_of_safe_tx_required from config: {min_tx_required} "
                "(contract calculation not available in state.json)"
            )
        else:
            self.context.logger.info(
                f"Using contract-calculated min_num_of_safe_tx_required from state: {min_tx_required}"
            )

        return {
            "staking_threshold_period": staking_threshold_period,
            "min_num_of_safe_tx_required": min_tx_required,
        }

    def _migrate_kpi_state_if_needed(
        self, kpi_state: dict[str, Any], current_nonce: int, params: dict[str, int]
    ) -> dict[str, Any]:
        """Migrate KPI state from v1/v2/v3 to v4 if necessary."""
        state_version = kpi_state.get("state_version", 1)
        if state_version >= 4:
            return kpi_state

        self.context.logger.warning(f"Migrating KPI state from v{state_version} to v4 (period-based)")

        # Conservative: Give fresh grace period during migration
        checkpoint_nonce = kpi_state.get("last_checkpoint_nonce") or current_nonce

        if checkpoint_nonce == current_nonce:
            self.context.logger.info(f"Migration: setting checkpoint nonce to current: {current_nonce}")
        else:
            self.context.logger.info(f"Migration: preserving existing checkpoint nonce: {checkpoint_nonce}")

        # Create v4 state (period-based)
        migrated_state = {
            "state_version": 4,
            "period_count": kpi_state.get("period_count", 0),
            "period_number_at_last_cp": 0,  # Fresh grace period
            "last_checkpoint_nonce": checkpoint_nonce,
        }
        self._save_kpi_state(migrated_state)

        staking_threshold_period = params["staking_threshold_period"]
        self.context.logger.info(f"KPI state migrated to v4. Fresh grace period: {staking_threshold_period} FSM cycles")
        return migrated_state

    def _get_validated_checkpoint_data(self, kpi_state: dict[str, Any], current_nonce: int) -> dict[str, int]:
        """Get and validate checkpoint tracking data."""
        checkpoint_nonce = kpi_state.get("last_checkpoint_nonce", 0)
        period_count = kpi_state.get("period_count", 0)
        period_number_at_last_cp = kpi_state.get("period_number_at_last_cp", 0)

        # Handle edge case: checkpoint nonce higher than current (stale/corrupt state)
        if checkpoint_nonce > current_nonce:
            self.context.logger.warning(
                f"Checkpoint nonce ({checkpoint_nonce}) > current ({current_nonce}). Resetting."
            )
            checkpoint_nonce = current_nonce

        return {
            "last_checkpoint_nonce": checkpoint_nonce,
            "period_count": period_count,
            "period_number_at_last_cp": period_number_at_last_cp,
        }

    def _perform_kpi_evaluation(
        self, checkpoint_data: dict[str, int], current_nonce: int, params: dict[str, int]
    ) -> None:
        """Perform KPI evaluation based on period-based grace period and transaction count."""
        period_count = checkpoint_data["period_count"]
        period_number_at_last_cp = checkpoint_data["period_number_at_last_cp"]
        staking_threshold_period = params["staking_threshold_period"]

        periods_elapsed = period_count - period_number_at_last_cp

        self.context.logger.info(
            f"KPI Evaluation - nonce: {current_nonce}, checkpoint_nonce: "
            f"{checkpoint_data['last_checkpoint_nonce']}, periods_elapsed: {periods_elapsed}, "
            f"grace_threshold: {staking_threshold_period} FSM cycles"
        )

        # Period-based grace check (matching Valory reference)
        if periods_elapsed < staking_threshold_period:
            remaining_periods = staking_threshold_period - periods_elapsed
            self.context.logger.info(
                f"Grace period active ({periods_elapsed}/{staking_threshold_period} FSM cycles, "
                f"{remaining_periods} cycles remaining). KPI check skipped."
            )
            self.is_staking_kpi_met = None
        else:
            tx_since_checkpoint = current_nonce - checkpoint_data["last_checkpoint_nonce"]
            min_required = params["min_num_of_safe_tx_required"]

            self.context.logger.info(
                f"Evaluation period active. Txs since checkpoint: {tx_since_checkpoint}, required: {min_required}"
            )

            if tx_since_checkpoint >= min_required:
                self.context.logger.info(f"Staking KPI met! {tx_since_checkpoint} >= {min_required} transactions")
                self.is_staking_kpi_met = True
            else:
                self.context.logger.info(f"KPI not met. Need {min_required - tx_since_checkpoint} more txs")
                self.is_staking_kpi_met = False

    def _save_evaluation_result(self, checkpoint_data: dict[str, int], current_nonce: int) -> None:
        """Save KPI evaluation result to state."""
        updated_state = {
            "state_version": 4,
            "period_count": checkpoint_data["period_count"],
            "period_number_at_last_cp": checkpoint_data["period_number_at_last_cp"],
            "last_checkpoint_nonce": checkpoint_data["last_checkpoint_nonce"],
            "current_nonce": current_nonce,
            "last_evaluation": datetime.now(UTC).isoformat(),
            "is_staking_kpi_met": self.is_staking_kpi_met,
        }
        self._save_kpi_state(updated_state)

    def _load_kpi_state(self) -> dict[str, Any]:
        """Load KPI state from state.json."""
        if not self.context.store_path:
            return {}

        state_file = self.context.store_path / "state.json"
        if not state_file.exists():
            return {}

        try:
            with open(state_file, encoding=DEFAULT_ENCODING) as f:
                state_data = json.load(f)
                # Extract relevant KPI fields (v4 period-based + legacy for migration)
                return {
                    "state_version": state_data.get("state_version", 1),
                    # V4 fields (period-based)
                    "period_count": state_data.get("period_count", 0),
                    "period_number_at_last_cp": state_data.get("period_number_at_last_cp", 0),
                    # Common fields
                    "last_checkpoint_nonce": state_data.get("last_checkpoint_nonce", 0),
                    "current_nonce": state_data.get("current_nonce", 0),
                    "min_num_of_safe_tx_required": state_data.get("min_num_of_safe_tx_required"),  # From contract
                    "last_evaluation": state_data.get("last_evaluation"),
                    "is_staking_kpi_met": state_data.get("is_staking_kpi_met", False),
                    "vanity_tx_prepared": state_data.get("vanity_tx_prepared", False),
                    "vanity_tx_hash": state_data.get("vanity_tx_hash"),
                    "vanity_tx_timestamp": state_data.get("vanity_tx_timestamp"),
                    "vanity_tx_broadcast": state_data.get("vanity_tx_broadcast", False),
                    "vanity_tx_final_hash": state_data.get("vanity_tx_final_hash"),
                    "vanity_tx_broadcast_timestamp": state_data.get("vanity_tx_broadcast_timestamp"),
                }
        except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as e:
            self.context.logger.warning(f"Failed to load KPI state: {e}")
            return {}

    def _save_kpi_state(self, kmp_data: dict[str, Any]) -> None:
        """Save KPI state to state.json."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available, cannot save KPI state")
            return

        state_file = self.context.store_path / "state.json"
        try:
            # Load existing state
            state_data = {}
            if state_file.exists():
                with open(state_file, encoding=DEFAULT_ENCODING) as f:
                    state_data = json.load(f)

            # Update KPI data in state
            # If has_required_funds is explicitly provided in kmp_data, use that value
            # Otherwise, fall back to checking the current instance variable
            kpi_state = {
                "state_version": kmp_data.get("state_version", 4),  # Default to v4
                "is_staking_kpi_met": kmp_data.get("is_staking_kpi_met", False),
                "has_required_funds": kmp_data.get("has_required_funds", self._check_agent_balance_threshold()),
                # V4 fields (period-based)
                "period_count": kmp_data.get("period_count", 0),
                "period_number_at_last_cp": kmp_data.get("period_number_at_last_cp", 0),
                # Common fields
                "last_checkpoint_nonce": kmp_data.get("last_checkpoint_nonce", 0),
                "current_nonce": kmp_data.get("current_nonce", 0),
                "last_evaluation": kmp_data.get("last_evaluation"),
                "vanity_tx_prepared": kmp_data.get("vanity_tx_prepared", False),
                "vanity_tx_hash": kmp_data.get("vanity_tx_hash"),
                "vanity_tx_timestamp": kmp_data.get("vanity_tx_timestamp"),
                "vanity_tx_broadcast": kmp_data.get("vanity_tx_broadcast", False),
                "vanity_tx_final_hash": kmp_data.get("vanity_tx_final_hash"),
                "vanity_tx_broadcast_timestamp": kmp_data.get("vanity_tx_broadcast_timestamp"),
            }

            state_data.update(kpi_state)

            # Save updated state
            with open(state_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(state_data, f, indent=2)

            self.context.logger.debug(f"Saved KPI state to state.json: {kpi_state}")

        except (PermissionError, OSError, json.JSONDecodeError) as e:
            self.context.logger.warning(f"Failed to save KPI state to state.json: {e}")

    def _check_agent_balance_threshold(self) -> bool:
        """Check if agent's ETH balance is above the threshold."""
        return self.has_required_funds

    def _check_agent_balance_async(self) -> None:
        """Check agent's ETH balance asynchronously."""
        try:
            self.context.logger.info("Checking agent balance...")

            agent_address = self.context.agent_address

            dialogue = self.submit_msg(
                performative=LedgerApiMessage.Performative.GET_BALANCE,
                connection_id=LEDGER_API_ADDRESS,
                ledger_id="ethereum",  # Use ethereum ledger for base chain, since LedgerAPI can't set "base" chain
                address=agent_address,
            )

            dialogue.validation_func = self._validate_balance_response
            self.agent_balance_check_submitted = True
            self.context.logger.debug(f"Submitted balance check for address: {agent_address}")

        except Exception as e:
            self.context.logger.exception(f"Failed to submit balance check: {e}")
            self.agent_balance_check_submitted = True
            self.agent_balance = 0
            self.has_required_funds = False

    def _validate_balance_response(self, message: LedgerApiMessage, _dialogue: LedgerApiDialogue) -> bool:
        """Validate balance response."""
        try:
            if message.performative == LedgerApiMessage.Performative.BALANCE:
                balance = int(message.balance)
                self.agent_balance = balance
                threshold = self.context.params.agent_balance_threshold
                self.has_required_funds = balance >= threshold

                self.context.logger.info(
                    f"Agent balance: {balance} wei ({balance / 1e18:.4f} ETH), "
                    f"threshold: {threshold} wei ({threshold / 1e18:.4f} ETH), "
                    f"sufficient: {self.has_required_funds}"
                )

                # Persist the balance check result to state
                kpi_state = self._load_kpi_state()
                kpi_state["has_required_funds"] = self.has_required_funds
                self._save_kpi_state(kpi_state)

                return True

            if message.performative == LedgerApiMessage.Performative.ERROR:
                self.context.logger.error(f"Error checking balance: {message.message}")
                self.agent_balance = 0
                self.has_required_funds = False
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating balance response: {e}")
            self.agent_balance = 0
            self.has_required_funds = False
            return False

    def _check_balance_response(self) -> None:
        """Check if balance response has been received."""
        try:
            ledger_messages = self.supported_protocols.get(LedgerApiMessage.protocol_id, [])

            if ledger_messages:
                # The validation function (_validate_balance_response) will have already
                # set self.agent_balance and self.has_required_funds
                # If we got a response, agent_balance should be set
                if self.agent_balance is not None:
                    self.context.logger.debug(
                        f"Balance check complete: {self.agent_balance} wei, "
                        f"has_required_funds: {self.has_required_funds}"
                    )
                    # Update KPI state with the balance check result
                    kpi_state = self._load_kpi_state()
                    kpi_state["has_required_funds"] = self.has_required_funds
                    self._save_kpi_state(kpi_state)
                else:
                    self.context.logger.warning("Received ledger message but balance not set")

                    self.agent_balance = 0
                    self.has_required_funds = False

        except Exception as e:
            self.context.logger.exception(f"Error checking balance response: {e}")
            # Set safe defaults
            self.agent_balance = 0
            self.has_required_funds = False

    def _should_prepare_vanity_tx(self) -> bool:
        """Determine if we should prepare a vanity transaction."""
        # Only prepare vanity tx if:
        # 1. KPI is explicitly False (not met)
        # 2. We're past the grace period threshold
        # 3. Agent has sufficient funds

        if self.is_staking_kpi_met is None:
            # Grace period - don't prepare vanity tx
            self.context.logger.debug("Grace period active - skipping vanity tx preparation")
            return False

        if self.is_staking_kpi_met:
            # KPI already met - no vanity tx needed
            self.context.logger.debug("KPI already met - no vanity tx needed")
            return False

        if not self.has_required_funds:
            # Insufficient funds - don't prepare vanity tx
            self.context.logger.warning("Insufficient agent funds to prepare vanity transaction")
            return False

        # Load current KPI state to check period threshold
        kpi_state = self._load_kpi_state()
        staking_threshold_period = getattr(self.context.params, "staking_threshold_period", 5)
        period_count = kpi_state.get("period_count", 0)
        period_number_at_last_cp = kpi_state.get("period_number_at_last_cp", 0)

        is_past_grace_period = period_count - period_number_at_last_cp >= staking_threshold_period

        if not is_past_grace_period:
            self.context.logger.debug(
                f"Still within grace period ({period_count - period_number_at_last_cp}/{staking_threshold_period}) "
                "- skipping vanity tx preparation"
            )
            return False

        self.context.logger.info(
            f"Should prepare vanity tx: KPI not met, past grace period, sufficient funds "
            f"(period {period_count}, checkpoint at {period_number_at_last_cp})"
        )
        return True

    def _prepare_vanity_tx_async(self) -> None:
        """Prepare vanity transaction asynchronously."""
        try:
            self.context.logger.info("Preparing vanity transaction...")

            staking_chain = self.context.params.staking_chain
            safe_addresses = self.context.params.safe_contract_addresses

            if isinstance(safe_addresses, str):
                try:
                    safe_addresses = json.loads(safe_addresses)
                except json.JSONDecodeError:
                    safe_addresses = {}

            safe_address = safe_addresses.get(staking_chain)
            if not safe_address:
                self.context.logger.warning("No safe address for vanity transaction")
                self.vanity_tx_prepared = True
                return

            self.context.logger.debug(f"Safe address for chain {staking_chain}: {safe_address}")

            # Prepare vanity transaction data
            tx_data = to_bytes(text="0x")
            self.context.logger.debug(f"Transaction data: {tx_data}")

            # Submit contract call to get safe transaction hash
            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=safe_address,
                contract_id=str(GnosisSafeContract.contract_id),
                callable="get_raw_safe_transaction_hash",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs(
                    {
                        "to_address": NULL_ADDRESS,
                        "value": ETHER_VALUE,
                        "data": tx_data,
                        "operation": SafeOperation.CALL.value,
                        "safe_tx_gas": SAFE_TX_GAS,
                        "chain_id": staking_chain,
                    }
                ),
            )

            # Add validation function and metadata
            dialogue.validation_func = self._validate_vanity_tx_response
            dialogue.safe_address = safe_address
            dialogue.chain = staking_chain
            dialogue.tx_data = tx_data

            # Track the pending call
            self.pending_contract_calls.append(dialogue)
            self.vanity_tx_request_submitted = True

            self.context.logger.info("Submitted vanity transaction hash request")

        except Exception as e:
            self.context.logger.exception(f"Failed to prepare vanity transaction: {e}")
            self.vanity_tx_prepared = True
            self.vanity_tx_request_submitted = True

    def _validate_vanity_tx_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate vanity transaction response message."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                if hasattr(message, "raw_transaction") and message.raw_transaction:
                    safe_tx_hash = message.raw_transaction.body.get("tx_hash")
                    if safe_tx_hash:
                        # Store the response
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "safe_tx_hash": safe_tx_hash,
                            "chain": dialogue.chain,
                            "safe_address": dialogue.safe_address,
                            "tx_data": dialogue.tx_data,
                        }
                        self.context.logger.info(f"Received safe transaction hash: {safe_tx_hash}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.error(f"Vanity tx hash request failed - Contract API error: {message.message}")
                self.vanity_tx_hash_failed = True
                self.vanity_tx_prepared = True
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating vanity tx response: {e}")
            self.vanity_tx_hash_failed = True
            self.vanity_tx_prepared = True
            return False

    def _check_vanity_tx_responses(self) -> None:
        """Check if vanity transaction responses have arrived and process them."""
        try:
            # Process any received responses
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    # Process vanity tx response
                    if "safe_tx_hash" in response:
                        safe_tx_hash = response["safe_tx_hash"]
                        tx_data = response["tx_data"]
                        self._finalize_vanity_tx(safe_tx_hash, tx_data)
                        # Clear the response after processing
                        del self.contract_responses[request_nonce]

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # Check if vanity tx hash request failed
            if self.vanity_tx_hash_failed:
                self.context.logger.error("Vanity tx hash request failed, transitioning to error handling")
                self.context.error_context = {
                    "error_type": "vanity_tx_hash_error",
                    "error_message": "Failed to retrieve safe transaction hash from contract API",
                    "originating_round": str(self._state),
                }
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

            # If no pending calls, mark as prepared
            if not self.pending_contract_calls and not self.vanity_tx_prepared:
                self.vanity_tx_prepared = True
                self.context.logger.info("Vanity transaction preparation completed")

        except Exception as e:
            self.context.logger.exception(f"Error checking vanity tx responses: {e}")
            self.vanity_tx_prepared = True

    def _finalize_vanity_tx(self, safe_tx_hash: str, _tx_data: bytes) -> None:
        """Finalize vanity transaction by creating the final hash."""
        try:
            # Remove '0x' prefix if present
            safe_tx_hash = safe_tx_hash.removeprefix("0x")

            # Create final transaction hash (this would normally use hash_payload_to_hex)
            # For now, we'll use the safe_tx_hash as the vanity tx hex
            self.vanity_tx_hex = f"0x{safe_tx_hash}"

            self.context.logger.info(f"Vanity transaction hash prepared: {self.vanity_tx_hex}")
            self.vanity_tx_prepared = True

        except Exception as e:
            self.context.logger.exception(f"Error finalizing vanity transaction: {e}")
            self.vanity_tx_prepared = True

    def _execute_vanity_tx_async(self) -> None:
        """Execute the vanity transaction on Gnosis Safe asynchronously."""
        try:
            self.context.logger.info("Executing vanity transaction...")

            staking_chain = getattr(self.context.params, "staking_chain", "base")
            safe_addresses = getattr(self.context.params, "safe_contract_addresses", {})

            if isinstance(safe_addresses, str):
                try:
                    safe_addresses = json.loads(safe_addresses)
                except json.JSONDecodeError:
                    safe_addresses = {}

            safe_address = safe_addresses.get(staking_chain)
            if not safe_address:
                self.context.logger.warning("No safe address for vanity transaction execution")
                self.vanity_tx_executed = True
                self.vanity_tx_execution_submitted = True
                return

            # Prepare transaction data for execution
            tx_data = to_bytes(text="0x")

            # Submit contract call to execute the Safe transaction
            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=safe_address,
                contract_id=str(GnosisSafeContract.contract_id),
                callable="get_raw_safe_transaction",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs(
                    {
                        "sender_address": self.context.agent_address,
                        "owners": (self.context.agent_address,),
                        "to_address": NULL_ADDRESS,
                        "value": ETHER_VALUE,
                        "data": tx_data,
                        "signatures_by_owner": {self.context.agent_address: self._get_preapproved_signature()},
                        "operation": SafeOperation.CALL.value,
                        "safe_tx_gas": SAFE_TX_GAS,
                        "base_gas": 0,
                        "gas_token": NULL_ADDRESS,
                        "refund_receiver": NULL_ADDRESS,
                    }
                ),
            )

            # Add validation function and metadata
            dialogue.validation_func = self._validate_vanity_tx_execution_response
            dialogue.safe_address = safe_address
            dialogue.chain = staking_chain
            dialogue.vanity_tx_hex = self.vanity_tx_hex

            # Track the pending call
            self.pending_contract_calls.append(dialogue)
            self.vanity_tx_execution_submitted = True

            self.context.logger.info(f"Submitted vanity transaction for execution: {self.vanity_tx_hex}")

        except Exception as e:
            self.context.logger.exception(f"Failed to execute vanity transaction: {e}")
            self.vanity_tx_executed = True
            self.vanity_tx_execution_submitted = True

    def _validate_vanity_tx_execution_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate vanity transaction execution response message."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                self.context.logger.info(f"Raw transaction: {message}")
                if hasattr(message, "raw_transaction") and message.raw_transaction:
                    raw_tx = message.raw_transaction
                    if raw_tx:
                        # Store the response - the raw transaction can be submitted to the network
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "execution_tx_data": raw_tx,
                            "chain": dialogue.chain,
                            "safe_address": dialogue.safe_address,
                            "vanity_tx_hex": dialogue.vanity_tx_hex,
                        }
                        self.context.logger.info("Vanity transaction raw data prepared for execution")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.error(f"Vanity tx execution request failed - Contract API error: {message.message}")
                self.vanity_tx_execution_failed = True
                return True

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating vanity tx execution response: {e}")
            self.vanity_tx_execution_failed = True
            return False

    def _check_vanity_tx_execution_responses(self) -> None:
        """Check if vanity transaction execution responses have arrived and process them."""
        try:
            # Process any received responses
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    # Process execution response
                    if "execution_tx_data" in response:
                        execution_tx_data = response["execution_tx_data"]
                        vanity_tx_hex = response["vanity_tx_hex"]
                        self.context.logger.info("Vanity transaction raw data received, ready for signing")

                        # Store transaction data for signing and broadcasting
                        self.vanity_tx_raw_data = execution_tx_data
                        self.vanity_tx_hex = vanity_tx_hex

                        # Sign the transaction
                        self._sign_vanity_transaction(execution_tx_data)

                        # Update KPI state to record successful preparation
                        kpi_state = self._load_kpi_state()
                        kpi_state["vanity_tx_prepared"] = True
                        kpi_state["vanity_tx_hash"] = vanity_tx_hex
                        kpi_state["vanity_tx_timestamp"] = datetime.now(UTC).isoformat()
                        self._save_kpi_state(kpi_state)

                        # Clear the response after processing
                        del self.contract_responses[request_nonce]

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # Check if vanity tx execution request failed
            if self.vanity_tx_execution_failed:
                self.context.logger.error("Vanity tx execution request failed, transitioning to error handling")
                self.context.error_context = {
                    "error_type": "vanity_tx_execution_error",
                    "error_message": "Failed to get raw safe transaction from contract API",
                    "originating_round": str(self._state),
                }
                self._event = MindshareabciappEvents.ERROR
                self._is_done = True
                return

        except Exception as e:
            self.context.logger.exception(f"Error checking vanity tx execution responses: {e}")
            self.vanity_tx_executed = True

    def _sign_vanity_transaction(self, raw_tx) -> None:
        """Sign the vanity transaction."""
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

        signing_msg, signing_dialogue = self.context.signing_dialogues.create(
            counterparty=self.context.decision_maker_address,
            performative=SigningMessage.Performative.SIGN_TRANSACTION,
            raw_transaction=raw_tx,
            terms=terms,
        )

        signing_dialogue.validation_func = self._validate_vanity_signing_response

        request_nonce = signing_dialogue.dialogue_label.dialogue_reference[0]
        self.context.requests.request_id_to_callback[request_nonce] = self.get_dialogue_callback_request()

        self.context.decision_maker_message_queue.put_nowait(signing_msg)

        self.vanity_tx_signing_submitted = True
        self.context.logger.info("Vanity transaction sent for signing")

    def _validate_vanity_signing_response(self, message: SigningMessage, _dialogue) -> bool:
        """Process vanity transaction signing response."""
        try:
            if message.performative == SigningMessage.Performative.SIGNED_TRANSACTION:
                self.vanity_tx_signed_data = message.signed_transaction
                self.context.logger.info("Vanity transaction signed successfully")

                # Immediately broadcast the signed transaction
                self._broadcast_vanity_transaction()
                return True

            if message.performative == SigningMessage.Performative.ERROR:
                error_code = message.error_code if hasattr(message, "error_code") else "unknown"
                self.context.logger.error(f"Vanity transaction signing failed with error: {error_code}")
                self.vanity_tx_signing_failed = True
                return True

            # Log unexpected performatives but don't fail validation
            self.context.logger.debug(
                f"Received unexpected signing message performative: {message.performative}, "
                "waiting for SIGNED_TRANSACTION"
            )
            return True  # Return True to avoid validation warning for intermediate messages

        except Exception as e:
            self.context.logger.exception(f"Error processing vanity signing response: {e}")
            self.vanity_tx_signing_failed = True
            return False

    def _broadcast_vanity_transaction(self) -> None:
        """Broadcast the signed vanity transaction."""
        signed_tx = self.vanity_tx_signed_data

        dialogue = self.submit_msg(
            performative=LedgerApiMessage.Performative.SEND_SIGNED_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            signed_transaction=signed_tx,
            kwargs=LedgerApiMessage.Kwargs({"ledger_id": "ethereum"}),
        )

        dialogue.validation_func = self._validate_vanity_broadcast_response
        self.vanity_tx_broadcast_submitted = True

        self.context.logger.info("Broadcasting vanity transaction to chain")

    def _validate_vanity_broadcast_response(self, message: LedgerApiMessage, _dialogue) -> bool:
        """Process vanity transaction broadcast response."""
        try:
            if message.performative == LedgerApiMessage.Performative.TRANSACTION_DIGEST:
                tx_hash = message.transaction_digest.body
                self.vanity_tx_final_hash = tx_hash
                self.context.logger.info(f"✅ Vanity transaction broadcast successful: {tx_hash}")

                # Update KPI state with broadcast information
                kpi_state = self._load_kpi_state()
                kpi_state["vanity_tx_broadcast"] = True
                kpi_state["vanity_tx_final_hash"] = tx_hash
                kpi_state["vanity_tx_broadcast_timestamp"] = datetime.now(UTC).isoformat()

                # Update nonce tracking for vanity tx
                # NOTE: Do NOT reset period_number_at_last_cp here!
                # Only checkpoint transactions (in call_checkpoint.py) should reset the grace period.
                # Vanity transactions only update the nonce counter.
                current_nonce = kpi_state.get("current_nonce", 0)
                # Vanity transaction increments nonce by 1
                checkpoint_nonce = current_nonce + 1

                # V4 fields (period-based)
                kpi_state["last_checkpoint_nonce"] = checkpoint_nonce
                # IMPORTANT: period_number_at_last_cp is NOT updated here (only for checkpoint tx)

                self.context.logger.info(
                    f"Nonce updated after successful vanity tx broadcast: "
                    f"last_checkpoint_nonce={checkpoint_nonce} (incremented from {current_nonce}). "
                    f"Grace period NOT reset (only checkpoint tx resets grace period)."
                )

                self._save_kpi_state(kpi_state)

                # Mark as executed so we can proceed to finalization
                self.vanity_tx_executed = True

                return True

            if message.performative == LedgerApiMessage.Performative.ERROR:
                error_msg = message.message
                self.context.logger.error(f"Vanity transaction broadcast failed: {error_msg}")

                # Handle "already known" error - transaction was already broadcast
                if isinstance(error_msg, dict) and error_msg.get("message") == "already known":
                    self.context.logger.info(
                        "Transaction already broadcast to chain, treating as successful. "
                        f"Using prepared hash: {self.vanity_tx_hex}"
                    )

                    # Update KPI state with the prepared hash
                    kpi_state = self._load_kpi_state()
                    kpi_state["vanity_tx_broadcast"] = True
                    kpi_state["vanity_tx_final_hash"] = self.vanity_tx_hex
                    kpi_state["vanity_tx_broadcast_timestamp"] = datetime.now(UTC).isoformat()
                    self._save_kpi_state(kpi_state)

                    # Mark as executed so we can proceed to finalization
                    self.vanity_tx_final_hash = self.vanity_tx_hex
                    self.vanity_tx_executed = True

                    return True

                # Handle retryable errors
                self.vanity_tx_broadcast_retries += 1

                if self.vanity_tx_broadcast_retries < 3:
                    # Retry - set flag to trigger rebroadcast on next act() cycle
                    self.context.logger.warning(
                        f"Vanity transaction broadcast failed (attempt {self.vanity_tx_broadcast_retries}/3). "
                        f"Error: {error_msg}. Retrying..."
                    )
                    # Don't reset state flags - this was causing infinite loop
                    # Instead, set retry flag to trigger rebroadcast in act() flow
                    self.vanity_tx_retry_needed = True
                    return True  # Return True to avoid validation warning
                # Max retries exceeded
                self.context.logger.error(
                    f"Vanity transaction broadcast failed after {self.vanity_tx_broadcast_retries} attempts. "
                    f"Final error: {error_msg}"
                )
                self.vanity_tx_broadcast_failed = True
                return False

            # Log unexpected performatives but don't fail validation
            self.context.logger.debug(
                f"Received unexpected broadcast message performative: {message.performative}, "
                "waiting for TRANSACTION_DIGEST"
            )
            return True

        except Exception as e:
            self.context.logger.exception(f"Error processing vanity broadcast response: {e}")
            return False

    def _finalize_kpi_check(self) -> None:
        """Finalize KPI check and determine transition."""
        self.context.logger.info("Finalizing staking KPI check...")

        if self.is_staking_kpi_met is None:
            self.context.logger.info("Grace period active - no KPI evaluation required")
            self._event = MindshareabciappEvents.DONE
        elif self.is_staking_kpi_met:
            self.context.logger.info("Staking KPI is met")
            self._event = MindshareabciappEvents.DONE
        else:
            if self.vanity_tx_hex:
                self.context.logger.info(f"Staking KPI not met, prepared vanity transaction: {self.vanity_tx_hex}")
                # Store vanity tx for potential execution
                self.context.vanity_tx_hex = self.vanity_tx_hex
            else:
                self.context.logger.info("Staking KPI not met, no vanity transaction prepared")

            self._event = MindshareabciappEvents.DONE

        self._is_done = True
