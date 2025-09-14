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
from packages.xiuxiuxar.skills.mindshare_app.dialogues import ContractApiDialogue
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
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def act(self) -> None:
        """Perform the act."""
        try:
            if not self.started:
                self.context.logger.info(f"Entering {self._state} state.")
                self.started = True

            # Initialize KPI checking on first call
            if not self.kpi_check_initialized:
                self._initialize_kpi_check()
                return  # Exit early, let FSM cycle for async response

            # Check if staking KPI check is complete
            if not self.staking_kpi_check_complete:
                self._check_contract_responses()
                if not self.staking_kpi_check_complete:
                    return  # Still waiting for response, let FSM cycle

            # If KPI is not met and we need vanity tx but haven't submitted request
            if (
                not self.is_staking_kpi_met
                and not self.vanity_tx_request_submitted
                and self._should_prepare_vanity_tx()
            ):
                self._prepare_vanity_tx_async()
                return

            # Check for vanity tx preparation responses
            if self.vanity_tx_request_submitted and not self.vanity_tx_prepared:
                self._check_vanity_tx_responses()
                if not self.vanity_tx_prepared:
                    return

            # If vanity tx prepared but not executed, execute it
            if self.vanity_tx_prepared and self.vanity_tx_hex and not self.vanity_tx_execution_submitted:
                self._execute_vanity_tx_async()
                return

            # Check for vanity tx execution responses
            if self.vanity_tx_execution_submitted and not self.vanity_tx_executed:
                self._check_vanity_tx_execution_responses()
                if not self.vanity_tx_executed:
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
                    nonce = message.state.body.get("safe_nonce") or message.state.body.get("nonce")
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
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating nonce response: {e}")
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

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

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
        """Evaluate if staking KPI is met based on current nonce."""
        try:
            # Get parameters for KPI evaluation
            staking_threshold_period = getattr(self.context.params, "staking_threshold_period", 22)  # periods
            min_num_of_safe_tx_required = getattr(self.context.params, "min_num_of_safe_tx_required", 5)  # transactions

            # Load persistent KPI state data
            kpi_state = self._load_kpi_state()
            period_count = kpi_state.get("period_count", 0)
            period_number_at_last_cp = kpi_state.get("period_number_at_last_cp", 0)
            last_checkpoint_nonce = kpi_state.get("last_checkpoint_nonce", 0)

            # Increment period count for this round
            period_count += 1

            self.context.logger.info(
                f"KPI Evaluation - Current nonce: {current_nonce}, "
                f"Last checkpoint nonce: {last_checkpoint_nonce}, "
                f"Period count: {period_count}, "
                f"Last CP period: {period_number_at_last_cp}"
            )

            # Check if we're past the threshold period
            is_period_threshold_exceeded = period_count - period_number_at_last_cp >= staking_threshold_period

            if not is_period_threshold_exceeded:
                self.context.logger.info("Period threshold not exceeded yet")
                self.is_staking_kpi_met = True  # KPI is considered met if not in evaluation period
            else:
                # Calculate transactions since last checkpoint
                multisig_nonces_since_last_cp = current_nonce - last_checkpoint_nonce

                self.context.logger.info(
                    f"Multisig transactions since last checkpoint: {multisig_nonces_since_last_cp}"
                )

                if multisig_nonces_since_last_cp >= min_num_of_safe_tx_required:
                    self.context.logger.info("Staking KPI already met!")
                    self.is_staking_kpi_met = True
                    # Update checkpoint data when KPI is met
                    period_number_at_last_cp = period_count
                    last_checkpoint_nonce = current_nonce
                else:
                    num_tx_left = min_num_of_safe_tx_required - multisig_nonces_since_last_cp
                    self.context.logger.info(f"Staking KPI not met. Need {num_tx_left} more transactions")
                    self.is_staking_kpi_met = False

            # Save updated KPI state
            updated_kpi_state = {
                "period_count": period_count,
                "period_number_at_last_cp": period_number_at_last_cp,
                "last_checkpoint_nonce": last_checkpoint_nonce,
                "current_nonce": current_nonce,
                "last_evaluation": datetime.now(UTC).isoformat(),
                "is_staking_kpi_met": self.is_staking_kpi_met,
            }
            self._save_kpi_state(updated_kpi_state)

            self.staking_kpi_check_complete = True

        except Exception as e:
            self.context.logger.exception(f"Error evaluating staking KPI: {e}")
            self.is_staking_kpi_met = False
            self.staking_kpi_check_complete = True

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
                # Extract relevant KPI fields
                return {
                    "period_count": state_data.get("period_count", 0),
                    "period_number_at_last_cp": state_data.get("period_number_at_last_cp", 0),
                    "last_checkpoint_nonce": state_data.get("last_checkpoint_nonce", 0),
                    "current_nonce": state_data.get("current_nonce", 0),
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
            kpi_state = {
                "is_staking_kpi_met": kmp_data.get("is_staking_kpi_met", False),
                "has_required_funds": self._check_agent_balance_threshold(),
                "period_count": kmp_data.get("period_count", 0),
                "period_number_at_last_cp": kmp_data.get("period_number_at_last_cp", 0),
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

    def _should_prepare_vanity_tx(self) -> bool:
        """Determine if we should prepare a vanity transaction."""
        # Only prepare vanity tx if KPI is not met and we're in evaluation period
        if self.is_staking_kpi_met:
            return False

        # Load current KPI state to check period threshold
        kpi_state = self._load_kpi_state()
        staking_threshold_period = getattr(self.context.params, "staking_threshold_period", 22)
        period_count = kpi_state.get("period_count", 0)
        period_number_at_last_cp = kpi_state.get("period_number_at_last_cp", 0)

        return period_count - period_number_at_last_cp >= staking_threshold_period

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
            tx_data = b"0x"
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
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating vanity tx response: {e}")
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

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

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

            staking_chain = getattr(self.context.params, "staking_chain", "ethereum")
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
                        "gas_price": 1,
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
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating vanity tx execution response: {e}")
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

                    # Remove from pending
                    self.pending_contract_calls.remove(dialogue)

            # If no pending calls, mark as executed
            if not self.pending_contract_calls and not self.vanity_tx_executed:
                self.vanity_tx_executed = True
                self.context.logger.info("Vanity transaction execution completed (no response)")

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

                self._broadcast_vanity_transaction()
                return True

            self.context.logger.error(f"Vanity transaction signing failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing vanity signing response: {e}")
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
                self.context.logger.info(f"Vanity transaction broadcast successful: {tx_hash}")

                kpi_state = self._load_kpi_state()
                kpi_state["vanity_tx_broadcast"] = True
                kpi_state["vanity_tx_final_hash"] = tx_hash
                kpi_state["vanity_tx_broadcast_timestamp"] = datetime.now(UTC).isoformat()
                self._save_kpi_state(kpi_state)

                return True

            self.context.logger.error(f"Vanity transaction broadcast failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing vanity broadcast response: {e}")
            return False

    def _finalize_kpi_check(self) -> None:
        """Finalize KPI check and determine transition."""
        self.context.logger.info("Finalizing staking KPI check...")

        if self.is_staking_kpi_met is None:
            self.context.logger.error("Staking KPI status unknown")
            self._event = MindshareabciappEvents.ERROR
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
