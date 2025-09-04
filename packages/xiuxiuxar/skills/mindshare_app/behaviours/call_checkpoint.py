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
from enum import Enum
from typing import Any
from pathlib import Path
from datetime import UTC, datetime

from aea.protocols.base import Message
from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.open_aea.protocols.signing.message import SigningMessage
from packages.valory.contracts.staking_token.contract import StakingTokenContract
from packages.valory.protocols.ledger_api.custom_types import Terms
from packages.xiuxiuxar.contracts.gnosis_safe.contract import SafeOperation, GnosisSafeContract
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


class StakingState(Enum):
    """Staking state enumeration for the staking."""

    UNSTAKED = 0
    STAKED = 1
    EVICTED = 2


class CallCheckpointRound(BaseState):
    """This class implements the behaviour of the state CallCheckpointRound."""

    supported_protocols = {
        ContractApiMessage.protocol_id: [],
        LedgerApiMessage.protocol_id: [],
        SigningMessage.protocol_id: [],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.CALLCHECKPOINTROUND
        self.started: bool = False
        self.checkpoint_initialized: bool = False
        self.service_staking_state: StakingState | None = None
        self.checkpoint_tx_hex: str | None = None
        self.min_num_of_safe_tx_required: int | None = None
        self.pending_contract_calls: list[ContractApiDialogue] = []
        self.contract_responses: dict[str, Any] = {}
        self.staking_state_check_complete: bool = False
        self.next_checkpoint_ts: int | None = None
        self.is_checkpoint_reached: bool = False
        self.checkpoint_preparation_complete: bool = False
        self.safe_tx_hash_prepared: bool = False
        self.checkpoint_tx_signing_submitted: bool = False
        self.checkpoint_tx_broadcast_submitted: bool = False
        self.checkpoint_tx_executed: bool = False
        self.checkpoint_tx_raw_data: bytes | None = None
        self.checkpoint_tx_signed_data: bytes | None = None
        self.checkpoint_tx_final_hash: str | None = None

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False
        self.started = False
        self.checkpoint_initialized = False
        self.service_staking_state = None
        self.checkpoint_tx_hex = None
        self.min_num_of_safe_tx_required = None
        self.pending_contract_calls = []
        self.contract_responses = {}
        self.staking_state_check_complete = False
        self.next_checkpoint_ts = None
        self.is_checkpoint_reached = False
        self.checkpoint_preparation_complete = False
        self.safe_tx_hash_prepared = False
        self.checkpoint_tx_signing_submitted = False
        self.checkpoint_tx_broadcast_submitted = False
        self.checkpoint_tx_executed = False
        self.checkpoint_tx_raw_data = None
        self.checkpoint_tx_signed_data = None
        self.checkpoint_tx_final_hash = None
        for k in self.supported_protocols:
            self.supported_protocols[k] = []

    def act(self) -> None:
        """Perform the act."""
        try:
            if not self._handle_initialization():
                return

            if not self._handle_staking_state():
                return

            if not self._handle_checkpoint_timing():
                return

            if not self._handle_checkpoint_preparation():
                return

            if not self._handle_transaction_signing():
                return

            if not self._handle_transaction_broadcast():
                return

            self._finalize_checkpoint_check()

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self._handle_error(e)

    def _handle_initialization(self) -> bool:
        """Handle initialization phase. Returns True if should continue."""
        if not self.started:
            self.context.logger.info(f"Entering {self._state} state.")
            self.started = True

        if not self.checkpoint_initialized:
            self._initialize_checkpoint_check()
            return False

        return True

    def _handle_staking_state(self) -> bool:
        """Handle staking state checking phase. Returns True if should continue."""
        if not self.staking_state_check_complete:
            self._check_contract_responses()
            return self.staking_state_check_complete

        return True

    def _handle_checkpoint_timing(self) -> bool:
        """Handle checkpoint timing verification. Returns True if should continue."""
        if self._should_check_checkpoint_timing():
            self._check_if_checkpoint_reached()
            if not self.is_checkpoint_reached:
                self.context.logger.info("Next checkpoint not reached yet")
                self._event = MindshareabciappEvents.CHECKPOINT_NOT_REACHED
                self._is_done = True
                return False

        return True

    def _handle_checkpoint_preparation(self) -> bool:
        """Handle checkpoint transaction preparation. Returns True if should continue."""
        if self._should_prepare_checkpoint():
            self._prepare_checkpoint_tx_async()
            return False

        if self.checkpoint_preparation_complete and not self.safe_tx_hash_prepared:
            self._check_checkpoint_preparation_responses()
            return self.safe_tx_hash_prepared

        return True

    def _handle_transaction_signing(self) -> bool:
        """Handle transaction signing phase. Returns True if should continue."""
        if self._should_sign_transaction():
            self._sign_checkpoint_transaction()
            return False

        if self.checkpoint_tx_signing_submitted and not self.checkpoint_tx_signed_data:
            self._check_signing_responses()
            return bool(self.checkpoint_tx_signed_data)

        return True

    def _handle_transaction_broadcast(self) -> bool:
        """Handle transaction broadcast phase. Returns True if should continue."""
        if self.checkpoint_tx_signed_data and not self.checkpoint_tx_broadcast_submitted:
            self._broadcast_checkpoint_transaction()
            return False

        if self.checkpoint_tx_broadcast_submitted and not self.checkpoint_tx_executed:
            self._check_broadcast_responses()
            return bool(self.checkpoint_tx_executed)

        return True

    def _should_check_checkpoint_timing(self) -> bool:
        """Check if checkpoint timing should be verified."""
        return (
            self.service_staking_state == StakingState.STAKED
            and not self.is_checkpoint_reached
            and self.next_checkpoint_ts is not None
        )

    def _should_prepare_checkpoint(self) -> bool:
        """Check if checkpoint should be prepared."""
        return (
            self.service_staking_state == StakingState.STAKED
            and self.is_checkpoint_reached
            and not self.checkpoint_preparation_complete
        )

    def _should_sign_transaction(self) -> bool:
        """Check if transaction should be signed."""
        return self.safe_tx_hash_prepared and self.checkpoint_tx_raw_data and not self.checkpoint_tx_signing_submitted

    def _handle_error(self, error: Exception) -> None:
        """Handle errors that occur during act()."""
        self.context.logger.error(f"CallCheckpointRound failed: {error}")
        self.context.error_context = {
            "error_type": "checkpoint_error",
            "error_message": str(error),
            "originating_round": str(self._state),
        }
        self._event = MindshareabciappEvents.ERROR
        self._is_done = True

    def _initialize_checkpoint_check(self) -> None:
        """Initialize checkpoint check."""
        if self.checkpoint_initialized:
            return

        self.context.logger.info("Initializing checkpoint check...")

        staking_chain = self.context.params.staking_chain
        if not staking_chain:
            self.context.logger.warning("Service has not been staked on any chain!")
            self.service_staking_state = StakingState.UNSTAKED
            self.staking_state_check_complete = True
            self.checkpoint_initialized = True
            return

        self._get_service_staking_state_async()
        self.checkpoint_initialized = True

    def _get_service_staking_state_async(self) -> None:
        """Get service staking state asynchronously."""
        try:
            staking_chain = self.context.params.staking_chain
            staking_token_contract_address = getattr(self.context.params, "staking_token_contract_address", None)
            service_id = getattr(self.context.params, "service_id", None)
            on_chain_service_id = getattr(self.context.params, "on_chain_service_id", None)

            if not staking_token_contract_address or not service_id:
                self.context.logger.warning("Missing staking contract address or service ID")
                self.service_staking_state = StakingState.UNSTAKED
                self.staking_state_check_complete = True
                return

            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_STATE,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=staking_token_contract_address,
                contract_id=str(StakingTokenContract.contract_id),
                callable="get_service_staking_state",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs(
                    {
                        "service_id": on_chain_service_id,
                        "chain_id": staking_chain,
                    }
                ),
            )

            dialogue.validation_func = self._validate_staking_state_response
            dialogue.request_type = "staking_state"
            self.pending_contract_calls.append(dialogue)

            self.context.logger.info(f"Submitted staking state request for service {service_id}")

        except Exception as e:
            self.context.logger.exception(f"Failed to get service staking state: {e}")
            self.service_staking_state = StakingState.UNSTAKED
            self.staking_state_check_complete = True

    def _validate_staking_state_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate staking state response message."""
        try:
            if message.performative == ContractApiMessage.Performative.STATE:
                if hasattr(message, "state") and message.state:
                    state_data = message.state.body.get("data")
                    if state_data is not None:
                        staking_state = int(state_data)
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "staking_state": staking_state,
                            "request_type": "staking_state",
                        }
                        self.context.logger.info(f"Received staking state: {staking_state}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating staking state response: {e}")
            return False

    def _check_contract_responses(self) -> None:
        """Check if contract responses have arrived and process them."""
        try:
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]
                    request_type = response.get("request_type")

                    if request_type == "staking_state":
                        self._process_staking_state_response(response)
                    elif request_type == "next_checkpoint":
                        self._process_next_checkpoint_response(response)
                    elif request_type == "checkpoint_preparation":
                        self._process_checkpoint_preparation_response(response)
                    elif request_type == "safe_tx_preparation":
                        self._process_safe_tx_preparation_response(response)

                    self.pending_contract_calls.remove(dialogue)

            if not self.pending_contract_calls and not self.staking_state_check_complete:
                self.service_staking_state = StakingState.UNSTAKED
                self.staking_state_check_complete = True
                self.context.logger.info("No staking state response received, assuming unstaked")

        except Exception as e:
            self.context.logger.exception(f"Error checking contract responses: {e}")
            self.service_staking_state = StakingState.UNSTAKED
            self.staking_state_check_complete = True

    def _process_staking_state_response(self, response: dict) -> None:
        """Process staking state response."""
        staking_state = response["staking_state"]

        if staking_state == 0:
            self.service_staking_state = StakingState.UNSTAKED
            self.context.logger.info("Service is not staked")
        elif staking_state == 1:
            self.service_staking_state = StakingState.STAKED
            self.context.logger.info("Service is staked")
            self._get_next_checkpoint_async()
        elif staking_state == 2:
            self.service_staking_state = StakingState.EVICTED
            self.context.logger.error("Service has been evicted!")
        else:
            self.context.logger.warning(f"Unknown staking state: {staking_state}")
            self.service_staking_state = StakingState.UNSTAKED

        if self.service_staking_state in {StakingState.UNSTAKED, StakingState.EVICTED}:
            self.staking_state_check_complete = True

    def _get_next_checkpoint_async(self) -> None:
        """Get next checkpoint timestamp asynchronously."""
        try:
            staking_token_contract_address = getattr(self.context.params, "staking_token_contract_address", None)
            if not staking_token_contract_address:
                self.staking_state_check_complete = True
                return

            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_STATE,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=staking_token_contract_address,
                contract_id=str(StakingTokenContract.contract_id),
                callable="get_next_checkpoint_ts",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs({"chain_id": self.context.params.staking_chain}),
            )

            dialogue.validation_func = self._validate_next_checkpoint_response
            dialogue.request_type = "next_checkpoint"
            self.pending_contract_calls.append(dialogue)

            self.context.logger.info("Submitted next checkpoint request")

        except Exception as e:
            self.context.logger.exception(f"Failed to get next checkpoint: {e}")
            self.staking_state_check_complete = True

    def _validate_next_checkpoint_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate next checkpoint response message."""
        try:
            if message.performative == ContractApiMessage.Performative.STATE:
                if hasattr(message, "state") and message.state:
                    checkpoint_data = message.state.body.get("data")
                    if checkpoint_data is not None:
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "next_checkpoint_ts": int(checkpoint_data),
                            "request_type": "next_checkpoint",
                        }
                        self.context.logger.info(f"Received next checkpoint timestamp: {checkpoint_data}")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating next checkpoint response: {e}")
            return False

    def _process_next_checkpoint_response(self, response: dict) -> None:
        """Process next checkpoint response."""
        self.next_checkpoint_ts = response["next_checkpoint_ts"]
        self.staking_state_check_complete = True
        self.context.logger.info(f"Next checkpoint at timestamp: {self.next_checkpoint_ts}")

    def _check_if_checkpoint_reached(self) -> None:
        """Check if checkpoint is reached."""
        if self.next_checkpoint_ts is None:
            self.is_checkpoint_reached = False
            return

        if self.next_checkpoint_ts == 0:
            self.is_checkpoint_reached = True
            return

        current_timestamp = int(datetime.now(UTC).timestamp())
        self.is_checkpoint_reached = self.next_checkpoint_ts <= current_timestamp

        self.context.logger.info(
            f"Checkpoint check: current={current_timestamp}, next_checkpoint={self.next_checkpoint_ts}, "
            f"reached={self.is_checkpoint_reached}"
        )

    def _prepare_checkpoint_tx_async(self) -> None:
        """Prepare checkpoint transaction asynchronously."""
        try:
            self.context.logger.info("Preparing checkpoint transaction...")
            staking_token_contract_address = getattr(self.context.params, "staking_token_contract_address", None)

            if not staking_token_contract_address:
                self.context.logger.warning("No staking token contract address")
                self.checkpoint_preparation_complete = True
                return

            dialogue = self.submit_msg(
                performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                connection_id=LEDGER_API_ADDRESS,
                contract_address=staking_token_contract_address,
                contract_id=str(StakingTokenContract.contract_id),
                callable="build_checkpoint_tx",
                ledger_id="ethereum",
                kwargs=ContractApiMessage.Kwargs({"chain_id": self.context.params.staking_chain}),
            )

            dialogue.validation_func = self._validate_checkpoint_preparation_response
            dialogue.request_type = "checkpoint_preparation"
            self.pending_contract_calls.append(dialogue)

            self.context.logger.info("Submitted checkpoint preparation request")

        except Exception as e:
            self.context.logger.exception(f"Failed to prepare checkpoint transaction: {e}")
            self.checkpoint_preparation_complete = True

    def _validate_checkpoint_preparation_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate checkpoint preparation response message."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                if hasattr(message, "raw_transaction") and message.raw_transaction:
                    checkpoint_data = message.raw_transaction.body.get("data")
                    if checkpoint_data:
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "checkpoint_data": checkpoint_data,
                            "request_type": "checkpoint_preparation",
                        }
                        self.context.logger.info("Received checkpoint transaction data")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating checkpoint preparation response: {e}")
            return False

    def _check_checkpoint_preparation_responses(self) -> None:
        """Check if checkpoint preparation responses have arrived."""
        try:
            for dialogue in self.pending_contract_calls.copy():
                request_nonce = dialogue.dialogue_label.dialogue_reference[0]

                if request_nonce in self.contract_responses:
                    response = self.contract_responses[request_nonce]

                    if response.get("request_type") == "checkpoint_preparation":
                        self._process_checkpoint_preparation_response(response)
                        self.pending_contract_calls.remove(dialogue)

            if not self.pending_contract_calls and not self.checkpoint_preparation_complete:
                self.checkpoint_preparation_complete = True
                self.context.logger.info("Checkpoint preparation completed (no response)")

        except Exception as e:
            self.context.logger.exception(f"Error checking checkpoint preparation responses: {e}")
            self.checkpoint_preparation_complete = True

    def _process_checkpoint_preparation_response(self, response: dict) -> None:
        """Process checkpoint preparation response."""
        checkpoint_data = response["checkpoint_data"]
        self.checkpoint_tx_hex = self._prepare_safe_tx_hash(checkpoint_data)
        self.checkpoint_preparation_complete = True
        self.context.logger.info(f"Checkpoint transaction prepared: {self.checkpoint_tx_hex}")

    def _process_safe_tx_preparation_response(self, response: dict) -> None:
        """Process Safe transaction preparation response."""
        safe_tx_raw_data = response["safe_tx_raw_data"]
        self.checkpoint_tx_raw_data = safe_tx_raw_data
        self.safe_tx_hash_prepared = True
        self.context.logger.info("Safe transaction raw data received and ready for signing")

    def _prepare_safe_tx_hash(self, data: bytes) -> str | None:
        """Prepare safe transaction hash and full transaction."""
        try:
            staking_chain = self.context.params.staking_chain
            safe_addresses = self.context.params.safe_contract_addresses

            if isinstance(safe_addresses, str):
                try:
                    safe_addresses = json.loads(safe_addresses)
                except json.JSONDecodeError:
                    safe_addresses = {}

            safe_address = safe_addresses.get(staking_chain)
            if not safe_address:
                self.context.logger.warning(f"No safe address found for staking chain {staking_chain}")
                return None

            staking_token_contract_address = getattr(self.context.params, "staking_token_contract_address", None)
            if not staking_token_contract_address:
                return None

            # Submit contract call to get the raw Safe transaction
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
                        "to_address": staking_token_contract_address,
                        "value": ETHER_VALUE,
                        "data": data,
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

            dialogue.validation_func = self._validate_safe_tx_preparation_response
            dialogue.request_type = "safe_tx_preparation"
            self.pending_contract_calls.append(dialogue)

            self.context.logger.info("Submitted Safe transaction preparation request")
            return None  # Will be set when response arrives

        except Exception as e:
            self.context.logger.exception(f"Failed to prepare safe transaction hash: {e}")
            return None

    def _validate_safe_tx_preparation_response(self, message: Message, dialogue: ContractApiDialogue) -> bool:
        """Validate Safe transaction preparation response message."""
        try:
            if message.performative == ContractApiMessage.Performative.RAW_TRANSACTION:
                if hasattr(message, "raw_transaction") and message.raw_transaction:
                    raw_tx = message.raw_transaction
                    if raw_tx:
                        self.contract_responses[dialogue.dialogue_label.dialogue_reference[0]] = {
                            "safe_tx_raw_data": raw_tx,
                            "request_type": "safe_tx_preparation",
                        }
                        self.context.logger.info("Received Safe transaction raw data")
                        return True

            elif message.performative == ContractApiMessage.Performative.ERROR:
                self.context.logger.warning(f"Contract API error: {message.message}")

            return False

        except Exception as e:
            self.context.logger.exception(f"Error validating Safe tx preparation response: {e}")
            return False

    def _sign_checkpoint_transaction(self) -> None:
        """Sign the checkpoint transaction."""
        self.context.logger.info("Signing checkpoint transaction...")

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
            raw_transaction=self.checkpoint_tx_raw_data,
            terms=terms,
        )

        signing_dialogue.validation_func = self._validate_checkpoint_signing_response

        request_nonce = signing_dialogue.dialogue_label.dialogue_reference[0]
        self.context.requests.request_id_to_callback[request_nonce] = self.get_dialogue_callback_request()

        self.context.decision_maker_message_queue.put_nowait(signing_msg)

        self.checkpoint_tx_signing_submitted = True
        self.context.logger.info("Checkpoint transaction sent for signing")

    def _validate_checkpoint_signing_response(self, message: SigningMessage, _dialogue) -> bool:
        """Process checkpoint transaction signing response."""
        try:
            if message.performative == SigningMessage.Performative.SIGNED_TRANSACTION:
                self.checkpoint_tx_signed_data = message.signed_transaction
                self.context.logger.info("Checkpoint transaction signed successfully")
                return True

            self.context.logger.error(f"Checkpoint transaction signing failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing checkpoint signing response: {e}")
            return False

    def _broadcast_checkpoint_transaction(self) -> None:
        """Broadcast the signed checkpoint transaction."""
        self.context.logger.info("Broadcasting checkpoint transaction to chain...")

        dialogue = self.submit_msg(
            performative=LedgerApiMessage.Performative.SEND_SIGNED_TRANSACTION,
            connection_id=LEDGER_API_ADDRESS,
            signed_transaction=self.checkpoint_tx_signed_data,
            kwargs=LedgerApiMessage.Kwargs({"ledger_id": "ethereum"}),
        )

        dialogue.validation_func = self._validate_checkpoint_broadcast_response
        self.checkpoint_tx_broadcast_submitted = True

        self.context.logger.info("Broadcasting checkpoint transaction to chain")

    def _validate_checkpoint_broadcast_response(self, message: LedgerApiMessage, _dialogue) -> bool:
        """Process checkpoint transaction broadcast response."""
        try:
            if message.performative == LedgerApiMessage.Performative.TRANSACTION_DIGEST:
                tx_hash = message.transaction_digest.body
                self.checkpoint_tx_final_hash = tx_hash
                self.checkpoint_tx_executed = True
                self.context.logger.info(f"Checkpoint transaction broadcast successful: {tx_hash}")
                return True

            self.context.logger.error(f"Checkpoint transaction broadcast failed: {message.performative}")
            return False

        except Exception as e:
            self.context.logger.exception(f"Error processing checkpoint broadcast response: {e}")
            return False

    def _check_signing_responses(self) -> None:
        """Check for signing responses."""
        # Signing responses are handled directly by the validation function

    def _check_broadcast_responses(self) -> None:
        """Check for broadcast responses."""
        # Broadcast responses are handled directly by the validation function

    def _save_staking_state_to_state_json(self) -> None:
        """Save staking state information to state.json."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available, cannot save staking state")
            return

        state_file = Path(self.context.store_path) / "state.json"
        try:
            # Load existing state
            state_data = {}
            if state_file.exists():
                with open(state_file, encoding=DEFAULT_ENCODING) as f:
                    state_data = json.load(f)

            # Update staking information
            staking_data = {
                "staking_status": self.service_staking_state.name if self.service_staking_state else "UNSTAKED",
                "next_checkpoint_ts": self.next_checkpoint_ts,
                "is_checkpoint_reached": self.is_checkpoint_reached,
                "checkpoint_tx_executed": self.checkpoint_tx_executed,
                "checkpoint_tx_final_hash": self.checkpoint_tx_final_hash,
                "last_checkpoint_check": datetime.now(UTC).isoformat(),
                "has_required_funds": True,
                "is_making_on_chain_transactions": bool(self.checkpoint_tx_executed and self.checkpoint_tx_final_hash),
            }

            state_data.update(staking_data)

            # Save updated state
            with open(state_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(state_data, f, indent=2)

            self.context.logger.debug(f"Saved staking state to state.json: {staking_data}")

        except (PermissionError, OSError, json.JSONDecodeError) as e:
            self.context.logger.warning(f"Failed to save staking state to state.json: {e}")

    def _finalize_checkpoint_check(self) -> None:
        """Finalize checkpoint check and determine transition."""
        self.context.logger.info("Finalizing checkpoint check...")

        # Save staking state to state.json
        self._save_staking_state_to_state_json()

        if self.service_staking_state == StakingState.UNSTAKED:
            self.context.logger.info("Service is not staked")
            self._event = MindshareabciappEvents.SERVICE_NOT_STAKED
        elif self.service_staking_state == StakingState.EVICTED:
            self.context.logger.info("Service has been evicted")
            self._event = MindshareabciappEvents.SERVICE_EVICTED
        elif self.service_staking_state == StakingState.STAKED:
            if self.checkpoint_tx_executed and self.checkpoint_tx_final_hash:
                self.context.logger.info(
                    f"Checkpoint transaction executed successfully: {self.checkpoint_tx_final_hash}"
                )
                self._event = MindshareabciappEvents.DONE
            elif self.checkpoint_tx_hex:
                self.context.logger.info(f"Checkpoint transaction prepared: {self.checkpoint_tx_hex}")
                self._event = MindshareabciappEvents.CHECKPOINT_PREPARED
            else:
                self.context.logger.info("Checkpoint not reached yet")
                self._event = MindshareabciappEvents.CHECKPOINT_NOT_REACHED
        else:
            self.context.logger.error("Unknown staking state")
            self._event = MindshareabciappEvents.ERROR

        self._is_done = True
