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

"""This module contains the handler for the Mindshare app skill."""

import json
from typing import TYPE_CHECKING, cast

from aea.skills.base import Handler
from aea.protocols.base import Message
from aea.configurations.data_types import PublicId

from packages.open_aea.protocols.signing import SigningMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.eightballer.protocols.orders import OrdersMessage
from packages.eightballer.protocols.default import DefaultMessage
from packages.eightballer.protocols.tickers import TickersMessage
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.eightballer.protocols.http.message import HttpMessage
from packages.xiuxiuxar.skills.mindshare_app.dialogues import (
    HttpDialogue,
    HttpDialogues,
    OrdersDialogue,
    OrdersDialogues,
    SigningDialogue,
    TickersDialogue,
    DefaultDialogues,
    SigningDialogues,
    TickersDialogues,
    LedgerApiDialogue,
    LedgerApiDialogues,
    ContractApiDialogue,
    ContractApiDialogues,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from packages.xiuxiuxar.skills.mindshare_app.models import Requests, HealthCheckService


class MinshareAppHandlerError(Exception):
    """Exception for the Mindshare app handler."""


class HttpHandler(Handler):
    """This implements the echo handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id

    def setup(self) -> None:
        """Implement the setup."""

    def handle(self, message: Message) -> None:
        """Implement the reaction to an envelope."""
        http_msg = cast("HttpMessage", message)

        # recover dialogue
        http_dialogues = cast("HttpDialogues", self.context.http_dialogues)
        http_dialogue = cast("HttpDialogue", http_dialogues.update(http_msg))
        if http_dialogue is None:
            self._handle_unidentified_dialogue(http_msg)
            return

        # handle message
        if http_msg.performative == HttpMessage.Performative.REQUEST:
            self._handle_request(http_msg, http_dialogue)
        elif http_msg.performative == HttpMessage.Performative.RESPONSE:
            self._handle_response(http_msg, http_dialogue)
        else:
            self._handle_invalid(http_msg, http_dialogue)

    def _handle_get_healthcheck(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle a Http request of verb GET for healthcheck."""
        # Get health check service
        health_service = cast("HealthCheckService", self.context.health_check_service)

        # Get FSM status
        fsm_status = health_service.get_fsm_status()

        # Build complete health check response
        data = {
            "seconds_since_last_transition": fsm_status["seconds_since_last_transition"],
            "is_tm_healthy": True,
            "period": fsm_status["period_count"],
            "reset_pause_duration": self.context.params.reset_pause_duration,
            "rounds": fsm_status["rounds"],
            "is_transitioning_fast": fsm_status["is_transitioning_fast"],
            "agent_health": health_service.get_agent_health(),
            "rounds_info": health_service.build_rounds_info(),
            "env_var_status": health_service.get_env_var_status(),
        }

        self._send_ok_response(http_msg, http_dialogue, data)

    def _send_ok_response(self, http_msg: HttpMessage, http_dialogue: HttpDialogue, data: dict) -> None:
        """Send an OK response with JSON data."""
        if self.enable_cors:
            cors_headers = "Access-Control-Allow-Origin: *\n"
            cors_headers += "Access-Control-Allow-Methods: GET, POST\n"
            cors_headers += "Access-Control-Allow-Headers: Content-Type,Accept\n"
            headers = cors_headers + "Content-Type: application/json\n" + http_msg.headers
        else:
            headers = "Content-Type: application/json\n" + http_msg.headers

        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=200,
            status_text="Success",
            headers=headers,
            body=json.dumps(data, indent=2).encode("utf-8"),
        )
        self.context.logger.info(f"responding with healthcheck: {http_response}")
        self.context.outbox.put_message(message=http_response)

    def _handle_unidentified_dialogue(self, http_msg: HttpMessage) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.info(f"received invalid http message={http_msg}, unidentified dialogue.")
        default_dialogues = cast("DefaultDialogues", self.context.default_dialogues)
        default_msg, _ = default_dialogues.create(
            counterparty=http_msg.sender,
            performative=DefaultMessage.Performative.ERROR,
            error_code=DefaultMessage.ErrorCode.INVALID_DIALOGUE,
            error_msg="Invalid dialogue.",
            error_data={"http_message": http_msg.encode()},
        )
        self.context.outbox.put_message(message=default_msg)

    def _handle_request(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle a Http request."""
        self.context.logger.info(
            f"received http request with method={http_msg.method}, url={http_msg.url} and body={http_msg.body}"
        )
        if http_msg.method == "get":
            if "healthcheck" in http_msg.url:
                self._handle_get_healthcheck(http_msg, http_dialogue)
            elif "metrics" in http_msg.url:
                self._handle_get(http_msg, http_dialogue)
            else:
                self._handle_get(http_msg, http_dialogue)
        else:
            self._handle_invalid(http_msg, http_dialogue)

    def _handle_get(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle a Http request of verb GET."""
        if self.enable_cors:
            cors_headers = "Access-Control-Allow-Origin: *\n"
            cors_headers += "Access-Control-Allow-Methods: POST\n"
            cors_headers += "Access-Control-Allow-Headers: Content-Type,Accept\n"
            headers = cors_headers + http_msg.headers
        else:
            headers = http_msg.headers

        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=200,
            status_text="Success",
            headers=headers,
            body=json.dumps(self.context.shared_state).encode("utf-8"),
        )
        self.context.logger.info(f"responding with: {http_response}")
        self.context.outbox.put_message(message=http_response)

    def _handle_post(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle a Http request of verb POST."""
        http_response = http_dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=http_msg,
            version=http_msg.version,
            status_code=200,
            status_text="Success",
            headers=http_msg.headers,
            body=http_msg.body,
        )
        self.context.logger.info(f"responding with: {http_response}")
        self.context.outbox.put_message(message=http_response)

    def _handle_response(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle HTTP response messages by routing them to the FSM behavior."""
        # HTTP responses (API responses from CoinGecko/TrendMoon) should be handled by the FSM behavior
        # rather than this handler which is designed for incoming HTTP requests (healthcheck)

        if http_msg.status_code == 429:
            self._handle_rate_limit_response(http_msg)
            return

        self.context.logger.debug(f"Routing HTTP response to behavior system: {http_msg.performative}")

        # Get the main FSM behavior
        if hasattr(self.context, "behaviours") and hasattr(self.context.behaviours, "main"):
            main_behavior = self.context.behaviours.main
            if hasattr(main_behavior, "handle_message"):
                main_behavior.handle_message(http_msg)
            else:
                self.context.logger.warning("Main behavior does not have handle_message method")
        else:
            self.context.logger.warning("Cannot route HTTP response - main behavior not found")

    def _handle_rate_limit_response(self, http_msg: HttpMessage) -> None:
        """Handle rate limit responses from APIs."""
        # Determine which API was rate limited based on the request URL
        self.context.logger.info(f"Handling rate limit response: {http_msg}")

        dialogue = cast("HttpDialogue", self.context.http_dialogues.get_dialogue(http_msg))
        if dialogue is None:
            self.context.logger.warning("Cannot find dialogue for rate limit response")
            return
        request_message = dialogue.get_message_by_id(http_msg.target)
        if request_message is None:
            self.context.logger.warning("Cannot find request message for rate limit response")
            return

        request_url = request_message.url if hasattr(request_message, "url") else ""

        if "coingecko.com" in request_url:
            self.context.logger.warning("CoinGecko API rate limit detected")
            if hasattr(self.context, "coingecko"):
                self.context.coingecko.handle_rate_limit_response()
        elif "trendmoon.ai" in request_url:
            self.context.logger.warning("TrendMoon API rate limit detected")
            if hasattr(self.context, "trendmoon"):
                self.context.trendmoon.handle_rate_limit_response()

        # Send rate limit response to the skill
        self.context.logger.error(f"Rate limited by API: {request_url}")

    def _handle_invalid(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle an invalid http message."""
        self.context.logger.warning(
            f"""
            Cannot handle http message of
            performative={http_msg.performative}
            dialogue={http_dialogue.dialogue_label}.
            """
        )

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def __init__(self, **kwargs):
        """Initialise the handler."""
        self.enable_cors = kwargs.pop("enable_cors", False)
        super().__init__(**kwargs)


class LedgerApiHandler(Handler):
    """Handler for ledger API responses."""

    SUPPORTED_PROTOCOL: PublicId | None = LedgerApiMessage.protocol_id

    allowed_response_performatives = frozenset(
        {
            LedgerApiMessage.Performative.TRANSACTION_DIGEST,
            LedgerApiMessage.Performative.TRANSACTION_DIGESTS,
            LedgerApiMessage.Performative.TRANSACTION_RECEIPT,
            LedgerApiMessage.Performative.ERROR,
        }
    )

    def setup(self) -> None:
        """Implement the setup."""

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def handle(self, message: Message) -> None:
        """Handle ledger API messages."""
        ledger_msg = cast("LedgerApiMessage", message)

        ledger_api_dialogues = cast("LedgerApiDialogues", self.context.ledger_api_dialogues)
        ledger_dialogue = cast("LedgerApiDialogue", ledger_api_dialogues.update(ledger_msg))

        if ledger_dialogue is None:
            self._handle_unidentified_dialogue(ledger_msg)
            return

        if message.performative not in self.allowed_response_performatives:
            self._handle_unallowed_performative(message)
            return

        request_nonce = ledger_dialogue.dialogue_label.dialogue_reference[0]
        ctx_requests = cast("Requests", self.context.requests)

        try:
            callback = cast("Callable", ctx_requests.request_id_to_callback.pop(request_nonce))
        except KeyError as e:
            msg = f"No callback defined for request with nonce: {request_nonce}"
            raise MinshareAppHandlerError(msg) from e

        self._log_message_handling(message)
        try:
            # Call with correct signature - this app uses (message, dialogue) not (message, dialogue, behaviour)
            callback(message, ledger_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error in callback execution for nonce {request_nonce}: {e}")
            return

        self.context.logger.info(f"Received ledger API response: {ledger_msg.performative}")

    def _handle_unidentified_dialogue(self, message: Message) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.warning("Received invalid message: unidentified dialogue. message=%s", message)

    def _handle_unallowed_performative(self, message: Message) -> None:
        """Handle a message with an unallowed response performative."""
        self.context.logger.warning("Received invalid message: unallowed performative. message=%s.", message)

    def _log_message_handling(self, message: Message) -> None:
        """Log the handling of the message."""
        self.context.logger.debug("Calling registered callback with message=%s", message)


class ContractApiHandler(Handler):
    """Handler for contract API responses."""

    SUPPORTED_PROTOCOL: PublicId | None = ContractApiMessage.protocol_id

    allowed_response_performatives = frozenset(
        {
            ContractApiMessage.Performative.RAW_TRANSACTION,
            ContractApiMessage.Performative.RAW_MESSAGE,
            ContractApiMessage.Performative.ERROR,
            ContractApiMessage.Performative.STATE,
        }
    )

    def setup(self) -> None:
        """Implement the setup."""

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def handle(self, message: Message) -> None:
        """Handle contract API messages."""
        contract_msg = cast("ContractApiMessage", message)

        # Recover dialogue
        contract_api_dialogues = cast("ContractApiDialogues", self.context.contract_api_dialogues)
        contract_dialogue = cast("ContractApiDialogue", contract_api_dialogues.update(contract_msg))

        if contract_dialogue is None:
            self._handle_unidentified_dialogue(contract_msg)
            return

        if message.performative not in self.allowed_response_performatives:
            self._handle_unallowed_performative(message)
            return

        request_nonce = contract_dialogue.dialogue_label.dialogue_reference[0]
        ctx_requests = cast("Requests", self.context.requests)

        try:
            callback = cast(
                "Callable",
                ctx_requests.request_id_to_callback.pop(request_nonce),
            )
        except KeyError as e:
            msg = f"No callback defined for request with nonce: {request_nonce}"
            raise MinshareAppHandlerError(msg) from e

        self._log_message_handling(message)
        try:
            # Call with correct signature - this app uses (message, dialogue) not (message, dialogue, behaviour)
            callback(message, contract_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error in callback execution for nonce {request_nonce}: {e}")
            return

    def _handle_unidentified_dialogue(self, message: Message) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.warning("Received invalid message: unidentified dialogue. message=%s", message)

    def _handle_unallowed_performative(self, message: Message) -> None:
        """Handle a message with an unallowed response performative."""
        self.context.logger.warning("Received invalid message: unallowed performative. message=%s.", message)

    def _log_message_handling(self, message: Message) -> None:
        """Log the handling of the message."""
        self.context.logger.debug("Calling registered callback with message=%s", message)


class OrdersHandler(Handler):
    """Handler for orders messages."""

    SUPPORTED_PROTOCOL: PublicId | None = OrdersMessage.protocol_id

    allowed_response_performatives = frozenset(
        {
            OrdersMessage.Performative.ORDER,
            OrdersMessage.Performative.ORDER_CREATED,
            OrdersMessage.Performative.ERROR,
        }
    )

    def setup(self) -> None:
        """Implement the setup."""

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def handle(self, message: Message) -> None:
        """Handle orders messages."""
        orders_msg = cast("OrdersMessage", message)

        # Recover dialogue
        orders_dialogues = cast("OrdersDialogues", self.context.orders_dialogues)
        orders_dialogue = cast("OrdersDialogue", orders_dialogues.update(orders_msg))

        if orders_dialogue is None:
            self._handle_unidentified_dialogue(orders_msg)
            return

        if message.performative not in self.allowed_response_performatives:
            self._handle_unallowed_performative(message)
            return

        request_nonce = orders_dialogue.dialogue_label.dialogue_reference[0]
        ctx_requests = cast("Requests", self.context.requests)

        try:
            callback = cast("Callable", ctx_requests.request_id_to_callback.pop(request_nonce))
        except KeyError as e:
            msg = f"No callback defined for request with nonce: {request_nonce}"
            raise MinshareAppHandlerError(msg) from e

        self._log_message_handling(message)
        try:
            # Call with correct signature - this app uses (message, dialogue) not (message, dialogue, behaviour)
            callback(message, orders_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error in callback execution for nonce {request_nonce}: {e}")
            return

        self.context.logger.info(f"Received orders response: {orders_msg.performative}")

    def _handle_unidentified_dialogue(self, message: Message) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.warning("Received invalid message: unidentified dialogue. message=%s", message)

    def _handle_unallowed_performative(self, message: Message) -> None:
        """Handle a message with an unallowed response performative."""
        self.context.logger.warning("Received invalid message: unallowed performative. message=%s.", message)

    def _log_message_handling(self, message: Message) -> None:
        """Log the handling of the message."""
        self.context.logger.debug("Calling registered callback with message=%s", message)


class TickersHandler(Handler):
    """Handler for tickers messages."""

    SUPPORTED_PROTOCOL: PublicId | None = TickersMessage.protocol_id

    allowed_response_performatives = frozenset(
        {
            TickersMessage.Performative.TICKER,
            TickersMessage.Performative.ERROR,
        }
    )

    def setup(self) -> None:
        """Implement the setup."""

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def handle(self, message: Message) -> None:
        """Handle tickers messages."""
        tickers_msg = cast("TickersMessage", message)

        # Recover dialogue
        tickers_dialogues = cast("TickersDialogues", self.context.tickers_dialogues)
        tickers_dialogue = cast("TickersDialogue", tickers_dialogues.update(tickers_msg))

        if tickers_dialogue is None:
            self._handle_unidentified_dialogue(tickers_msg)
            return

        if message.performative not in self.allowed_response_performatives:
            self._handle_unallowed_performative(message)
            return

        request_nonce = tickers_dialogue.dialogue_label.dialogue_reference[0]
        ctx_requests = cast("Requests", self.context.requests)

        try:
            callback = cast(
                "Callable",
                ctx_requests.request_id_to_callback.pop(request_nonce),
            )
        except KeyError as e:
            msg = f"No callback defined for request with nonce: {request_nonce}"
            raise MinshareAppHandlerError(msg) from e

        self._log_message_handling(message)
        try:
            # Call with correct signature - this app uses (message, dialogue) not (message, dialogue, behaviour)
            callback(message, tickers_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error in callback execution for nonce {request_nonce}: {e}")
            return

        self.context.logger.info(f"Received tickers response: {tickers_msg.performative}")

    def _handle_unidentified_dialogue(self, message: Message) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.warning("Received invalid message: unidentified dialogue. message=%s", message)

    def _handle_unallowed_performative(self, message: Message) -> None:
        """Handle a message with an unallowed response performative."""
        self.context.logger.warning("Received invalid message: unallowed performative. message=%s.", message)

    def _log_message_handling(self, message: Message) -> None:
        """Log the handling of the message."""
        self.context.logger.debug("Calling registered callback with message=%s", message)


class SigningHandler(Handler):
    """Handler for signing messages."""

    SUPPORTED_PROTOCOL: PublicId | None = SigningMessage.protocol_id

    allowed_response_performatives = frozenset(
        {
            SigningMessage.Performative.SIGNED_TRANSACTION,
            SigningMessage.Performative.ERROR,
        }
    )

    def setup(self) -> None:
        """Implement the setup."""

    def teardown(self) -> None:
        """Implement the handler teardown."""

    def handle(self, message: Message) -> None:
        """Handle signing messages."""
        signing_msg = cast("SigningMessage", message)

        # Recover dialogue
        signing_dialogues = cast("SigningDialogues", self.context.signing_dialogues)
        signing_dialogue = cast("SigningDialogue", signing_dialogues.update(signing_msg))

        if signing_dialogue is None:
            self._handle_unidentified_dialogue(signing_msg)
            return

        if message.performative not in self.allowed_response_performatives:
            self._handle_unallowed_performative(message)
            return

        request_nonce = signing_dialogue.dialogue_label.dialogue_reference[0]
        ctx_requests = cast("Requests", self.context.requests)

        try:
            callback = cast("Callable", ctx_requests.request_id_to_callback.pop(request_nonce))
        except KeyError as e:
            msg = f"No callback defined for request with nonce: {request_nonce}"
            raise MinshareAppHandlerError(msg) from e

        self._log_message_handling(message)
        try:
            # Call with correct signature - this app uses (message, dialogue) not (message, dialogue, behaviour)
            callback(message, signing_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error in callback execution for nonce {request_nonce}: {e}")
            return

        self.context.logger.info(f"Received signing response: {signing_msg.performative}")

    def _handle_unidentified_dialogue(self, message: Message) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.warning("Received invalid message: unidentified dialogue. message=%s", message)

    def _handle_unallowed_performative(self, message: Message) -> None:
        """Handle a message with an unallowed response performative."""
        self.context.logger.warning("Received invalid message: unallowed performative. message=%s.", message)

    def _log_message_handling(self, message: Message) -> None:
        """Log the handling of the message."""
        self.context.logger.debug("Calling registered callback with message=%s", message)
