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

from typing import Any

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class HandleErrorRound(BaseState):
    """This class implements the behaviour of the state HandleErrorRound."""

    RETRYABLE_ERRORS = {
        "ConnectionError": True,
        "timeout_error": True,
        "network_error": True,
    }

    NON_RETRYABLE_ERRORS = {
        "configuration_error": False,
        "invalid_price_error": False,
        "cowswap_fee_error": False,  # Position size too small for fees
        "ticker_timeout_error": False,  # Exceeded max ticker retry attempts
        "price_sanity_check_failed": False,  # Price validation failed
        "trade_construction_error": False,  # General construction errors
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.HANDLEERRORROUND
        self._retry_states = {}

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")

        # Check error context to determine if this is a retryable error
        error_context = getattr(self.context, "error_context", {})
        error_type = error_context.get("error_type", "unknown_error")
        error_message = error_context.get("error_message", "No error message")
        originating_round = error_context.get("originating_round", "Unknown")

        # Log detailed error information once
        self.context.logger.error(
            f"Handling error from {originating_round}: " f"Type: {error_type}, Message: {error_message}"
        )

        # Log additional error details if present
        if "error_details" in error_context:
            self.context.logger.error(f"Error details: {error_context['error_details']}")

        if error_type in self.NON_RETRYABLE_ERRORS:
            self.context.logger.info(f"Non-retryable error detected: {error_type}. Moving to paused round for cycling.")
            self._event = MindshareabciappEvents.RETRIES_EXCEEDED
        elif error_type in self.RETRYABLE_ERRORS:
            self.context.logger.info(f"Retryable error detected: {error_type}. Attempting retry.")
            self._event = MindshareabciappEvents.RETRY
        else:
            # Default to non-retryable for unknown errors
            self.context.logger.warning(f"Unknown error type: {error_type}. Treating as non-retryable.")
            self._event = MindshareabciappEvents.RETRIES_EXCEEDED

        # Clear error context to prevent re-processing
        self.context.error_context = {}

        self._is_done = True
