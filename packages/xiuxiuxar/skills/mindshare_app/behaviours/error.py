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

        self._is_done = True
