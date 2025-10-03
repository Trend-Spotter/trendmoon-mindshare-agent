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

from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class SetupRound(BaseState):
    """This class implements the behaviour of the state SetupRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.SETUPROUND
        self.setup_success: bool = False
        self.setup_data: dict[str, Any] = {}
        self.started: bool = False

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self._is_done = False

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

    def act(self) -> None:
        """Perform the act."""
        self.context.logger.info(f"Entering {self._state} state.")
        if self.started:
            return
        self.started = True

        try:
            self._initialize_state()
            self.setup_success = True
            self._event = MindshareabciappEvents.DONE

        except Exception as e:
            self.context.logger.exception(f"Setup failed. {e!s}")
            self.context.error_context = {
                "error_type": "setup_error",
                "error_message": str(e),
                "originating_round": str(self._state),
            }
            self._event = MindshareabciappEvents.ERROR

        finally:
            self._is_done = True
