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

"""This module contains the model for the Mindshare app."""

from typing import TYPE_CHECKING, Any, cast
from datetime import UTC, datetime

import requests
from aea.skills.base import Model


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.mindshare_app.behaviours import (
        MindshareabciappFsmBehaviour,
    )

MARGIN = 5


class Coingecko(Model):
    """This class implements the CoinGecko API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.coingecko_api_key: str = kwargs.get("coingecko_api_key")

    def validate_required_params(self, params: dict[str, str], required_keys: list[str], param_type: str) -> None:
        """Validate that required parameters are present and not None."""
        if params is None or params == {}:
            msg = f"{param_type} is required"
            raise ValueError(msg)

        for key in required_keys:
            if key not in params or params[key] is None:
                msg = f"{key} is required in {param_type}"
                raise ValueError(msg)

    def coin_ohlc_chart_by_id(self, path_params: dict[str, str], query_params: dict[str, str]) -> list[list[Any]]:
        """Fetch OHLC data for a coin from CoinGecko."""
        try:
            self.validate_required_params(path_params, ["id"], "path_params")
            self.validate_required_params(query_params, ["vs_currency", "days"], "query_params")

            base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/ohlc"
            url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in query_params.items())

            headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            # Log the error and return empty list to avoid crashing
            logging.exception(f"Error fetching OHLC data: {e!s}")
            return []

    def coin_price_by_id(self, query_params: dict[str, str]) -> dict[str, Any]:
        """Fetch price data for a coin from CoinGecko."""
        try:
            self.validate_required_params(query_params, ["vs_currencies"], "query_params")

            base_url = "https://api.coingecko.com/api/v3/simple/price"
            url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in query_params.items())

            headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            # Log the error and return empty dict to avoid crashing
            logging.exception(f"Error fetching price data: {e!s}")
            return {}

    def coin_historical_data_by_id(self, path_params: dict[str, str], query_params: dict[str, str]) -> dict[str, Any]:
        """Fetch historical data for a coin from CoinGecko."""
        try:
            self.validate_required_params(path_params, ["id"], "path_params")
            self.validate_required_params(query_params, ["date"], "query_params")

            base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/history"
            url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in query_params.items())

            headers = {"accept": "application/json", "x-cg-demo-api-key": self.coingecko_api_key}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            # Log the error and return empty dict to avoid crashing
            logging.exception(f"Error fetching historical data: {e!s}")
            return {}


class Trendmoon(Model):
    """This class implements the Trendmoon API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class HealthCheckService(Model):
    """This class provides health check data aggregation services."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_fsm_status(self) -> dict[str, Any]:
        """Get current FSM status information."""
        try:
            fsm_behaviour = cast("MindshareabciappFsmBehaviour", self.context.behaviours.main)
            current_state_name = fsm_behaviour.current

            seconds_since_last_transition = self._get_seconds_since_transition(fsm_behaviour)
            rounds, period_count = self._get_rounds_and_period_count(fsm_behaviour)

            if current_state_name:
                return self._build_active_state_status(
                    current_state_name, seconds_since_last_transition, rounds, period_count
                )

            return self._build_unknown_state_status(seconds_since_last_transition, rounds, period_count)

        except (AttributeError, KeyError, TypeError) as e:
            self.context.logger.exception(f"Error getting FSM state: {e}")
            return self._build_error_state_status()

    def _get_seconds_since_transition(self, fsm_behaviour: "MindshareabciappFsmBehaviour") -> float:
        """Calculate seconds since last transition."""
        last_timestamp = None

        if hasattr(fsm_behaviour, "last_transition_timestamp"):
            last_timestamp = fsm_behaviour.last_transition_timestamp
        elif hasattr(fsm_behaviour, "_last_transition_timestamp"):
            last_timestamp = fsm_behaviour._last_transition_timestamp  # noqa: SLF001

        if last_timestamp:
            current_time = datetime.now(UTC)
            time_diff = current_time - last_timestamp
            return time_diff.total_seconds()

        return 0.0

    def _get_rounds_and_period_count(self, fsm_behaviour: "MindshareabciappFsmBehaviour") -> tuple[list[str], int]:
        """Get round history and period count from FSM."""
        rounds = []
        period_count = 0

        if hasattr(fsm_behaviour, "previous_rounds"):
            rounds = fsm_behaviour.previous_rounds[-10:]  # Last 10 rounds
        elif hasattr(fsm_behaviour, "_previous_rounds"):
            rounds = fsm_behaviour._previous_rounds[-10:]  # noqa: SLF001

        if hasattr(fsm_behaviour, "period_count"):
            period_count = fsm_behaviour.period_count
        elif hasattr(fsm_behaviour, "_period_count"):
            period_count = fsm_behaviour._period_count  # noqa: SLF001

        return rounds, period_count

    def _build_active_state_status(
        self, current_state_name: str, seconds_since_last_transition: float, rounds: list[str], period_count: int
    ) -> dict[str, Any]:
        """Build status dict for active FSM state."""
        if current_state_name not in rounds:
            rounds.append(current_state_name)

        # Calculate if transitioning fast based on time in current state
        reset_and_pause_timeout = self.context.params.reset_pause_duration + MARGIN
        timeout_threshold = 2 * reset_and_pause_timeout
        is_transitioning_fast = seconds_since_last_transition < timeout_threshold

        return {
            "current_round": current_state_name,
            "is_transitioning_fast": is_transitioning_fast,
            "rounds": rounds,
            "seconds_since_last_transition": seconds_since_last_transition,
            "period_count": period_count,
        }

    def _build_unknown_state_status(
        self, seconds_since_last_transition: float, rounds: list[str], period_count: int
    ) -> dict[str, Any]:
        """Build status dict for unknown FSM state."""
        return {
            "current_round": "unknown",
            "is_transitioning_fast": False,
            "rounds": rounds or ["unknown"],
            "seconds_since_last_transition": seconds_since_last_transition,
            "period_count": period_count,
        }

    def _build_error_state_status(self) -> dict[str, Any]:
        """Build status dict for error state."""
        return {
            "current_round": "unknown",
            "is_transitioning_fast": False,
            "rounds": ["unknown"],
            "seconds_since_last_transition": None,
            "period_count": 0,
        }

    def build_rounds_info(self) -> dict[str, Any]:
        """Build rounds_info dict by extracting info from the actual FSM."""
        rounds_info = {}

        try:
            fsm_behaviour = cast("MindshareabciappFsmBehaviour", self.context.behaviours.main)

            # Extract transitions from the FSM's transitions attribute
            transitions_map = {}
            for source_state, event_map in fsm_behaviour.transitions.items():
                transitions_map[source_state] = {}
                for event, destination_state in event_map.items():
                    if event:  # event can be None
                        # Handle both string events and enum events
                        event_key = event.value if hasattr(event, "value") else str(event).lower()
                        transitions_map[source_state][event_key] = destination_state

            # Build rounds info using FSM data
            for state_name in fsm_behaviour.states:
                rounds_info[state_name] = {
                    "name": state_name.replace("round", "").replace("_", " ").title(),
                    "description": f"Handle {state_name.replace('round', '').replace('_', ' ')} operations",
                    "transitions": transitions_map.get(state_name, {}),
                }

        except (AttributeError, KeyError, TypeError) as e:
            self.context.logger.warning(f"Could not extract FSM transitions: {e}")
            # Fallback: basic info without transitions - need to import at runtime to avoid circular import
            try:
                from packages.xiuxiuxar.skills.mindshare_app.behaviours import MindshareabciappStates  # noqa: PLC0415

                for state in MindshareabciappStates:
                    state_name = state.value
                    rounds_info[state_name] = {
                        "name": state_name.replace("round", "").replace("_", " ").title(),
                        "description": f"Handle {state_name.replace('round', '').replace('_', ' ')} operations",
                        "transitions": {},
                    }
            except ImportError:
                self.context.logger.warning("Could not import MindshareabciappStates for fallback")

        return rounds_info

    def get_agent_health(self) -> dict[str, Any]:
        """Get agent health information."""
        return {
            "is_making_on_chain_transactions": False,
            "is_staking_kpi_met": True,
            "has_required_funds": True,
            "staking_status": "active",
        }

    def get_env_var_status(self) -> dict[str, Any]:
        """Get environment variable status."""
        return {"needs_update": False, "env_vars": {}}


class Params(Model):
    """This class implements the parameters for the Mindshare app."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.safe_contract_addresses = kwargs.pop("safe_contract_addresses", {})
        self.coingecko_api_key = kwargs.pop("coingecko_api_key", "")
        self.store_path = kwargs.pop("store_path", "./persistent_data")
        self.reset_pause_duration = kwargs.pop("reset_pause_duration", 10)
        super().__init__(*args, **kwargs)
