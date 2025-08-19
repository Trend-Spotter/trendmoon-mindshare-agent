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
    from collections.abc import Callable

    from packages.xiuxiuxar.skills.mindshare_app.behaviours import (
        MindshareabciappFsmBehaviour,
    )

MARGIN = 5


class FrozenMixin:  # pylint: disable=too-few-public-methods
    """Mixin for classes to enforce read-only attributes."""

    _frozen: bool = False

    def __delattr__(self, *args: Any) -> None:
        """Override __delattr__ to make object immutable."""
        if self._frozen:
            msg = "This object is frozen! To unfreeze switch `self._frozen` via `__dict__`."
            raise AttributeError(msg)
        super().__delattr__(*args)

    def __setattr__(self, *args: Any) -> None:
        """Override __setattr__ to make object immutable."""
        if self._frozen:
            msg = "This object is frozen! To unfreeze switch `self._frozen` via `__dict__`."
            raise AttributeError(msg)
        super().__setattr__(*args)


class Coingecko(Model):
    """This class implements the CoinGecko API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def validate_required_params(self, params: dict[str, str], required_keys: list[str], param_type: str) -> None:
        """Validate that required parameters are present and not None."""
        if params is None or params == {}:
            msg = f"{param_type} is required"
            raise ValueError(msg)

        for key in required_keys:
            if key not in params or params[key] is None:
                msg = f"{key} is required in {param_type}"
                raise ValueError(msg)

    def make_coingecko_request(self, base_url: str, query_params: dict[str, str]) -> Any:
        """Make a request to the CoinGecko API."""
        coingecko_api_key = self.context.params.coingecko_api_key
        if coingecko_api_key is None or coingecko_api_key == "":
            msg = "Coingecko API key is not set"
            raise ValueError(msg)

        if query_params is None or query_params == {}:
            url = base_url
        else:
            url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in query_params.items())

        headers = {"accept": "application/json", "x-cg-demo-api-key": coingecko_api_key}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        return response.json()

    def coin_ohlc_data_by_id(self, path_params: dict[str, str], query_params: dict[str, str]) -> list[list[Any]]:
        """Fetch OHLC data for a coin from CoinGecko."""
        try:
            self.validate_required_params(path_params, ["id"], "path_params")
            self.validate_required_params(query_params, ["vs_currency", "days"], "query_params")

            base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/ohlc"

            return self.make_coingecko_request(base_url, query_params)
        except Exception as e:
            self.context.logger.exception(f"Error fetching OHLC data: {e!s}")
            return None

    def coin_historical_chart_data_by_id(
        self, path_params: dict[str, str], query_params: dict[str, str]
    ) -> dict[str, Any]:
        """Fetch historical chart data for a coin from CoinGecko."""
        try:
            self.validate_required_params(path_params, ["id"], "path_params")
            self.validate_required_params(query_params, ["vs_currency", "days"], "query_params")

            base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/market_chart"

            return self.make_coingecko_request(base_url, query_params)
        except Exception as e:
            self.context.logger.exception(f"Error fetching historical chart data: {e!s}")
            return None

    def coin_price_by_id(self, query_params: dict[str, str]) -> dict[str, Any]:
        """Fetch price data for a coin from CoinGecko."""
        try:
            self.validate_required_params(query_params, ["vs_currencies"], "query_params")

            base_url = "https://api.coingecko.com/api/v3/simple/price"

            return self.make_coingecko_request(base_url, query_params)
        except Exception as e:
            self.context.logger.exception(f"Error fetching price data: {e!s}")
            return None

    def get_ohlcv_data(self, path_params: dict[str, str], query_params: dict[str, str]) -> list[list[Any]]:
        """Merge OHLCV and volume data by matching closest timestamps."""
        ohlcv_data = []

        ohlc_data = self.context.coingecko.coin_ohlc_data_by_id(path_params, query_params)
        volume_data = self.context.coingecko.coin_historical_chart_data_by_id(path_params, query_params)

        if not ohlc_data or not volume_data:
            self.context.logger.error("Missing data returned from CoinGecko")
            return None

        total_volumes = volume_data["total_volumes"]

        for ohlc in ohlc_data:
            ohlc_timestamp = ohlc[0]

            # Find the volume entry with the closest timestamp to the OHLC timestamp
            closest_volume_entry = min(total_volumes, key=lambda x: abs(x[0] - ohlc_timestamp))
            volume = closest_volume_entry[1]

            # Create OHLCV entry: [timestamp, open, high, low, close, volume]
            ohlcv_entry = [ohlc_timestamp] + ohlc[1:] + [volume]
            ohlcv_data.append(ohlcv_entry)

        return ohlcv_data


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
        self.coingecko_api_key = kwargs.pop("coingecko_api_key", "")
        self.data_sufficiency_threshold = kwargs.pop("data_sufficiency_threshold", 0.5)
        self.safe_contract_addresses = kwargs.pop("safe_contract_addresses", {})
        self.store_path = kwargs.pop("store_path", "./persistent_data")
        self.max_positions = kwargs.pop("max_positions", 10)
        self.max_exposure_per_position = kwargs.pop("max_exposure_per_position", 20.0)
        self.max_total_exposure = kwargs.pop("max_total_exposure", 80.0)
        self.min_capital_buffer = kwargs.pop("min_capital_buffer", 500.0)
        self.min_position_size_usdc = kwargs.pop("min_position_size_usdc", 100)
        self.reset_pause_duration = kwargs.pop("reset_pause_duration", 10)
        self.trading_strategy = kwargs.pop("trading_strategy", "balanced")
        self.max_slippage_bps = kwargs.pop("max_slippage_bps", 150)
        self.stop_loss_atr_multiplier = kwargs.pop("stop_loss_atr_multiplier", 1.5)
        self.take_profit_risk_ratio = kwargs.pop("take_profit_risk_ratio", 2.0)
        self.max_position_size_usdc = kwargs.pop("max_position_size_usdc", 5000.0)
        self.price_collection_timeout = kwargs.pop("price_collection_timeout", 15)
        self.cowswap_slippage_tolerance = kwargs.pop("cowswap_slippage_tolerance", 0.005)
        self.cowswap_timeout_seconds = kwargs.pop("cowswap_timeout_seconds", 300)
        self.moving_average_length = kwargs.pop("moving_average_length", 20)
        super().__init__(*args, **kwargs)


class Requests(Model, FrozenMixin):
    """Keep the current pending requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the state."""
        # mapping from dialogue reference nonce to callback
        self.request_id_to_callback: dict[str, Callable] = {}
        super().__init__(*args, **kwargs)
        self._frozen = True
