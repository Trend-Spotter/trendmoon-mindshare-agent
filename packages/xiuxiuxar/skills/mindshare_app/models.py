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

import time
from typing import TYPE_CHECKING, Any, cast
from datetime import UTC, datetime

import requests
from aea.skills.base import Model

from packages.eightballer.protocols.http.message import HttpMessage
from packages.eightballer.connections.http_client.connection import PUBLIC_ID as HTTP_CLIENT_PUBLIC_ID


if TYPE_CHECKING:
    from collections.abc import Callable

    from packages.xiuxiuxar.skills.mindshare_app.behaviours.round_behaviour import (
        MindshareabciappFsmBehaviour,
    )

MARGIN = 5
MINUTE_UNIX = 60


class FrozenMixin:
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


class APIRateLimiter:
    """Generic rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int,
        monthly_credits: int | None = None,
        requests_per_second: int | None = None,
        context: Any = None,
    ) -> None:
        """Initialize the rate limiter."""
        self._limit = self._remaining_limit = requests_per_minute
        self._credits = monthly_credits
        self.context = context

        self._remaining_credits = monthly_credits or float("inf")
        self._last_request_time = time.time()

        # Per-second rate limiting
        self._requests_per_second = requests_per_second
        self._second_requests = []  # Track timestamps of requests in current second

    @property
    def last_request_time(self) -> float:
        """Get the timestamp of the last request."""
        return self._last_request_time

    @property
    def rate_limited(self) -> bool:
        """Check whether we are rate limited."""
        return self.remaining_limit == 0 or self._second_rate_limited()

    @property
    def no_credits(self) -> bool:
        """Check whether all the credits have been spent."""
        return self.remaining_credits == 0

    @property
    def cannot_request(self) -> bool:
        """Check whether we cannot perform a request."""
        return self.rate_limited or self.no_credits

    @property
    def remaining_limit(self) -> int:
        """Get the remaining limit per minute."""
        return self._remaining_limit

    @property
    def remaining_credits(self) -> int:
        """Get the remaining requests' cap per month."""
        return self._remaining_credits

    def _second_rate_limited(self) -> bool:
        """Check if we're rate limited by per-second limits."""
        if not self._requests_per_second:
            return False

        current_time = time.time()
        # Remove requests older than 1 second
        self._second_requests = [t for t in self._second_requests if current_time - t < 1.0]

        return len(self._second_requests) >= self._requests_per_second

    def _update_limits(self) -> None:
        """Update the remaining limits and the credits if necessary."""
        current_time = time.time()
        time_passed = current_time - self._last_request_time
        limit_increase = int(time_passed / MINUTE_UNIX) * self._limit

        old_remaining_limit = self._remaining_limit
        self._remaining_limit = min(self._limit, self._remaining_limit + limit_increase)

        self.context.logger.debug(
            f"_update_limits(): "
            f"time_passed={time_passed:.2f}s, "
            f"limit_increase={limit_increase}, "
            f"old_remaining_limit={old_remaining_limit}, "
            f"new_remaining_limit={self._remaining_limit}"
        )

        # Reset monthly credits if month has passed
        if self._credits and self._can_reset_credits(current_time):
            old_credits = self._remaining_credits
            self._remaining_credits = self._credits
            self.context.logger.info(
                f"Monthly credits reset: " f"old_credits={old_credits}, " f"new_credits={self._remaining_credits}"
            )

    def _can_reset_credits(self, current_time: float) -> bool:
        """Check whether the monthly credits can be reset."""
        current_date = datetime.fromtimestamp(current_time, tz=UTC)

        # Handle year rollover properly
        if current_date.month == 12:
            next_month_year = current_date.year + 1
            next_month = 1
        else:
            next_month_year = current_date.year
            next_month = current_date.month + 1

        first_day_of_next_month = datetime(next_month_year, next_month, 1, tzinfo=UTC)
        return current_time >= first_day_of_next_month.timestamp()

    def _burn_credit(self) -> None:
        """Use one credit."""
        current_time = time.time()

        self.context.logger.debug(
            f"_burn_credit(): "
            f"Before burn: remaining_limit={self._remaining_limit}, "
            f"remaining_credits={self._remaining_credits}"
        )

        self._remaining_limit -= 1
        if self._credits:
            self._remaining_credits -= 1
        self._last_request_time = current_time

        # Track per-second request
        if self._requests_per_second:
            self._second_requests.append(current_time)
            self.context.logger.debug(
                f"Added per-second request tracking: "
                f"current_second_requests={len(self._second_requests)}, "
                f"requests_per_second_limit={self._requests_per_second}"
            )

    def check_and_burn(self) -> bool:
        """Check whether we can perform a new request, and if yes, update the remaining limit and credits."""
        # Log initial state
        self.context.logger.debug(
            f"check_and_burn() called - Current state: "
            f"remaining_limit={self._remaining_limit}, "
            f"remaining_credits={self._remaining_credits}, "
            f"last_request_time={self._last_request_time}, "
            f"time_since_last={time.time() - self._last_request_time:.2f}s"
        )

        self._update_limits()

        # Log state after limits update
        self.context.logger.debug(
            f"After _update_limits(): "
            f"remaining_limit={self._remaining_limit}, "
            f"remaining_credits={self._remaining_credits}"
        )

        # Check each condition separately with detailed logging
        is_rate_limited = self.rate_limited
        is_no_credits = self.no_credits
        cannot_request = self.cannot_request

        self.context.logger.debug(
            f"Rate limit checks: "
            f"rate_limited={is_rate_limited}, "
            f"no_credits={is_no_credits}, "
            f"cannot_request={cannot_request}"
        )

        if is_rate_limited:
            self.context.logger.warning(
                f"Rate limited detected: "
                f"remaining_limit={self._remaining_limit}, "
                f"second_rate_limited={self._second_rate_limited()}, "
                f"requests_per_second={self._requests_per_second}, "
                f"current_second_requests={len(self._second_requests)}"
            )

        if is_no_credits:
            self.context.logger.warning(
                f"No credits remaining: "
                f"remaining_credits={self._remaining_credits}, "
                f"monthly_credits={self._credits}"
            )

        if cannot_request:
            self.context.logger.warning(
                f"Cannot perform request - returning False. "
                f"Reason: rate_limited={is_rate_limited}, no_credits={is_no_credits}"
            )
            return False

        # If we can request, burn the credit and log
        self.context.logger.debug(
            f"Request allowed - burning credit. "
            f"Before burn: remaining_limit={self._remaining_limit}, "
            f"remaining_credits={self._remaining_credits}"
        )

        self._burn_credit()

        self.context.logger.debug(
            f"Credit burned successfully. "
            f"After burn: remaining_limit={self._remaining_limit}, "
            f"remaining_credits={self._remaining_credits}, "
            f"last_request_time={self._last_request_time}"
        )

        return True

    def calculate_wait_time(self) -> int:
        """Calculate the wait time in seconds to the next request."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # Check per-second rate limiting first
        if self._requests_per_second and self._second_rate_limited():
            # Wait until the oldest request in the current second window expires
            oldest_request_time = min(self._second_requests) if self._second_requests else current_time
            wait_time = max(0, 1.0 - (current_time - oldest_request_time))
            return int(wait_time) + 1  # Add 1 second buffer

        if self.remaining_limit == 0:
            # Wait for the remainder of the current minute
            wait_time = max(0, MINUTE_UNIX - int(time_since_last))
            return min(wait_time, MINUTE_UNIX)

        return 0

    def handle_rate_limit_response(self) -> None:
        """Handle rate limit response from API."""
        self._remaining_limit = 0
        self._last_request_time = time.time()

    def check_and_burn_with_wait(self) -> tuple[bool, float | None]:
        """Check whether we can perform a new request. If rate limited, return wait time instead of blocking.

        Returns:
            tuple: (can_proceed, wait_until_timestamp)
            - can_proceed: True if request can proceed, False if rate limited
            - wait_until_timestamp: None if no wait needed, future timestamp if should wait until then

        """
        # Check if we can make a request
        if self.check_and_burn():
            return True, None

        wait_time = self.calculate_wait_time()

        if wait_time == 0:
            self.context.logger.error("Rate limited but no wait time calculated")
            return False, None

        # Return the timestamp when the request can be retried
        wait_until_timestamp = time.time() + wait_time
        self.context.logger.info(f"Rate limited, should retry after {wait_time}s at {wait_until_timestamp}")

        return False, wait_until_timestamp


class Coingecko(Model):
    """This class implements the CoinGecko API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Setup the CoinGecko API client."""
        requests_per_minute = self.context.params.coingecko_rate_limit_per_minute
        monthly_credits = self.context.params.coingecko_monthly_credits
        self.rate_limiter = APIRateLimiter(requests_per_minute, monthly_credits, context=self.context)

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

    def submit_coingecko_request(self, base_url: str, query_params: dict[str, str]) -> tuple[str, float | None]:
        """Submit request to CoinGecko API using HTTP dialogue system."""
        can_proceed, wait_until = self.rate_limiter.check_and_burn_with_wait()

        if not can_proceed:
            if wait_until is not None:
                self.context.logger.info(f"CoinGecko rate limited, deferring request until {wait_until}")
                return "", wait_until
            self.context.logger.error(
                f"Failed to get rate limit approval. Remaining: {self.rate_limiter.remaining_limit}/min, "
                f"{self.rate_limiter.remaining_credits}/month"
            )
            msg = "Rate limited by CoinGecko API - unable to calculate wait time"
            raise ValueError(msg)

        # Add delay to avoid hitting rate limits
        time.sleep(2)

        coingecko_api_key = self.context.params.coingecko_api_key
        if coingecko_api_key is None or coingecko_api_key == "":
            msg = "Coingecko API key is not set"
            raise ValueError(msg)

        if query_params is None or query_params == {}:
            url = base_url
        else:
            url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in query_params.items())

        headers = {"accept": "application/json", "x-cg-demo-api-key": coingecko_api_key}
        headers_str = "\n".join(f"{k}: {v}" for k, v in headers.items())

        http_dialogues = self.context.http_dialogues
        request_http_message, http_dialogue = http_dialogues.create(
            counterparty=str(HTTP_CLIENT_PUBLIC_ID),
            performative=HttpMessage.Performative.REQUEST,
            method="GET",
            url=url,
            headers=headers_str,
            body=b"",
            version="",
        )
        self.context.outbox.put_message(message=request_http_message)

        # Return dialogue reference for tracking and no wait time (since we already waited)
        return http_dialogue.dialogue_label.dialogue_reference[0], None

    def handle_rate_limit_response(self) -> None:
        """Handle 429 rate limit response from CoinGecko."""
        self.context.logger.error(
            "Rate limited by CoinGecko API! Setting local rate limiter to prevent further requests."
        )
        self.rate_limiter.handle_rate_limit_response()

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
            ohlcv_entry = [ohlc_timestamp, *ohlc[1:], volume]
            ohlcv_data.append(ohlcv_entry)

        return ohlcv_data

    def submit_coin_ohlc_request(
        self, path_params: dict[str, str], query_params: dict[str, str]
    ) -> tuple[str, float | None]:
        """Submit async request for OHLC data from CoinGecko."""
        self.validate_required_params(path_params, ["id"], "path_params")
        self.validate_required_params(query_params, ["vs_currency", "days"], "query_params")

        base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/ohlc"
        return self.submit_coingecko_request(base_url, query_params)

    def submit_coin_historical_chart_request(
        self, path_params: dict[str, str], query_params: dict[str, str]
    ) -> tuple[str, float | None]:
        """Submit async request for historical chart data from CoinGecko."""
        self.validate_required_params(path_params, ["id"], "path_params")
        self.validate_required_params(query_params, ["vs_currency", "days"], "query_params")

        base_url = f"https://api.coingecko.com/api/v3/coins/{path_params['id']}/market_chart"
        return self.submit_coingecko_request(base_url, query_params)

    def submit_coin_price_request(self, query_params: dict[str, str]) -> tuple[str, float | None]:
        """Submit async request for price data from CoinGecko."""
        self.validate_required_params(query_params, ["vs_currencies"], "query_params")

        base_url = "https://api.coingecko.com/api/v3/simple/price"
        return self.submit_coingecko_request(base_url, query_params)


class Trendmoon(Model):
    """This class implements the Trendmoon API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Setup the Trendmoon API client."""
        requests_per_minute = self.context.params.trendmoon_rate_limit_per_minute
        monthly_credits = self.context.params.trendmoon_monthly_credits
        self.rate_limiter = APIRateLimiter(
            requests_per_minute,
            monthly_credits,
            requests_per_second=self.context.params.trendmoon_rate_limit_per_second,
            context=self.context,
        )

    def submit_trendmoon_request(self, endpoint: str, query_params: dict[str, str]) -> tuple[str, float | None]:
        """Submit request to TrendMoon API using HTTP dialogue system.

        Returns:
            tuple: (request_id, wait_until_timestamp)
            - request_id: The dialogue reference for tracking, or empty string if rate limited
            - wait_until_timestamp: Unix timestamp to wait until, or None if no wait needed

        """
        can_proceed, wait_until = self.rate_limiter.check_and_burn_with_wait()

        if not can_proceed:
            if wait_until is None:
                self.context.logger.error(
                    f"Failed to get rate limit approval after retries. "
                    f"Remaining: {self.rate_limiter.remaining_limit}/min"
                )
                msg = "Rate limited by TrendMoon API - max retries exceeded"
                raise ValueError(msg)
            return "", wait_until

        # Add small delay to avoid hitting rate limits
        time.sleep(0.2)

        url = f"https://api.qa.trendmoon.ai{endpoint}"

        if query_params:
            filtered_params = {k: v for k, v in query_params.items() if v is not None}
            if filtered_params:
                url += "?" + "&".join(f"{k}={v}" for k, v in filtered_params.items())

        headers = {"accept": "application/json", "Api-key": self.context.params.trendmoon_api_key}
        headers_str = "\n".join(f"{k}: {v}" for k, v in headers.items())

        http_dialogues = self.context.http_dialogues
        request_message, dialogue = http_dialogues.create(
            counterparty=str(HTTP_CLIENT_PUBLIC_ID),
            performative=HttpMessage.Performative.REQUEST,
            method="GET",
            url=url,
            headers=headers_str,
            body=b"",
            version="",
        )

        self.context.outbox.put_message(message=request_message)

        return dialogue.dialogue_label.dialogue_reference[0], None

    def submit_coin_trends_request(
        self,
        symbol: str | None = None,
        coin_ids: list[str] | None = None,
        contract_addresses: list[str] | None = None,
        date_interval: int | None = None,
        time_interval: str | None = "1h",
    ) -> tuple[str, float | None]:
        """Submit request for coin trends."""
        query_params = {
            "symbol": symbol,
            "coin_ids": coin_ids,
            "contract_addresses": contract_addresses,
            "date_interval": date_interval,
            "time_interval": time_interval,
        }

        self.context.logger.debug(f"TrendMoon request params: {query_params}")
        return self.submit_trendmoon_request(endpoint="/social/trend", query_params=query_params)

    def handle_rate_limit_response(self) -> None:
        """Handle 429 rate limit response from TrendMoon."""
        self.context.logger.error(
            "Rate limited by TrendMoon API! Setting local rate limiter to prevent further requests."
        )
        self.rate_limiter.handle_rate_limit_response()


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
        # Data sources
        self.coingecko_api_key = kwargs.pop("coingecko_api_key", "")
        self.trendmoon_api_key = kwargs.pop("trendmoon_api_key", "")
        self.coingecko_rate_limit_per_minute = kwargs.pop("coingecko_rate_limit_per_minute", 50)
        self.coingecko_monthly_credits = kwargs.pop("coingecko_monthly_credits", 100000)
        self.trendmoon_rate_limit_per_minute = kwargs.pop("trendmoon_rate_limit_per_minute", 100)
        self.trendmoon_rate_limit_per_second = kwargs.pop("trendmoon_rate_limit_per_second", 5)
        self.trendmoon_monthly_credits = kwargs.pop("trendmoon_monthly_credits", 10000)

        # Staking
        self.staking_chain = kwargs.pop("staking_chain", "base")
        self.staking_threshold_period = kwargs.pop("staking_threshold_period", 22)
        self.min_num_of_safe_tx_required = kwargs.pop("min_num_of_safe_tx_required", 5)
        self.staking_token_contract_address = kwargs.pop(
            "staking_token_contract_address", "0xEB5638eefE289691EcE01943f768EDBF96258a80"
        )
        self.service_id = kwargs.pop("service_id", "mindshare")
        self.on_chain_service_id = kwargs.pop("on_chain_service_id", None)

        # Trading
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
        self.stop_loss_pct = kwargs.pop("stop_loss_pct", 0.84)
        self.take_profit_risk_ratio = kwargs.pop("take_profit_risk_ratio", 2.0)
        self.trailing_stop_loss_activation_level = kwargs.pop("trailing_stop_loss_activation_level", 1.25)
        self.trailing_stop_loss_pct = kwargs.pop("trailing_stop_loss_pct", 0.97)
        self.max_position_size_usdc = kwargs.pop("max_position_size_usdc", 5000.0)
        self.price_collection_timeout = kwargs.pop("price_collection_timeout", 15)
        self.cowswap_slippage_tolerance = kwargs.pop("cowswap_slippage_tolerance", 0.005)
        self.cowswap_timeout_seconds = kwargs.pop("cowswap_timeout_seconds", 300)
        self.moving_average_length = kwargs.pop("moving_average_length", 20)
        self.rsi_period_length = kwargs.pop("rsi_period_length", 14)
        self.rsi_lower_limit = kwargs.pop("rsi_lower_limit", 39)
        self.rsi_upper_limit = kwargs.pop("rsi_upper_limit", 66)
        self.rsi_overbought_limit = kwargs.pop("rsi_overbought_limit", 79)
        self.macd_fast_period = kwargs.pop("macd_fast_period", 12)
        self.macd_slow_period = kwargs.pop("macd_slow_period", 26)
        self.macd_signal_period = kwargs.pop("macd_signal_period", 9)
        self.adx_period_length = kwargs.pop("adx_period_length", 14)
        self.adx_threshold = kwargs.pop("adx_threshold", 41)
        super().__init__(*args, **kwargs)


class Requests(Model, FrozenMixin):
    """Keep the current pending requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the state."""
        # mapping from dialogue reference nonce to callback
        self.request_id_to_callback: dict[str, Callable] = {}
        super().__init__(*args, **kwargs)
        self._frozen = True
