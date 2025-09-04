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
import time
import operator
from typing import Any
from datetime import UTC, datetime, timedelta
from collections.abc import Generator

from autonomy.deploy.constants import DEFAULT_ENCODING

from packages.eightballer.protocols.http.message import HttpMessage
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    ALLOWED_ASSETS,
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


MAX_COLLECTION_TIME = 300  # 5 minutes
TIMEOUT_SECONDS = 120  # HTTP response timeout
RETRY_CHECK_INTERVAL = 10  # seconds between retry checks
MAX_RETRIES = 3
OHLCV_DATA_DAYS = 30
OHLCV_MAX_AGE_HOURS = 4
OHLCV_MAX_AGE_BUFFER_MINUTES = 30
SOCIAL_UPDATE_THRESHOLD_HOURS = 1
DEFAULT_RATE_LIMIT_WAIT_TIME = 60


class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, message: str, wait_until: float | None = None, wait_time: int = DEFAULT_RATE_LIMIT_WAIT_TIME):
        super().__init__(message)
        self.wait_time = wait_time
        self.wait_until = wait_until


class DataCollectionRound(BaseState):
    """This class implements the behaviour of the state DataCollectionRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = MindshareabciappStates.DATACOLLECTIONROUND
        self.started: bool = False
        self.collected_data: dict[str, Any] = {}
        self.started_at: datetime | None = None
        self.collection_initialized = False
        self.pending_tokens: list[dict[str, str]] = []
        self.completed_tokens: list[dict[str, str]] = []
        self.failed_tokens: list[dict[str, str]] = []
        self.current_token_index: int = 0
        self.rate_limiting_occurred: bool = False
        self.tokens_needing_ohlcv_update: set[str] = set()
        self.tokens_needing_social_update: set[str] = set()

        # Generator state tracking
        self._collection_generator: Generator[None, None, bool] | None = None
        self._generator_completed: bool = False

        # Async HTTP tracking
        self.pending_http_requests: dict[str, dict] = {}
        self.received_responses: list[dict[str, Any]] = []
        self.async_requests_submitted: bool = False
        self.total_expected_responses: int = 0

        # Add to DataCollectionRound state
        self.failed_requests: dict[str, dict] = {}  # Track failed requests for retry

    def setup(self) -> None:
        """Perform the setup."""
        super().setup()
        self.started = False
        self._is_done = False
        self.collection_initialized = False
        self.pending_tokens = []
        self.completed_tokens = []
        self.failed_tokens = []
        self.current_token_index = 0
        self.rate_limiting_occurred = False
        self.collected_data = {}
        self.started_at = None
        self.tokens_needing_ohlcv_update = set()
        self.tokens_needing_social_update = set()

        # Reset generator state
        self._collection_generator = None
        self._generator_completed = False

        # Reset async HTTP tracking
        self.pending_http_requests = {}
        self.received_responses = []
        self.async_requests_submitted = False
        self.total_expected_responses = 0

        # Add to DataCollectionRound state
        self.failed_requests = {}  # Track failed requests for retry

    def act(self) -> None:
        """Perform the act using generator pattern to yield control."""
        try:
            if not self.started:
                self.context.logger.info(f"Entering {self._state} state.")
                self.started = True

            # Check for timeout to prevent indefinite execution
            if hasattr(self, "started_at") and self.started_at:
                elapsed_time = (datetime.now(UTC) - self.started_at).total_seconds()
                if elapsed_time > MAX_COLLECTION_TIME:
                    self.context.logger.warning(
                        f"Data collection timeout after {elapsed_time:.1f}s, proceeding with collected data"
                    )
                    self._finalize_collection()
                    self._event = (
                        MindshareabciappEvents.DONE if self._is_data_sufficient() else MindshareabciappEvents.ERROR
                    )
                    self._is_done = True
                    return

            # Initialize generator on first call
            if self._collection_generator is None:
                self._collection_generator = self._collect_all_data()
                self._generator_completed = False

            # Continue generator execution
            if not self._generator_completed:
                try:
                    next(self._collection_generator)
                    # Generator yielded, will continue next round
                    return
                except StopIteration as e:
                    # Generator completed
                    self._generator_completed = True
                    success = e.value if hasattr(e, "value") else True

                    # Only finalize if the generator indicates success (no pending retries)
                    if success:
                        self.context.logger.info("Data collection completed, finalizing...")
                        self._finalize_collection()
                        self._event = (
                            MindshareabciappEvents.DONE if self._is_data_sufficient() else MindshareabciappEvents.ERROR
                        )
                        self._is_done = True
                    else:
                        # Generator returned False, indicating pending retries
                        self.context.logger.info("Data collection has pending retries, continuing...")
                        # Reset generator for next iteration
                        self._collection_generator = None
                        self._generator_completed = False
                        # Don't set _is_done, let the round continue

        except Exception as e:
            self.context.logger.exception(f"Data collection failed: {e}")
            self.context.error_context = {
                "error_type": "data_collection_error",
                "error_message": str(e),
                "originating_round": str(self._state),
                "current_token": self.pending_tokens[self.current_token_index]["symbol"]
                if self.current_token_index < len(self.pending_tokens)
                else "unknown",
            }
            self._event = MindshareabciappEvents.ERROR
            self._is_done = True

    def _initialize_collection(self) -> None:
        """Initialize data collection on first run."""
        if self.collection_initialized:
            return

        self.context.logger.info("Initializing data collection...")
        self.pending_tokens = ALLOWED_ASSETS["base"].copy()
        self.completed_tokens = []
        self.failed_tokens = []
        self.current_token_index = 0

        self.context.logger.info(
            f"Processing {len(self.pending_tokens)} tokens: {[t['symbol'] for t in self.pending_tokens]}"
        )

        existing_data = self._load_and_analyze_existing_data()

        if existing_data:
            self.collected_data = existing_data
        else:
            self.context.logger.info("No existing data found, initializing fresh collection")
            self._initialize_fresh_data_structure()

        self.started_at = datetime.now(UTC)
        self.collection_initialized = True

    def _initialize_fresh_data_structure(self) -> None:
        """Initialize a fresh data structure."""
        self.collected_data = {
            "ohlcv": {},
            "current_prices": {},
            "market_data": {},
            "social_data": {},
            "collection_timestamp": datetime.now(UTC).isoformat(),
            "last_ohlcv_update": {},
            "last_social_update": {},
            "errors": [],
            "cache_hits": 0,
            "api_calls": 0,
        }

        for token in self.pending_tokens:
            symbol = token["symbol"]
            self.tokens_needing_ohlcv_update.add(symbol)
            self.tokens_needing_social_update.add(symbol)

    def _load_and_analyze_existing_data(self) -> dict | None:
        """Load existing data and analyze what needs updating."""
        try:
            if not self.context.store_path:
                self.context.logger.warning("No store path available")
                return None

            data_file = self.context.store_path / "collected_data.json"
            if not data_file.exists():
                self.context.logger.info(f"Data file does not exist: {data_file}")
                return None

            with open(data_file, encoding=DEFAULT_ENCODING) as f:
                existing_data = json.load(f)

            # Ensure tracking fields exist
            if "last_ohlcv_update" not in existing_data:
                existing_data["last_ohlcv_update"] = {}
            if "last_social_update" not in existing_data:
                existing_data["last_social_update"] = {}

            current_time = datetime.now(UTC)

            for token in self.pending_tokens:
                symbol = token["symbol"]

                ohlcv_needs_update = self._needs_ohlcv_update(existing_data, symbol, current_time)
                if ohlcv_needs_update:
                    self.tokens_needing_ohlcv_update.add(symbol)

                # Check social data freshness
                social_needs_update = self._needs_social_update(existing_data, symbol, current_time)
                if social_needs_update:
                    self.tokens_needing_social_update.add(symbol)

            return existing_data

        except Exception as e:
            self.context.logger.exception(f"Failed to load existing data: {e}")
            return None

    def _needs_ohlcv_update(self, existing_data: dict, symbol: str, current_time: datetime) -> bool:
        """Check if OHLCV data needs updating."""
        # Check if we have OHLCV data for this symbol
        if symbol not in existing_data.get("ohlcv", {}):
            self.context.logger.debug(f"{symbol}: No OHLCV data exists")
            return True

        ohlcv_data = existing_data["ohlcv"][symbol]
        if not ohlcv_data:
            self.context.logger.debug(f"{symbol}: OHLCV data is empty")
            return True

        try:
            # OHLCV data format: [[timestamp, open, high, low, close, volume], ...]
            latest_candle_timestamp = max(candle[0] for candle in ohlcv_data)
            latest_candle_time = datetime.fromtimestamp(latest_candle_timestamp / 1000, UTC)

            # CoinGecko provides 4-hour candles, check if we're missing recent candles
            time_since_latest = current_time - latest_candle_time
            expected_max_age = timedelta(hours=OHLCV_MAX_AGE_HOURS, minutes=OHLCV_MAX_AGE_BUFFER_MINUTES)

            needs_update = time_since_latest > expected_max_age

            self.context.logger.debug(
                f"{symbol}: Latest candle {latest_candle_time.isoformat()}, "
                f"age: {time_since_latest}, threshold: {expected_max_age}, "
                f"needs_update: {needs_update}"
            )

            return needs_update

        except (ValueError, TypeError, IndexError) as e:
            self.context.logger.warning(f"Error analyzing OHLCV data for {symbol}: {e}")
            return True

    def _needs_social_update(self, existing_data: dict, symbol: str, current_time: datetime) -> bool:
        """Determine if social data needs updating for a specific token."""
        # First check if we have social data at all
        if symbol not in existing_data.get("social_data", {}):
            self.context.logger.debug(f"{symbol}: No social data exists")
            return True

        last_update_str = existing_data.get("last_social_update", {}).get(symbol)

        if not last_update_str:
            return True

        try:
            last_update = datetime.fromisoformat(last_update_str)
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=UTC)

            time_since_update = current_time - last_update
            update_threshold = timedelta(hours=SOCIAL_UPDATE_THRESHOLD_HOURS)

            needs_update = time_since_update > update_threshold

            self.context.logger.debug(
                f"{symbol}: Last social update {last_update.isoformat()}, "
                f"age: {time_since_update}, threshold: {update_threshold}, "
                f"needs_update: {needs_update}"
            )

            return needs_update

        except (ValueError, TypeError) as e:
            self.context.logger.warning(f"Error parsing social update time for {symbol}: {e}")
            return True

    def _collect_all_data(self) -> Generator[None, None, bool]:
        """Generator that yields control between token processing operations."""
        # Initialize collection on first run
        if not self.collection_initialized:
            self._initialize_collection()
            yield  # Yield after initialization

        # Phase 1: Submit all async requests for pending tokens
        if not self.async_requests_submitted and self.pending_tokens:
            self.context.logger.info("Phase 1: Submitting async requests for all tokens")
            for token_info in self.pending_tokens:
                self.context.logger.info(
                    f"Submitting requests for {token_info['symbol']} "
                    f"({self.current_token_index + 1}/{len(self.pending_tokens)})"
                )

                try:
                    # Submit async requests without blocking
                    self._collect_token_data(token_info)
                    self.completed_tokens.append(token_info)

                except RateLimitError:
                    self.rate_limiting_occurred = True
                    self.context.logger.info(f"Rate limited for {token_info['symbol']}, will retry in phase 2")
                    # Token is already added to failed_requests by _collect_token_data
                    continue
                except (ValueError, TypeError, OSError, ConnectionError) as e:
                    self.context.logger.warning(f"Failed to submit requests for {token_info['symbol']}: {e}")
                    self.failed_tokens.append(token_info)
                    self.collected_data["errors"].append(
                        f"Error processing {token_info['symbol']} ({token_info['address']}): {e}"
                    )

                self.current_token_index += 1
                yield  # Yield after each token to allow message processing

            # Keep failed_requests for retrying in phase 2 - don't move to failed_tokens yet
            if self.failed_requests:
                self.context.logger.info(
                    f"Phase 1 complete: {len(self.failed_requests)} tokens rate limited, will retry in phase 2"
                )

            self.async_requests_submitted = True
            self.total_expected_responses = len(self.pending_http_requests)
            total_initial_requests = self.total_expected_responses
            rate_limited_count = len(self.failed_requests)
            self.context.logger.info(
                f"Phase 1 complete: Submitted {total_initial_requests} async requests, "
                f"{rate_limited_count} rate limited for retry"
            )
            yield

        # Phase 2: Wait for and process all responses, including retries
        self.context.logger.info("Phase 2: Waiting for async responses and handling retries")
        yield from self._wait_for_async_responses()

        # All tokens processed successfully
        total_completed = len(self.completed_tokens)
        total_failed = len(self.failed_tokens)
        total_tokens = len(ALLOWED_ASSETS["base"])
        self.context.logger.info(
            f"All async data collection completed: {total_completed}/{total_tokens} tokens successful, "
            f"{total_failed} failed"
        )
        return True  # noqa: B901  # Intentional PEP 380-style generator return

    def _collect_token_data(self, token_info: dict[str, str]) -> None:
        """Submit async requests for a single token."""
        symbol = token_info["symbol"]
        address = token_info["address"]

        self.context.logger.info(f"Processing {symbol} ({address})")

        try:
            # Submit async requests for data that needs updating
            if symbol in self.tokens_needing_ohlcv_update:
                self._submit_async_ohlcv_request(token_info)
                self.tokens_needing_ohlcv_update.discard(symbol)
            else:
                self.context.logger.info(f"Skipping OHLCV update for {symbol} (data is fresh)")

            if symbol in self.tokens_needing_social_update:
                self._submit_async_social_request(token_info)
                self.tokens_needing_social_update.discard(symbol)
            else:
                self.context.logger.info(f"Skipping social update for {symbol} (data is fresh)")

            # Always submit async price request for current data
            self._submit_async_price_request(token_info)

            self.context.logger.debug(f"Successfully submitted async requests for {symbol}")

        except Exception as e:
            self.context.logger.warning(f"Failed to submit requests for {symbol}: {e}")
            raise  # Re-raise to be handled by the generator

    def _submit_async_ohlcv_request(self, token_info: dict[str, str]) -> None:
        """Submit async request for OHLCV data."""
        symbol = token_info["symbol"]
        coingecko_id = token_info["coingecko_id"]

        self.context.logger.info(f"Submitting async OHLCV request for {symbol}")

        path_params = {"id": coingecko_id}
        query_params = {"vs_currency": "usd", "days": OHLCV_DATA_DAYS}

        ohlc_request_id, ohlc_wait_until = self.context.coingecko.submit_coin_ohlc_request(path_params, query_params)
        volume_request_id, volume_wait_until = self.context.coingecko.submit_coin_historical_chart_request(
            path_params, query_params
        )

        # Check if we need to wait for rate limits
        if ohlc_wait_until is not None or volume_wait_until is not None:
            wait_until = max(ohlc_wait_until or 0, volume_wait_until or 0)
            self._handle_rate_limit(symbol, token_info, "ohlcv", wait_until)

        # Track requests for identification
        self.pending_http_requests[ohlc_request_id] = {
            "request_type": "ohlc",
            "token_symbol": symbol,
            "coingecko_id": coingecko_id,
        }
        self.pending_http_requests[volume_request_id] = {
            "request_type": "volume",
            "token_symbol": symbol,
            "coingecko_id": coingecko_id,
        }

        self.collected_data["api_calls"] += 2

    def _handle_rate_limit(self, symbol: str, token_info: dict[str, str], request_type: str, wait_until: float) -> None:
        """Handle rate limiting by storing request for retry and raising RateLimitError."""
        self.context.logger.info(
            f"Rate limited for {symbol} {request_type}, will retry at "
            f"{datetime.fromtimestamp(wait_until, tz=UTC).strftime('%H:%M:%S')}"
        )
        # Store the token for retry later
        self.failed_requests[symbol] = {
            "token_info": token_info,
            "retry_after": wait_until,
            "request_type": request_type,
            "retry_count": 0,
        }
        msg = f"Rate limited for {symbol} {request_type}, retry after {wait_until}"
        raise RateLimitError(msg, wait_until=wait_until)

    def _process_ohlcv_response(self, token_symbol: str, ohlc_data: list, volume_data: dict) -> None:
        """Process OHLCV response data."""
        if not ohlc_data or not volume_data:
            self.context.logger.error(f"Missing OHLCV data for {token_symbol}")
            return

        # Merge OHLC and volume data like the original method
        total_volumes = volume_data.get("total_volumes", [])
        ohlcv_data = []

        for ohlc in ohlc_data:
            ohlc_timestamp = ohlc[0]
            closest_volume_entry = min(total_volumes, key=lambda x: abs(x[0] - ohlc_timestamp))
            volume = closest_volume_entry[1]
            ohlcv_entry = [ohlc_timestamp, *ohlc[1:], volume]
            ohlcv_data.append(ohlcv_entry)

        # Merge with existing data if any
        existing_ohlcv = self.collected_data["ohlcv"].get(token_symbol, [])
        if existing_ohlcv:
            merged_data = self._merge_ohlcv_data(existing_ohlcv, ohlcv_data)
            self.collected_data["ohlcv"][token_symbol] = merged_data
            self.context.logger.info(f"Merged OHLCV data for {token_symbol}: {len(merged_data)} total candles")
        else:
            self.collected_data["ohlcv"][token_symbol] = ohlcv_data
            self.context.logger.info(f"Set fresh OHLCV data for {token_symbol}: {len(ohlcv_data)} candles")

        # Update tracking timestamp
        self.collected_data["last_ohlcv_update"][token_symbol] = datetime.now(UTC).isoformat()

    def _merge_ohlcv_data(self, existing_data: list, fresh_data: list) -> list:
        """Merge existing and fresh OHLCV data, avoiding duplicates."""
        if not existing_data:
            return fresh_data

        if not fresh_data:
            return existing_data

        existing_timestamps = {candle[0] for candle in existing_data}

        new_candles = [candle for candle in fresh_data if candle[0] not in existing_timestamps]

        merged_data = existing_data + new_candles
        merged_data.sort(key=operator.itemgetter(0))

        self.context.logger.info(
            f"OHLCV merge: {len(existing_data)} existing + {len(new_candles)} new = {len(merged_data)} total candles"
        )

        return merged_data

    def _submit_async_social_request(self, token_info: dict[str, str]) -> None:
        """Submit async request for social data."""
        symbol = token_info["symbol"]

        self.context.logger.info(f"Submitting async social data request for {symbol}")

        trends_request_id, wait_until = self.context.trendmoon.submit_coin_trends_request(
            symbol=symbol, time_interval="1h", date_interval=30
        )

        # Check if we need to wait for rate limits
        if wait_until is not None:
            self._handle_rate_limit(symbol, token_info, "social", wait_until)

        # Track request for identification
        self.pending_http_requests[trends_request_id] = {"request_type": "social", "token_symbol": symbol}

        self.collected_data["api_calls"] += 1

    def _process_social_response(self, token_symbol: str, social_data: dict) -> None:
        """Process social response data."""
        if social_data:
            self.collected_data["social_data"][token_symbol] = social_data
            self.collected_data["last_social_update"][token_symbol] = datetime.now(UTC).isoformat()
            self.context.logger.info(f"Updated social data for {token_symbol}")
        else:
            self.context.logger.warning(f"No social data available for {token_symbol}")

    def _submit_async_price_request(self, token_info: dict[str, str]) -> None:
        """Submit async request for current price data."""
        symbol = token_info["symbol"]
        coingecko_id = token_info["coingecko_id"]

        self.context.logger.info(f"Submitting async price request for {symbol}")

        query_params = {
            "ids": coingecko_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        price_request_id, wait_until = self.context.coingecko.submit_coin_price_request(query_params)

        # Check if we need to wait for rate limits
        if wait_until is not None:
            self._handle_rate_limit(symbol, token_info, "price", wait_until)

        # Track request for identification
        self.pending_http_requests[price_request_id] = {
            "request_type": "price",
            "token_symbol": symbol,
            "coingecko_id": coingecko_id,
        }

        self.collected_data["api_calls"] += 1

    def _process_price_response(self, token_symbol: str, coingecko_id: str, price_data: dict) -> None:
        """Process current price response data."""
        if price_data and coingecko_id in price_data:
            self.collected_data["current_prices"][token_symbol] = price_data[coingecko_id]
            self.context.logger.info(f"Updated price data for {token_symbol}")
        else:
            self.context.logger.warning(f"No price data available for {token_symbol}")

    def handle_http_response(self, message: HttpMessage) -> bool:
        """Handle incoming HTTP response messages."""
        if message.performative != HttpMessage.Performative.RESPONSE:
            return False

        # Find matching request by dialogue reference
        if not message.dialogue_reference:
            return False

        request_id = message.dialogue_reference[0]
        request_info = self.pending_http_requests.get(request_id)

        if not request_info:
            return False

        # Parse response
        if message.status_code >= 400:
            response_text = message.body.decode(DEFAULT_ENCODING) if message.body else "No response body"
            self.context.logger.error(
                f"HTTP error for {request_info['request_type']}: {message.status_code} - {response_text}"
            )
            self.context.logger.error(f"Request was for token: {request_info.get('token_symbol', 'unknown')}")
        else:
            response_data = self._parse_response_body(message.body)
            self.received_responses.append(
                {
                    "request_type": request_info["request_type"],
                    "token_symbol": request_info.get("token_symbol", "unknown"),
                    "coingecko_id": request_info.get("coingecko_id", None),
                    "data": response_data,
                    "status_code": message.status_code,
                }
            )
            self.context.logger.info(
                f"Successfully received {request_info['request_type']} response "
                f"for {request_info.get('token_symbol', 'unknown')}"
            )
            self.context.logger.debug(f"Response data size: {len(response_data) if response_data else 0} items")

        # Remove from pending requests
        del self.pending_http_requests[request_id]

        return True

    def _parse_response_body(self, body: bytes) -> Any:
        """Parse HTTP response body as JSON."""
        if body:
            try:
                return json.loads(body.decode(DEFAULT_ENCODING))
            except json.JSONDecodeError:
                self.context.logger.exception("Failed to parse JSON response")
                return None
        return None

    def _process_all_responses(self) -> None:
        """Process all received HTTP responses."""
        # Group responses by token and type
        token_responses = {}

        for response in self.received_responses:
            token_symbol = response["token_symbol"]
            request_type = response["request_type"]

            if token_symbol not in token_responses:
                token_responses[token_symbol] = {}
            token_responses[token_symbol][request_type] = response

        # Process responses for each token
        for token_symbol, responses in token_responses.items():
            try:
                # Process OHLCV data (needs both ohlc and volume)
                if "ohlc" in responses and "volume" in responses:
                    ohlc_data = responses["ohlc"]["data"]
                    volume_data = responses["volume"]["data"]
                    self._process_ohlcv_response(token_symbol, ohlc_data, volume_data)

                # Process social data
                if "social" in responses:
                    social_data = responses["social"]["data"]
                    self._process_social_response(token_symbol, social_data)

                # Process price data
                if "price" in responses:
                    price_data = responses["price"]["data"]
                    coingecko_id = responses["price"]["coingecko_id"]
                    self._process_price_response(token_symbol, coingecko_id, price_data)

            except Exception as e:
                self.context.logger.exception(f"Error processing responses for {token_symbol}: {e}")

    def _wait_for_async_responses(self) -> Generator:
        """Wait for all async HTTP responses to be received."""
        if not self.pending_http_requests and not self.failed_requests:
            return

        self.context.logger.info(f"Waiting for {len(self.pending_http_requests)} async HTTP responses...")
        start_time = datetime.now(UTC)
        last_retry_check = 0  # Track when we last checked for retries to reduce logging

        # Recalculate expected responses based on what was actually submitted
        expected_from_requests = len(self.pending_http_requests)
        expected_from_retries = 0  # Will be updated when retries are successful

        while (len(self.received_responses) < expected_from_requests + expected_from_retries) or self.failed_requests:
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > TIMEOUT_SECONDS:
                self.context.logger.warning(
                    f"Timeout waiting for HTTP responses. Got {len(self.received_responses)}, "
                    f"expected {expected_from_requests + expected_from_retries}"
                )
                break

            # Check for failed requests that can be retried (but not too frequently)
            current_time = time.time()
            if self.failed_requests and (current_time - last_retry_check) > RETRY_CHECK_INTERVAL:
                if self._can_retry_failed_requests():
                    retry_count = len(
                        [req for req in self.failed_requests.values() if req.get("retry_count", 0) < MAX_RETRIES]
                    )
                    self.context.logger.info(f"Retrying {retry_count} failed requests...")
                    initial_pending_count = len(self.pending_http_requests)
                    yield from self._retry_failed_requests()
                    # Update expected count based on successful retries
                    new_pending_count = len(self.pending_http_requests)
                    if new_pending_count > initial_pending_count:
                        expected_from_retries += new_pending_count - initial_pending_count
                last_retry_check = current_time

            # Break if no more progress can be made
            if not self.pending_http_requests and not self.failed_requests:
                break

            yield

        self.context.logger.info(f"Received {len(self.received_responses)} responses")
        self._process_all_responses()

        # After processing responses, move any remaining failed requests to failed_tokens
        if self.failed_requests:
            self.context.logger.info(f"Moving {len(self.failed_requests)} remaining failed requests to failed tokens")
            for request_info in self.failed_requests.values():
                token_info = request_info["token_info"]
                if token_info not in self.failed_tokens:
                    self.failed_tokens.append(token_info)
            # Clear failed_requests as they are now in failed_tokens
            self.failed_requests.clear()

    def _can_retry_failed_requests(self) -> bool:
        """Check if any failed requests can be retried."""
        current_time = time.time()
        for request_info in self.failed_requests.values():
            retry_count = request_info.get("retry_count", 0)
            retry_after = request_info.get("retry_after")

            # Check if we haven't exceeded max retries and the wait time has passed
            if retry_count < MAX_RETRIES and (retry_after is None or current_time >= retry_after):
                return True
        return False

    def _should_retry_request(self, request_info: dict, current_time: float) -> bool:
        """Check if a request should be retried based on timing and count."""
        retry_count = request_info.get("retry_count", 0)
        retry_after = request_info.get("retry_after")

        # Skip if we haven't waited long enough
        if retry_after is not None and current_time < retry_after:
            return False

        return retry_count < MAX_RETRIES

    def _prepare_request_for_retry(self, token_info: dict[str, str], request_type: str) -> None:
        """Prepare a token's data requirements for retry based on request type."""
        symbol_name = token_info["symbol"]

        # Re-add to the appropriate update set based on request type
        if request_type == "ohlcv":
            self.tokens_needing_ohlcv_update.add(symbol_name)
        elif request_type == "social":
            self.tokens_needing_social_update.add(symbol_name)

    def _handle_retry_success(self, symbol: str, token_info: dict[str, str], request_type: str) -> None:
        """Handle successful retry of a failed request."""
        del self.failed_requests[symbol]
        if token_info not in self.completed_tokens:
            self.completed_tokens.append(token_info)
        self.context.logger.info(f"Successfully retried {request_type} request for {symbol}")

    def _handle_retry_failure(self, symbol: str, request_info: dict, error: Exception) -> None:
        """Handle failed retry attempt."""
        retry_count = request_info.get("retry_count", 0)

        if isinstance(error, RateLimitError):
            self.rate_limiting_occurred = True
            self.context.logger.info(f"Still rate limited for {symbol}, will retry again later")
            request_info["retry_count"] = retry_count + 1
            if error.wait_until is not None:
                request_info["retry_after"] = error.wait_until
        else:
            self.context.logger.warning(f"Failed to retry request for {symbol}: {error}")
            request_info["retry_count"] = retry_count + 1

    def _cleanup_max_retries_reached(self, symbol: str, request_info: dict) -> None:
        """Clean up requests that have reached maximum retry attempts."""
        if request_info.get("retry_count", 0) >= MAX_RETRIES:
            self.context.logger.error(f"Max retries reached for {symbol}, giving up")
            del self.failed_requests[symbol]
            token_info = request_info["token_info"]
            if token_info not in self.failed_tokens:
                self.failed_tokens.append(token_info)

    def _retry_failed_requests(self) -> Generator:
        """Retry failed requests."""
        current_time = time.time()
        retry_attempted = 0

        for symbol, request_info in list(self.failed_requests.items()):
            if not self._should_retry_request(request_info, current_time):
                continue

            token_info = request_info["token_info"]
            request_type = request_info.get("request_type", "unknown")
            retry_count = request_info.get("retry_count", 0)

            try:
                self._prepare_request_for_retry(token_info, request_type)

                self.context.logger.info(f"Retrying {request_type} request for {symbol} (attempt {retry_count + 1})")

                self._collect_token_data(token_info)
                self._handle_retry_success(symbol, token_info, request_type)
                retry_attempted += 1

            except (RateLimitError, ValueError, TypeError, OSError, ConnectionError) as e:
                self._handle_retry_failure(symbol, request_info, e)

            # Clean up requests that have reached max retries
            self._cleanup_max_retries_reached(symbol, request_info)
            yield

        if retry_attempted > 0:
            self.context.logger.info(f"Attempted to retry {retry_attempted} failed requests")

    def _log_collection_summary(self) -> None:
        """Log enhanced collection summary."""
        completed = len(self.completed_tokens)
        failed = len(self.failed_tokens)
        total = completed + failed
        collection_time = self.collected_data.get("collection_time", 0)
        api_calls = self.collected_data.get("api_calls", 0)

        # Calculate cache hits more accurately - tokens that didn't need updates
        total_tokens = len(ALLOWED_ASSETS["base"])
        ohlcv_cache_hits = total_tokens - len(
            [t for t in self.pending_tokens if t["symbol"] in self.tokens_needing_ohlcv_update]
        )
        social_cache_hits = total_tokens - len(
            [t for t in self.pending_tokens if t["symbol"] in self.tokens_needing_social_update]
        )
        cache_hits = ohlcv_cache_hits + social_cache_hits

        total_possible_updates = total_tokens * 2  # OHLCV + Social per token
        efficiency = (cache_hits / total_possible_updates * 100) if total_possible_updates > 0 else 0

        self.context.logger.info("=" * 60)
        self.context.logger.info("COLLECTION SUMMARY")
        self.context.logger.info("=" * 60)
        self.context.logger.info(f"Tokens: {completed}/{total} successful, {failed} failed")
        self.context.logger.info(f"Time: {collection_time:.1f}s")
        self.context.logger.info(f"API calls: {api_calls}")
        self.context.logger.info(f"Cache hits: {cache_hits}/{total_possible_updates}")
        self.context.logger.info(f"Efficiency: {efficiency:.1f}%")
        self.context.logger.info(
            f"OHLCV updates: {len([t for t in self.pending_tokens if t['symbol'] in self.tokens_needing_ohlcv_update])}"
        )
        self.context.logger.info(
            f"Social updates: "
            f"{len([t for t in self.pending_tokens if t['symbol'] in self.tokens_needing_social_update])}"
        )
        self.context.logger.info("=" * 60)

    def _store_collected_data(self) -> None:
        """Store collected data to persistent storage."""
        if not self.context.store_path:
            self.context.logger.warning("No store path available, skipping data storage")
            return

        try:
            self.context.store_path.mkdir(parents=True, exist_ok=True)

            data_file = self.context.store_path / "collected_data.json"
            with open(data_file, "w", encoding=DEFAULT_ENCODING) as f:
                json.dump(self.collected_data, f, indent=2)
            self.context.logger.info(f"Collected data stored to {data_file}")

            file_size = data_file.stat().st_size / 1024
            self.context.logger.info(f"  File size: {file_size:.1f} KB")

        except Exception as e:
            self.context.logger.exception(f"Failed to store collected data: {e}")

    def _has_technical_data(self, symbol: str) -> bool:
        """Check if a token has technical data (OHLCV and current price)."""
        return symbol in self.collected_data["ohlcv"] and symbol in self.collected_data["current_prices"]

    def _has_social_data(self, symbol: str) -> bool:
        """Check if a token has social data."""
        return symbol in self.collected_data["social_data"]

    def _count_data_availability(self, all_tokens: list[dict[str, str]]) -> tuple[int, int, int]:
        """Count tokens with technical, social, and both types of data."""
        tokens_with_technical = 0
        tokens_with_social = 0
        tokens_with_both_data = 0

        for token in all_tokens:
            symbol = token["symbol"]
            has_technical = self._has_technical_data(symbol)
            has_social = self._has_social_data(symbol)

            if has_technical:
                tokens_with_technical += 1
            if has_social:
                tokens_with_social += 1
            if has_technical and has_social:
                tokens_with_both_data += 1

        return tokens_with_technical, tokens_with_social, tokens_with_both_data

    def _determine_sufficiency_threshold(
        self, tokens_with_technical: int, tokens_with_both: int, min_required: int
    ) -> tuple[bool, str]:
        """Determine if data is sufficient based on rate limiting context."""
        if self.rate_limiting_occurred:
            # If rate limited during this round, accept if we have technical data for sufficient tokens
            is_sufficient = tokens_with_technical >= min_required
            reason = "rate limited during collection, using technical data threshold"
        else:
            # Normal case - prefer both data types
            is_sufficient = tokens_with_both >= min_required
            reason = "normal collection, using both data types threshold"

        return is_sufficient, reason

    def _is_data_sufficient(self) -> bool:
        """Check if collected data is sufficient for analysis."""
        # Consider all tokens, not just completed ones
        all_tokens = self.completed_tokens + self.failed_tokens
        total_tokens = len(ALLOWED_ASSETS["base"])
        min_required = max(1, int(total_tokens * self.context.params.data_sufficiency_threshold))

        # Count tokens with different types of data availability
        tokens_with_technical, tokens_with_social, tokens_with_both = self._count_data_availability(all_tokens)

        # Determine sufficiency based on rate limiting context
        is_sufficient, reason = self._determine_sufficiency_threshold(
            tokens_with_technical, tokens_with_both, min_required
        )

        self.context.logger.info(
            f"Data sufficiency ({reason}): {tokens_with_technical} technical, {tokens_with_social} social, "
            f"{tokens_with_both} both, {min_required} required. Sufficient: {is_sufficient}"
        )

        return is_sufficient

    def _finalize_collection(self) -> None:
        """Finalize data collection and store results."""
        if self.started_at:
            collection_time = (datetime.now(UTC) - self.started_at).total_seconds()
            self.collected_data["collection_time"] = collection_time

        self._store_collected_data()
        self._log_collection_summary()
