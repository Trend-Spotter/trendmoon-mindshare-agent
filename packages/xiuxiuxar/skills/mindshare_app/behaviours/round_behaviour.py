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

import os
from typing import Any
from datetime import datetime

from pytz import UTC
from aea.protocols.base import Message
from aea.skills.behaviours import FSMBehaviour

from packages.eightballer.protocols.http.message import HttpMessage
from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import MindshareabciappEvents, MindshareabciappStates
from packages.xiuxiuxar.skills.mindshare_app.behaviours.error import HandleErrorRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.pause import PausedRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.setup import SetupRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.analysis import AnalysisRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.execution import ExecutionRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.call_checkpoint import CallCheckpointRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.data_collection import DataCollectionRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.risk_evaluation import RiskEvaluationRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.position_monitor import PositionMonitoringRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.signal_aggregation import SignalAggregationRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.trade_construction import TradeConstructionRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.portfolio_validation import PortfolioValidationRound
from packages.xiuxiuxar.skills.mindshare_app.behaviours.check_staking_kpi_met import CheckStakingKPIRound


class MindshareabciappFsmBehaviour(FSMBehaviour):
    """This class implements a simple Finite State Machine behaviour."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: PLR0915
        super().__init__(**kwargs)
        self._last_transition_timestamp: datetime | None = None
        self._previous_rounds: list[str] = []
        self._period_count: int = 0

        self.register_state(MindshareabciappStates.SETUPROUND.value, SetupRound(**kwargs), True)
        self.register_state(MindshareabciappStates.HANDLEERRORROUND.value, HandleErrorRound(**kwargs))
        self.register_state(MindshareabciappStates.DATACOLLECTIONROUND.value, DataCollectionRound(**kwargs))
        self.register_state(MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value, PortfolioValidationRound(**kwargs))
        self.register_state(MindshareabciappStates.RISKEVALUATIONROUND.value, RiskEvaluationRound(**kwargs))
        self.register_state(MindshareabciappStates.PAUSEDROUND.value, PausedRound(**kwargs))
        self.register_state(MindshareabciappStates.CHECKSTAKINGKPIROUND.value, CheckStakingKPIRound(**kwargs))
        self.register_state(MindshareabciappStates.CALLCHECKPOINTROUND.value, CallCheckpointRound(**kwargs))
        self.register_state(MindshareabciappStates.SIGNALAGGREGATIONROUND.value, SignalAggregationRound(**kwargs))
        self.register_state(MindshareabciappStates.POSITIONMONITORINGROUND.value, PositionMonitoringRound(**kwargs))
        self.register_state(MindshareabciappStates.ANALYSISROUND.value, AnalysisRound(**kwargs))
        self.register_state(MindshareabciappStates.TRADECONSTRUCTIONROUND.value, TradeConstructionRound(**kwargs))
        self.register_state(MindshareabciappStates.EXECUTIONROUND.value, ExecutionRound(**kwargs))

        self.register_transition(
            source=MindshareabciappStates.ANALYSISROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.ANALYSISROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.DATACOLLECTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CALLCHECKPOINTROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CALLCHECKPOINTROUND.value,
            event=MindshareabciappEvents.NEXT_CHECKPOINT_NOT_REACHED_YET,
            destination=MindshareabciappStates.CHECKSTAKINGKPIROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CALLCHECKPOINTROUND.value,
            event=MindshareabciappEvents.SERVICE_EVICTED,
            destination=MindshareabciappStates.DATACOLLECTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CALLCHECKPOINTROUND.value,
            event=MindshareabciappEvents.SERVICE_NOT_STAKED,
            destination=MindshareabciappStates.DATACOLLECTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.CALLCHECKPOINTROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.DATACOLLECTIONROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.POSITIONMONITORINGROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.DATACOLLECTIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.EXECUTIONROUND.value,
            event=MindshareabciappEvents.EXECUTED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.EXECUTIONROUND.value,
            event=MindshareabciappEvents.ORDER_PLACED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.EXECUTIONROUND.value,
            event=MindshareabciappEvents.FAILED,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RESET,
            destination=MindshareabciappStates.SETUPROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RETRIES_EXCEEDED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.HANDLEERRORROUND.value,
            event=MindshareabciappEvents.RETRY,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PAUSEDROUND.value,
            event=MindshareabciappEvents.RESET,
            destination=MindshareabciappStates.SETUPROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PAUSEDROUND.value,
            event=MindshareabciappEvents.RESUME,
            destination=MindshareabciappStates.CALLCHECKPOINTROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.AT_LIMIT,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.CAN_TRADE,
            destination=MindshareabciappStates.ANALYSISROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.EXIT_SIGNAL,
            destination=MindshareabciappStates.EXECUTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.POSITIONMONITORINGROUND.value,
            event=MindshareabciappEvents.POSITIONS_CHECKED,
            destination=MindshareabciappStates.PORTFOLIOVALIDATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.APPROVED,
            destination=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.RISKEVALUATIONROUND.value,
            event=MindshareabciappEvents.REJECTED,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SETUPROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.CALLCHECKPOINTROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SETUPROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.NO_SIGNAL,
            destination=MindshareabciappStates.PAUSEDROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.SIGNALAGGREGATIONROUND.value,
            event=MindshareabciappEvents.SIGNAL_GENERATED,
            destination=MindshareabciappStates.RISKEVALUATIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
            event=MindshareabciappEvents.DONE,
            destination=MindshareabciappStates.EXECUTIONROUND.value,
        )
        self.register_transition(
            source=MindshareabciappStates.TRADECONSTRUCTIONROUND.value,
            event=MindshareabciappEvents.ERROR,
            destination=MindshareabciappStates.HANDLEERRORROUND.value,
        )

    def act(self) -> None:
        """Override act to track transitions."""
        if self.current is None:
            super().act()
            return

        # Store the current state before potential transition
        previous_state = self.current

        # Call parent act which handles the FSM logic
        super().act()

        # Check if a transition occurred
        if self.current != previous_state and self.current is not None:
            # A transition occurred - track it
            current_time = datetime.now(UTC)
            self._last_transition_timestamp = current_time

            # Track round history (keep last 25 rounds)
            if previous_state and previous_state not in self._previous_rounds[-1:]:  # Avoid duplicates
                self._previous_rounds.append(previous_state)
                if len(self._previous_rounds) > 25:
                    self._previous_rounds = self._previous_rounds[-25:]

            # Check if we transitioned back to data collection round (indicates a new operational period/cycle)
            if (
                self.current == MindshareabciappStates.DATACOLLECTIONROUND.value
                and previous_state != MindshareabciappStates.DATACOLLECTIONROUND.value
            ):
                self._period_count += 1
                self.context.logger.info(f"FSM started new operational period: {self._period_count}")

            self.context.logger.info(f"FSM transitioned from {previous_state} to {self.current}")

            current_state = self.get_state(self.current)
            if current_state is not None:
                current_state.setup()

    @property
    def last_transition_timestamp(self) -> datetime | None:
        """Get the timestamp of the last transition."""
        return self._last_transition_timestamp

    @property
    def previous_rounds(self) -> list[str]:
        """Get the history of previous rounds."""
        return self._previous_rounds.copy()

    @property
    def period_count(self) -> int:
        """Get the current period count."""
        return self._period_count

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up Mindshareabciapp FSM behaviour.")
        self._last_transition_timestamp = datetime.now(UTC)

    def handle_message(self, message: Message) -> None:
        """Handle incoming messages, including HTTP responses for async data collection."""
        # Handle HTTP responses during data collection
        if isinstance(message, HttpMessage):
            current_state = self.get_state(self.current) if self.current else None
            if isinstance(current_state, DataCollectionRound) and current_state.handle_http_response(message):
                self.context.logger.debug(f"HTTP response handled by DataCollectionRound: {message.performative}")
                return  # Message was handled

            # Log unhandled HTTP messages for debugging
            self.context.logger.debug(f"Unhandled HTTP message: {message.performative} in state {self.current}")

    def teardown(self) -> None:
        """Implement the teardown."""
        self.context.logger.info("Tearing down Mindshareabciapp FSM behaviour.")

    def terminate(self) -> None:
        """Implement the termination."""
        os._exit(0)
