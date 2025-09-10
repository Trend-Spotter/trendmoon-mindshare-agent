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

"""This package contains the Mindshare App behaviour for the pause round."""

from typing import Any
from datetime import UTC, datetime, timedelta

from packages.xiuxiuxar.skills.mindshare_app.behaviours.base import (
    BaseState,
    MindshareabciappEvents,
    MindshareabciappStates,
)


class PausedRound(BaseState):
    """This class implements the behaviour of the state PausedRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_done = False
        self.started = False
        self._state = MindshareabciappStates.PAUSEDROUND

    sleep_until: datetime | None = None

    def act(self) -> None:
        """Perform the action of the state."""

        if not self.started:
            self._is_done = False
            self.started = True
            cool_down = timedelta(seconds=self.context.params.reset_pause_duration)
            self.started_at = datetime.now(tz=UTC)
            self.sleep_until = self.started_at + cool_down
            self.context.logger.info(f"Cool down for {cool_down}s")
            return

        now = datetime.now(tz=UTC)
        if now < self.sleep_until:
            remaining = (self.sleep_until - now).total_seconds()
            self.context.logger.debug(f"Cooling down remaining: {remaining}s")
            return
        self.context.logger.info(f"Cool down finished. at {now}")
        self._is_done = True
        self._event = MindshareabciappEvents.RESUME
        self.started = False
