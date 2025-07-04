# ------------------------------------------------------------------------------
#
#   Copyright 2022 Valory AG
#   Copyright 2018-2021 Fetch.AI Limited
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
"""This module contains the tests of the dialogue classes of the http_echo skill."""

from typing import cast
from pathlib import Path

from aea.test_tools.test_skill import COUNTERPARTY_AGENT_ADDRESS, BaseSkillTestCase

from packages.xiuxiuxar.skills.mindshare_app import PUBLIC_ID
from packages.eightballer.protocols.http.message import HttpMessage
from packages.eightballer.protocols.default.message import DefaultMessage
from packages.xiuxiuxar.skills.mindshare_app.dialogues import (
    HttpDialogues,
    DefaultDialogue,
    DefaultDialogues,
)


ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent


class TestDialogues(BaseSkillTestCase):
    """Test dialogue class of http_echo."""

    path_to_skill = Path(ROOT_DIR, "packages", PUBLIC_ID.author, "skills", PUBLIC_ID.name)

    @classmethod
    def setup(cls):  # pylint: disable=W0221
        """Setup the test class."""
        super().setup_class()
        cls.default_dialogues = cast(DefaultDialogues, cls._skill.skill_context.default_dialogues)
        cls.http_dialogues = cast(HttpDialogues, cls._skill.skill_context.http_dialogues)

    def test_default_dialogues(self):
        """Test the DefaultDialogues class."""
        _, dialogue = self.default_dialogues.create(
            counterparty=COUNTERPARTY_AGENT_ADDRESS,
            performative=DefaultMessage.Performative.BYTES,
            content=b"some_content",
        )
        assert dialogue.role == DefaultDialogue.Role.AGENT
        assert dialogue.self_address == str(self.skill.skill_context.skill_id)

    def test_http_dialogues(self):
        """Test the HttpDialogues class."""
        _, dialogue = self.http_dialogues.create(
            counterparty=COUNTERPARTY_AGENT_ADDRESS,
            performative=HttpMessage.Performative.REQUEST,
            method="some_method",
            url="some_url",
            version="some_version",
            headers="some_headers",
            body=b"some_body",
        )
        assert dialogue.self_address == str(self.skill.skill_context.skill_id)
