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

"""This module contains the classes required for dialogue management.

- DefaultDialogue: The dialogue class maintains state of a dialogue of type default and manages it.
- DefaultDialogues: The dialogues class keeps track of all dialogues of type default.
- HttpDialogue: The dialogue class maintains state of a dialogue of type http and manages it.
- HttpDialogues: The dialogues class keeps track of all dialogues of type http.
"""

from packages.eightballer.protocols.http.dialogues import (
    HttpDialogue as BaseHttpDialogue,
    HttpDialogues as BaseHttpDialogues,
)
from packages.eightballer.protocols.default.dialogues import (
    DefaultDialogue as BaseDefaultDialogue,
    DefaultDialogues as BaseDefaultDialogues,
)


DefaultDialogue = BaseDefaultDialogue
DefaultDialogues = BaseDefaultDialogues


HttpDialogue = BaseHttpDialogue
HttpDialogues = BaseHttpDialogues
