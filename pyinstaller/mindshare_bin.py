# ------------------------------------------------------------------------------
#
#   Copyright 2025 Xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Script for building the AEA responsible for running Mindshare."""
import os
import sys
from pathlib import Path

import aea.configurations.validation as validation_module
from aea.cli.core import cli
from aea.mail.base_pb2 import DESCRIPTOR  # noqa: F401
from multiaddr.codecs.idna import to_bytes as _
from aea_ledger_cosmos.cosmos import *  # noqa: F403
from multiaddr.codecs.uint16be import to_bytes as _
from aea.crypto.registries.base import *  # noqa: F403
from aea_ledger_ethereum.ethereum import *  # noqa: F403
from google.protobuf.descriptor_pb2 import FileDescriptorProto  # noqa: F401


# patch for the _CUR_DIR value
# we need this because pyinstaller generated binaries handle paths differently
if getattr(sys, "_MEIPASS", None):
    # Running as PyInstaller bundle
    validation_module._CUR_DIR = Path(sys._MEIPASS) / "aea" / "configurations"  # noqa: SLF001
    validation_module._SCHEMAS_DIR = str(Path(sys._MEIPASS) / "aea" / "configurations" / "schemas")  # noqa: SLF001
else:
    # Running normally
    pass


if __name__ == "__main__":
    cli(prog_name="aea")  # pragma: no cover
