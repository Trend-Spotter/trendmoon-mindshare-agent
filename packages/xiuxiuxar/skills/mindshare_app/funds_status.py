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

"""Funds Status module for Pearl v1 compliance."""

from typing import Dict, Tuple, Any, cast
from web3 import Web3

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# ERC20 Token ABI (minimal - just balanceOf and decimals)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


class FundsStatusService:
    """Service for computing agent funds status (Pearl v1)."""

    def __init__(self, context: Any) -> None:
        """Initialize the funds status service."""
        self.context = context
        self._web3_clients: Dict[str, Web3] = {}

    def get_web3(self, chain: str, rpc_url: str) -> Web3:
        """Get or create a Web3 client for the specified chain."""
        if chain not in self._web3_clients:
            self._web3_clients[chain] = Web3(Web3.HTTPProvider(rpc_url))
        return self._web3_clients[chain]

    def fetch_balance(
        self, chain: str, address: str, asset_address: str, rpc_url: str
    ) -> Tuple[int, int]:
        """
        Fetch the balance and decimals for an asset.

        Args:
            chain: Chain name (lowercase)
            address: Wallet address to check
            asset_address: Token contract address (or zero address for native)
            rpc_url: RPC endpoint URL

        Returns:
            Tuple of (balance in wei, decimals)
        """
        w3 = self.get_web3(chain, rpc_url)

        # Native token (ETH, etc.)
        if asset_address == ZERO_ADDRESS:
            balance = w3.eth.get_balance(Web3.to_checksum_address(address))
            return balance, 18

        # ERC20 token
        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(asset_address), abi=ERC20_ABI
            )
            balance = contract.functions.balanceOf(
                Web3.to_checksum_address(address)
            ).call()

            try:
                decimals = contract.functions.decimals().call()
            except Exception:
                decimals = 18  # Default if decimals() call fails

            return balance, decimals

        except Exception as e:
            self.context.logger.warning(
                f"Failed to fetch balance for {asset_address} on {chain}: {e}"
            )
            return 0, 18

    def compute_funds_status(self, fund_requirements: dict, rpc_urls: dict) -> dict:
        """
        Compute the current funds status for all chains, addresses, and assets.

        This implements the "fixed threshold and topup" strategy:
        - If balance < threshold: request (topup - balance)
        - Otherwise: no request needed (deficit = 0)

        Args:
            fund_requirements: Dictionary with fund requirements per chain/address/asset
            rpc_urls: Dictionary with RPC URLs per chain

        Returns:
            Dictionary following Pearl v1 /funds-status schema
        """
        payload: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}

        for chain, addresses in fund_requirements.items():
            chain_result: Dict[str, Dict[str, Dict[str, str]]] = {}
            rpc_url = rpc_urls.get(chain)

            if not rpc_url:
                self.context.logger.warning(
                    f"No RPC URL configured for chain '{chain}', skipping"
                )
                continue

            for address, assets in addresses.items():
                checksum_address = Web3.to_checksum_address(address)
                asset_result: Dict[str, Dict[str, str]] = {}

                for asset_address, config in assets.items():
                    # Checksum the asset address
                    checksum_asset = (
                        ZERO_ADDRESS
                        if asset_address == ZERO_ADDRESS
                        else Web3.to_checksum_address(asset_address)
                    )

                    # Fetch current balance
                    balance, decimals = self.fetch_balance(
                        chain, checksum_address, checksum_asset, rpc_url
                    )

                    # Calculate deficit using threshold/topup strategy
                    threshold = int(config["threshold"])
                    topup = int(config["topup"])

                    if balance < threshold:
                        # Request funds to reach topup level
                        deficit = max(topup - balance, 0)
                    else:
                        # Balance is sufficient
                        deficit = 0

                    # Add to result
                    asset_result[checksum_asset] = {
                        "balance": str(balance),
                        "deficit": str(deficit),
                        "decimals": str(decimals),
                    }

                chain_result[checksum_address] = asset_result

            # Include all chains in response (even if no deficit)
            if chain_result:
                payload[chain] = chain_result

        return payload
