# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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

"""This module contains the class to connect to an Gnosis Safe contract."""

import logging
import secrets
import binascii
from enum import Enum
from typing import Any, cast

from hexbytes import HexBytes
from requests import HTTPError
from eth_utils import event_abi_to_log_topic
from aea.common import JSONLike
from eth_typing import HexStr, HexAddress, ChecksumAddress
from web3.types import Wei, Nonce, TxData, TxParams, FilterParams, BlockIdentifier
from aea.crypto.base import LedgerApi
from web3.exceptions import ContractLogicError, TransactionNotFound
from packaging.version import Version
from aea.contracts.base import Contract
from web3._utils.events import get_event_data  # noqa: PLC2701
from aea_ledger_ethereum import EthereumApi
from aea.configurations.base import PublicId

from packages.xiuxiuxar.contracts.gnosis_safe.encode import encode_typed_data
from packages.xiuxiuxar.contracts.gnosis_safe_proxy_factory.contract import (
    GnosisSafeProxyFactoryContract,
)


PUBLIC_ID = PublicId.from_str("xiuxiuxar/gnosis_safe:0.1.0")
MIN_GAS = MIN_GASPRICE = 1
# see https://github.com/safe-global/safe-eth-py/blob/6c0e0d80448e5f3496d0d94985bca239df6eb399/gnosis/safe/safe_tx.py#L354
GAS_ADJUSTMENT = 75_000
TOPIC_BYTES = 32
TOPIC_CHARS = TOPIC_BYTES * 2
Ox = "0x"
Ox_CHARS = len(Ox)

_logger = logging.getLogger(f"aea.packages.{PUBLIC_ID.author}.contracts.{PUBLIC_ID.name}.contract")

NULL_ADDRESS: str = "0x" + "0" * 40
SAFE_CONTRACT = "0xd9Db270c1B5E3Bd161E8c8503c55cEABeE709552"
DEFAULT_CALLBACK_HANDLER = "0xf48f2B2d2a534e402487b3ee7C18c33Aec0Fe5e4"
PROXY_FACTORY_CONTRACT = "0xa6B71E26C5e0845f74c812102Ca7114b6a896AB2"
SAFE_DEPLOYED_BYTECODE = "0x608060405273ffffffffffffffffffffffffffffffffffffffff600054167fa619486e0000000000000000000000000000000000000000000000000000000060003514156050578060005260206000f35b3660008037600080366000845af43d6000803e60008114156070573d6000fd5b3d6000f3fea2646970667358221220d1429297349653a4918076d650332de1a1068c5f3e07c5c82360c277770b955264736f6c63430007060033"  # noqa: E501


def _get_nonce() -> int:
    """Generate a nonce for the Safe deployment."""
    return secrets.SystemRandom().randint(0, 2**256 - 1)


def checksum_address(agent_address: str) -> ChecksumAddress:
    """Get the checksum address."""
    return ChecksumAddress(HexAddress(HexStr(agent_address)))


def pad_address_for_topic(address: str) -> HexBytes:
    """Left-pad an Ethereum address to 32 bytes for use in a topic."""
    return HexBytes(Ox + address[Ox_CHARS:].zfill(TOPIC_CHARS))


class SafeOperation(Enum):
    """Operation types."""

    CALL = 0
    DELEGATE_CALL = 1
    CREATE = 2


class GnosisSafeContract(Contract):  # noqa: PLR0904
    """The Gnosis Safe contract."""

    contract_id = PUBLIC_ID
    _SENTINEL_OWNERS = "0x0000000000000000000000000000000000000001"

    @classmethod
    def get_raw_transaction(cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any) -> JSONLike | None:
        """Get the Safe transaction."""
        raise NotImplementedError

    @classmethod
    def get_raw_message(cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any) -> bytes | None:
        """Get raw message."""
        raise NotImplementedError

    @classmethod
    def get_state(cls, ledger_api: LedgerApi, contract_address: str, **kwargs: Any) -> JSONLike | None:
        """Get state."""
        raise NotImplementedError

    @classmethod
    def get_deploy_transaction(cls, ledger_api: LedgerApi, deployer_address: str, **kwargs: Any) -> JSONLike | None:
        """Get deploy transaction."""
        owners = kwargs.pop("owners")
        threshold = kwargs.pop("threshold")
        ledger_api = cast(EthereumApi, ledger_api)
        tx_params, contract_address = cls._get_deploy_transaction(
            ledger_api, deployer_address, owners=owners, threshold=threshold, **kwargs
        )
        result = dict(cast(dict, tx_params))
        # piggyback the contract address
        result["contract_address"] = contract_address
        return result

    @classmethod
    def _get_deploy_transaction(  # pylint: disable=too-many-locals,too-many-arguments  # noqa: PLR0914
        cls,
        ledger_api: EthereumApi,
        deployer_address: str,
        owners: list[str],
        threshold: int,
        salt_nonce: int | None = None,
        gas: int = 0,
        gas_price: int | None = None,
        max_fee_per_gas: int | None = None,
        max_priority_fee_per_gas: int | None = None,
    ) -> tuple[TxParams, str]:
        """Get the deployment transaction of the new Safe."""
        salt_nonce = salt_nonce if salt_nonce is not None else _get_nonce()
        salt_nonce = cast(int, salt_nonce)
        to_address = NULL_ADDRESS
        data = b""
        payment_token = NULL_ADDRESS
        payment = 0
        payment_receiver = NULL_ADDRESS

        if len(owners) < threshold:
            msg = "Threshold cannot be bigger than the number of unique owners"
            raise ValueError(msg)

        safe_contract_address = SAFE_CONTRACT
        proxy_factory_address = PROXY_FACTORY_CONTRACT
        fallback_handler = DEFAULT_CALLBACK_HANDLER

        account_address = checksum_address(deployer_address)
        account_balance: int = ledger_api.api.eth.get_balance(account_address)
        if not account_balance:
            msg = "Client does not have any funds"
            raise ValueError(msg)

        ether_account_balance = round(ledger_api.api.from_wei(account_balance, "ether"), 6)
        _logger.info(
            "Network %s - Sender %s - Balance: %s ETH",
            ledger_api.api.net.version,
            account_address,
            ether_account_balance,
        )

        if not ledger_api.api.eth.get_code(safe_contract_address) or not ledger_api.api.eth.get_code(
            proxy_factory_address
        ):
            msg = "Network not supported"
            raise ValueError(msg)  # pragma: nocover

        _logger.info(
            "Creating new Safe with owners=%s threshold=%s " "fallback-handler=%s salt-nonce=%s",
            owners,
            threshold,
            fallback_handler,
            salt_nonce,
        )
        safe_contract = cls.get_instance(ledger_api, safe_contract_address)
        safe_creation_tx_data = HexBytes(
            safe_contract.functions.setup(
                owners,
                threshold,
                to_address,
                data,
                fallback_handler,
                payment_token,
                payment,
                payment_receiver,
            ).build_transaction({"gas": MIN_GAS, "gasPrice": MIN_GASPRICE})["data"]
        )

        nonce = ledger_api._try_get_transaction_count(  # pylint: disable=protected-access  # noqa: SLF001
            account_address
        )
        if nonce is None:
            msg = "No nonce returned."
            raise ValueError(msg)  # pragma: nocover

        (
            tx_params,
            contract_address,
        ) = GnosisSafeProxyFactoryContract.build_tx_deploy_proxy_contract_with_nonce(
            ledger_api,
            proxy_factory_address,
            safe_contract_address,
            account_address,
            safe_creation_tx_data,
            salt_nonce,
            nonce=nonce,
            gas=gas,
            gas_price=gas_price,
            max_fee_per_gas=max_fee_per_gas,
            max_priority_fee_per_gas=max_priority_fee_per_gas,
        )
        return tx_params, contract_address

    @classmethod
    def get_raw_safe_transaction_hash(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        to_address: str,
        value: int,
        data: bytes,
        operation: int = SafeOperation.CALL.value,
        safe_tx_gas: int = 0,
        base_gas: int = 0,
        gas_price: int = 0,
        gas_token: str = NULL_ADDRESS,
        refund_receiver: str = NULL_ADDRESS,
        safe_nonce: int | None = None,
        safe_version: str | None = None,
        chain_id: int | None = None,
    ) -> JSONLike:
        """Get the hash of the raw Safe transaction."""
        safe_contract = cls.get_instance(ledger_api, contract_address)
        if safe_nonce is None:
            safe_nonce = safe_contract.functions.nonce().call(block_identifier="latest")
        if safe_version is None:
            safe_version = safe_contract.functions.VERSION().call(block_identifier="latest")
        if chain_id is None:
            chain_id = ledger_api.api.eth.chain_id

        data_ = HexBytes(data).hex()

        # Safes >= 1.0.0 Renamed `baseGas` to `dataGas`
        safe_version_ = Version(safe_version)
        base_gas_name = "baseGas" if safe_version_ >= Version("1.0.0") else "dataGas"

        structured_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "verifyingContract", "type": "address"},
                ],
                "SafeTx": [
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "data", "type": "bytes"},
                    {"name": "operation", "type": "uint8"},
                    {"name": "safeTxGas", "type": "uint256"},
                    {"name": base_gas_name, "type": "uint256"},
                    {"name": "gasPrice", "type": "uint256"},
                    {"name": "gasToken", "type": "address"},
                    {"name": "refundReceiver", "type": "address"},
                    {"name": "nonce", "type": "uint256"},
                ],
            },
            "primaryType": "SafeTx",
            "domain": {
                "verifyingContract": contract_address,
            },
            "message": {
                "to": to_address,
                "value": value,
                "data": data_,
                "operation": operation,
                "safeTxGas": safe_tx_gas,
                base_gas_name: base_gas,
                "gasPrice": gas_price,
                "gasToken": gas_token,
                "refundReceiver": refund_receiver,
                "nonce": safe_nonce,
            },
        }

        # Safes >= 1.3.0 Added `chainId` to the domain
        if safe_version_ >= Version("1.3.0"):
            # EIP712Domain(uint256 chainId,address verifyingContract)
            structured_data["types"]["EIP712Domain"].insert(  # type: ignore
                0, {"name": "chainId", "type": "uint256"}
            )
            structured_data["domain"]["chainId"] = chain_id  # type: ignore

        return {"tx_hash": HexBytes(encode_typed_data(structured_data)).hex()}

    @classmethod
    def get_packed_signatures(cls, owners: tuple[str], signatures_by_owner: dict[str, str]) -> bytes:
        """Get the packed signatures."""
        sorted_owners = sorted(owners, key=str.lower)
        signatures = b""
        for signer in sorted_owners:
            if signer not in signatures_by_owner:
                continue  # pragma: nocover
            signature = signatures_by_owner[signer]
            signature_bytes = binascii.unhexlify(signature)
            signatures += signature_bytes
        # Packed signature data ({bytes32 r}{bytes32 s}{uint8 v})
        return signatures

    @classmethod
    def get_raw_safe_transaction(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        owners: tuple[str],
        to_address: str,
        value: int,
        data: bytes,
        signatures_by_owner: dict[str, str],
        operation: int = SafeOperation.CALL.value,
        safe_tx_gas: int = 0,
        base_gas: int = 0,
        safe_gas_price: int = 0,
        gas_token: str = NULL_ADDRESS,
        refund_receiver: str = NULL_ADDRESS,
        gas_price: int | None = None,
        nonce: Nonce | None = None,
        max_fee_per_gas: int | None = None,
        max_priority_fee_per_gas: int | None = None,
        old_price: dict[str, Wei] | None = None,
        fallback_gas: int = 0,
    ) -> JSONLike:
        """Get the raw Safe transaction."""
        sender_address = ledger_api.api.to_checksum_address(sender_address)
        to_address = ledger_api.api.to_checksum_address(to_address)
        ledger_api = cast(EthereumApi, ledger_api)
        signatures = cls.get_packed_signatures(owners, signatures_by_owner)
        safe_contract = cls.get_instance(ledger_api, contract_address)

        w3_tx = safe_contract.functions.execTransaction(
            to_address,
            value,
            data,
            operation,
            safe_tx_gas,
            base_gas,
            safe_gas_price,
            gas_token,
            refund_receiver,
            signatures,
        )
        # see https://github.com/safe-global/safe-eth-py/blob/6c0e0d80448e5f3496d0d94985bca239df6eb399/gnosis/safe/safe_tx.py#L354
        configured_gas = base_gas + safe_tx_gas + GAS_ADJUSTMENT if base_gas != 0 or safe_tx_gas != 0 else MIN_GAS
        tx_parameters: dict[str, str | int] = {
            "from": sender_address,
            "gas": configured_gas,
        }
        actual_nonce = ledger_api.api.eth.get_transaction_count(ledger_api.api.to_checksum_address(sender_address))
        if actual_nonce != nonce:
            nonce = actual_nonce
            old_price = None
        if gas_price is not None:
            tx_parameters["gasPrice"] = gas_price
        if max_fee_per_gas is not None:
            tx_parameters["maxFeePerGas"] = max_fee_per_gas  # pragma: nocover
        if max_priority_fee_per_gas is not None:  # pragma: nocover
            tx_parameters["maxPriorityFeePerGas"] = max_priority_fee_per_gas
        if gas_price is None and max_fee_per_gas is None and max_priority_fee_per_gas is None:
            gas_pricing = ledger_api.try_get_gas_pricing(old_price=old_price)
            if gas_pricing is None:
                _logger.warning(f"Could not get gas price with {old_price=}")
            else:
                tx_parameters.update(gas_pricing)

        # note, the next line makes an eth_estimateGas call if gas is not set!
        transaction_dict = w3_tx.build_transaction(tx_parameters)
        if configured_gas != MIN_GAS:
            transaction_dict["gas"] = Wei(configured_gas)
        else:
            gas_estimate = ledger_api._try_get_gas_estimate(  # pylint: disable=protected-access  # noqa: SLF001
                transaction_dict
            )
            transaction_dict["gas"] = Wei(gas_estimate) if gas_estimate is not None else fallback_gas
        transaction_dict["nonce"] = nonce  # pragma: nocover
        return transaction_dict

    @classmethod
    def verify_contract(cls, ledger_api: LedgerApi, contract_address: str) -> JSONLike:
        """Verify the contract's bytecode."""
        ledger_api = cast(EthereumApi, ledger_api)
        deployed_bytecode = ledger_api.api.eth.get_code(contract_address).hex()
        # we cannot use cls.contract_interface["ethereum"]["deployedBytecode"] because the
        # contract is created via a proxy
        local_bytecode = SAFE_DEPLOYED_BYTECODE
        verified = deployed_bytecode == local_bytecode
        return {"verified": verified}

    @classmethod
    def verify_tx(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        tx_hash: str,
        owners: tuple[str],
        to_address: str,
        value: int,
        data: bytes,
        signatures_by_owner: dict[str, str],
        operation: int = SafeOperation.CALL.value,
        safe_tx_gas: int = 0,
        base_gas: int = 0,
        gas_price: int = 0,
        gas_token: str = NULL_ADDRESS,
        refund_receiver: str = NULL_ADDRESS,
        safe_version: str | None = None,
    ) -> JSONLike:
        """Verify a tx hash exists on the blockchain."""
        to_address = ledger_api.api.to_checksum_address(to_address)
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        signatures = cls.get_packed_signatures(owners, signatures_by_owner)

        if safe_version is None:
            safe_version = safe_contract.functions.VERSION().call(block_identifier="latest")
        # Safes >= 1.0.0 Renamed `baseGas` to `dataGas`
        safe_version_ = Version(safe_version)
        base_gas_name = "baseGas" if safe_version_ >= Version("1.0.0") else "dataGas"

        try:
            _logger.info(f"Trying to get transaction and receipt from hash {tx_hash}")
            transaction = ledger_api.api.eth.get_transaction(tx_hash)
            receipt = ledger_api.get_transaction_receipt(tx_hash)
            _logger.info(f"Received transaction: {transaction}, with receipt: {receipt}.")
            if receipt is None:
                raise ValueError  # pragma: nocover
        except (TransactionNotFound, ValueError):  # pragma: nocover
            return {"verified": False, "status": -1}

        expected = {
            "contract_address": contract_address,
            "to_address": to_address,
            "value": value,
            "data": data,
            "operation": operation,
            "safe_tx_gas": safe_tx_gas,
            "base_gas": base_gas,
            "gas_price": gas_price,
            "gas_token": gas_token,
            "refund_receiver": refund_receiver,
            "signatures": signatures,
        }
        decoded: tuple[Any, dict] = (None, {})
        diff: dict = {}
        try:
            decoded = safe_contract.decode_function_input(transaction["input"])
            actual = {
                "contract_address": transaction["to"],
                "to_address": decoded[1]["to"],
                "value": decoded[1]["value"],
                "data": decoded[1]["data"],
                "operation": decoded[1]["operation"],
                "safe_tx_gas": decoded[1]["safeTxGas"],
                "base_gas": decoded[1][base_gas_name],
                "gas_price": decoded[1]["gasPrice"],
                "gas_token": decoded[1]["gasToken"],
                "refund_receiver": decoded[1]["refundReceiver"],
                "signatures": decoded[1]["signatures"],
            }
            diff = {k: (v, actual[k]) for k, v in expected.items() if v != actual[k]}
            verified = receipt["status"] and "execTransaction" in str(decoded[0]) and len(diff) == 0
        except (TransactionNotFound, KeyError, ValueError):  # pragma: nocover
            verified = False
        return {
            "verified": verified,
            "status": receipt["status"],
            "transaction": transaction,
            "actual": decoded,  # type: ignore
            "expected": expected,
            "diff": diff,
        }

    @classmethod
    def revert_reason(  # pylint: disable=unused-argument
        cls,
        ledger_api: EthereumApi,
        contract_address: str,  # noqa: ARG003
        tx: TxData,
    ) -> JSONLike:
        """Check the revert reason of a transaction."""
        ledger_api = cast(EthereumApi, ledger_api)

        # build a new transaction to replay:
        replay_tx = {
            "to": tx["to"],
            "from": tx["from"],
            "value": tx["value"],
            "data": tx["input"],
        }

        try:
            # replay the transaction locally:
            ledger_api.api.eth.call(replay_tx, tx["blockNumber"] - 1)
        except ContractLogicError as e:
            # execution reverted exception
            return {"revert_reason": repr(e)}
        except HTTPError:  # pragma: nocover
            # http exception
            raise
        else:
            # given tx not reverted
            msg = f"The given transaction has not been reverted!\ntx: {tx}"
            raise ValueError(msg)

    @classmethod
    def get_safe_nonce(cls, ledger_api: EthereumApi, contract_address: str) -> JSONLike:
        """Retrieve the safe's nonce."""
        safe_contract = cls.get_instance(ledger_api, contract_address)
        safe_nonce = safe_contract.functions.nonce().call(block_identifier="latest")
        return {"safe_nonce": safe_nonce}

    @classmethod
    def get_ingoing_transfers(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        from_block: BlockIdentifier | None = None,
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """A list of transfers into the contract."""
        safe_contract = cls.get_instance(ledger_api, contract_address)

        if from_block is None:
            logging.info(
                "'from_block' not provided, checking for transfers to the safe contract in the last 50 blocks."
            )
            current_block = ledger_api.api.eth.get_block("latest")["number"]
            from_block = max(0, current_block - 50)  # check in the last 50 blocks

        event_abi = safe_contract.events.SafeReceived().abi
        event_topic = event_abi_to_log_topic(event_abi)

        filter_params: FilterParams = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": safe_contract.address,
            "topics": [event_topic],
        }

        w3 = ledger_api.api.eth
        logs = w3.get_logs(filter_params)
        entries = [get_event_data(w3.codec, event_abi, log) for log in logs]

        return {
            "data": [
                {
                    "sender": entry["args"]["sender"],
                    "amount": int(entry["args"]["value"]),
                    "blockNumber": entry["blockNumber"],
                }
                for entry in entries
            ]
        }

    @classmethod
    def get_balance(cls, ledger_api: EthereumApi, contract_address: str) -> JSONLike:
        """Retrieve the safe's balance."""
        return {"balance": ledger_api.get_balance(address=contract_address)}

    @classmethod
    def get_amount_spent(  # pylint: disable=unused-argument
        cls,
        ledger_api: EthereumApi,
        contract_address: str,  # noqa: ARG003
        tx_hash: str,
    ) -> JSONLike:
        """Get the amount of ether spent in a tx."""
        tx_receipt = ledger_api.get_transaction_receipt(tx_hash)
        tx = ledger_api.get_transaction(tx_hash)

        tx_value = int(tx["value"])
        gas_price = int(tx["gasPrice"])
        gas_used = int(tx_receipt["gasUsed"])
        total_spent = tx_value + (gas_price * gas_used)

        return {"amount_spent": total_spent}

    @classmethod
    def get_safe_txs(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get all the safe tx hashes."""

        ledger_api = cast(EthereumApi, ledger_api)
        factory_contract = cls.get_instance(ledger_api, contract_address)
        event_abi = factory_contract.events.ExecutionSuccess().abi
        event_topic = event_abi_to_log_topic(event_abi)

        filter_params: FilterParams = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": factory_contract.address,
            "topics": [event_topic],
        }

        w3 = ledger_api.api.eth
        logs = w3.get_logs(filter_params)
        entries = [get_event_data(w3.codec, event_abi, log) for log in logs]

        return {
            "txs": [
                {
                    "tx_hash": entry["transactionHash"].hex(),
                    "block_number": entry["blockNumber"],
                }
                for entry in entries
            ]
        }

    @classmethod
    def get_removed_owner_events(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        removed_owner: str | None = None,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get all RemovedOwner events for a safe contract."""
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        event_abi = safe_contract.events.RemovedOwner().abi
        event_topic = event_abi_to_log_topic(event_abi)

        filter_params: FilterParams = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": safe_contract.address,
            "topics": [event_topic],
        }

        w3 = ledger_api.api.eth
        logs = w3.get_logs(filter_params)
        entries = [get_event_data(w3.codec, event_abi, log) for log in logs]

        checksummed_removed_owner = (
            ledger_api.api.to_checksum_address(removed_owner) if removed_owner is not None else None
        )

        removed_owner_events = [
            {
                "tx_hash": entry["transactionHash"].hex(),
                "block_number": entry["blockNumber"],
                "owner": entry["args"]["owner"],
            }
            for entry in entries
            if checksummed_removed_owner is None
            or ledger_api.api.to_checksum_address(entry["args"]["owner"]) == checksummed_removed_owner
        ]

        return {"data": removed_owner_events}

    @classmethod
    def get_zero_transfer_events(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        sender_address: str,
        from_block: BlockIdentifier = "earliest",
        to_block: BlockIdentifier = "latest",
    ) -> JSONLike:
        """Get all zero transfer events from a given sender to the safe address."""
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        event_abi = safe_contract.events.SafeReceived().abi
        event_topic = event_abi_to_log_topic(event_abi)
        sender_address = ledger_api.api.to_checksum_address(sender_address)
        padded_sender = pad_address_for_topic(sender_address)

        filter_params: FilterParams = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": safe_contract.address,
            # cannot filter for 0 value transfers using topics as the value is not indexed
            "topics": [event_topic, padded_sender],
        }

        w3 = ledger_api.api.eth
        logs = w3.get_logs(filter_params)
        entries = [get_event_data(w3.codec, event_abi, log) for log in logs]
        zero_transfer_events = [
            {
                "tx_hash": entry["transactionHash"].hex(),
                "block_number": entry["blockNumber"],
                "sender": ledger_api.api.to_checksum_address(entry["args"]["sender"]),
            }
            for entry in entries
            if int(entry["args"]["value"]) == 0
        ]
        return {
            "data": zero_transfer_events,
        }

    @classmethod
    def get_remove_owner_data(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        owner: str,
        threshold: int,
    ) -> JSONLike:
        """Get a removeOwner() encoded tx."""
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        # Note that owners in the safe are stored as a linked list, we need to know the parent (prev_owner) of an owner
        # when removing. https://github.com/safe-global/safe-smart-account/tree/v1.3.0/contracts/base/OwnerManager.sol#L15
        owners = [ledger_api.api.to_checksum_address(owner) for owner in safe_contract.functions.getOwners().call()]
        owner = ledger_api.api.to_checksum_address(owner)
        prev_owner = cls._get_prev_owner(owners, owner)
        data = safe_contract.encode_abi(
            abi_element_identifier="removeOwner",
            args=[
                ledger_api.api.to_checksum_address(prev_owner),
                owner,
                threshold,
            ],
        )
        return {
            "data": data,
        }

    @classmethod
    def get_swap_owner_data(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        old_owner: str,
        new_owner: str,
    ) -> JSONLike:
        """Get a swapOwner() encoded tx."""
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        # Note that owners in the safe are stored as a linked list, we need to know the parent (prev_owner) of an owner
        # when swapping. https://github.com/safe-global/safe-smart-account/tree/v1.3.0/contracts/base/OwnerManager.sol#L15
        owners = [ledger_api.api.to_checksum_address(owner) for owner in safe_contract.functions.getOwners().call()]
        old_owner = ledger_api.api.to_checksum_address(old_owner)
        prev_owner = cls._get_prev_owner(owners, old_owner)
        data = safe_contract.encode_abi(
            abi_element_identifier="swapOwner",
            args=[
                ledger_api.api.to_checksum_address(prev_owner),
                old_owner,
                ledger_api.api.to_checksum_address(new_owner),
            ],
        )
        return {
            "data": data,
        }

    @classmethod
    def _get_prev_owner(cls, linked_list: list[str], target: str) -> str:
        """Given a linked list of strings, it returns the one pointing to target."""
        root = cls._SENTINEL_OWNERS
        index = linked_list.index(target) - 1
        if index < 0:
            # if target is the first element in the list, the root is pointing to it
            return root
        return linked_list[index]

    @classmethod
    def get_owners(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
    ) -> JSONLike:
        """Get the safe owners."""
        ledger_api = cast(EthereumApi, ledger_api)
        safe_contract = cls.get_instance(ledger_api, contract_address)
        owners = [ledger_api.api.to_checksum_address(owner) for owner in safe_contract.functions.getOwners().call()]
        return {"owners": owners}

    @classmethod
    def get_approve_hash_tx(
        cls,
        ledger_api: EthereumApi,
        contract_address: str,
        tx_hash: str,
        sender: str,
    ) -> JSONLike:
        """Get approve has tx."""
        ledger_api = cast(EthereumApi, ledger_api)
        return ledger_api.build_transaction(
            contract_instance=cls.get_instance(ledger_api, contract_address),
            method_name="approveHash",
            method_args={
                "hashToApprove": tx_hash,
            },
            tx_args={
                "sender_address": sender,
            },
        )
