#!/usr/bin/env python3
"""
Pearl v1 - Private Key Decryption Utility

This script handles encrypted Ethereum private keys for Pearl v1 compatibility.
It can decrypt V3 keystores and supports backward compatibility with plaintext keys.

Usage:
    # Check if key is encrypted
    python scripts/decrypt_key.py --check

    # Decrypt and output to stdout
    python scripts/decrypt_key.py --password mypassword

    # Decrypt and overwrite file
    python scripts/decrypt_key.py --password mypassword --in-place
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from eth_account import Account


def is_encrypted_keystore(content: str) -> bool:
    """
    Check if the private key file content is an encrypted V3 keystore.

    Args:
        content: The content read from the private key file

    Returns:
        True if content is encrypted JSON keystore, False if plaintext hex
    """
    content = content.strip()

    # Try to parse as JSON
    try:
        data = json.loads(content)
        # Check if it has the required V3 keystore fields
        return isinstance(data, dict) and "crypto" in data and "version" in data
    except (json.JSONDecodeError, ValueError):
        # Not JSON, assume it's plaintext hex
        return False


def load_private_key(key_file_path: Path, password: Optional[str] = None) -> str:
    """
    Load and decrypt (if necessary) an Ethereum private key.

    This function supports both:
    - Plaintext hex private keys (for Quickstart compatibility)
    - Encrypted V3 Ethereum keystore JSON (for Pearl v1)

    Args:
        key_file_path: Path to the ethereum_private_key.txt file
        password: Optional password to decrypt encrypted keystores

    Returns:
        The private key as a hex string (with 0x prefix)

    Raises:
        FileNotFoundError: If key file doesn't exist
        ValueError: If encrypted key but no password provided, or decryption fails
    """
    if not key_file_path.exists():
        raise FileNotFoundError(f"Private key file not found: {key_file_path}")

    # Read the file content
    with open(key_file_path, "r") as f:
        content = f.read().strip()

    # Determine if it's encrypted or plaintext
    if is_encrypted_keystore(content):
        # Encrypted V3 keystore - requires password
        if password is None:
            raise ValueError(
                "Private key is encrypted but no password was provided. "
                "Use --password argument to decrypt."
            )

        try:
            # Decrypt using eth_account
            keystore_json = json.loads(content)
            private_key_bytes = Account.decrypt(keystore_json, password)

            # Convert to hex string with 0x prefix
            if hasattr(private_key_bytes, "hex"):
                private_key = private_key_bytes.hex()
                if not private_key.startswith("0x"):
                    private_key = "0x" + private_key
            elif isinstance(private_key_bytes, bytes):
                private_key = "0x" + private_key_bytes.hex()
            else:
                # Already a string
                private_key = str(private_key_bytes)
                if not private_key.startswith("0x"):
                    private_key = "0x" + private_key

            return private_key

        except Exception as e:
            raise ValueError(f"Failed to decrypt private key: {e}")
    else:
        # Plaintext hex key - backward compatible with Quickstart
        private_key = content.strip()

        # Remove any whitespace
        private_key = "".join(private_key.split())

        # Ensure it has 0x prefix
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        # Validate it's a valid hex key (64 characters after 0x)
        if len(private_key) != 66:  # 0x + 64 hex chars
            raise ValueError(
                f"Invalid private key length: expected 66 characters (0x + 64 hex), got {len(private_key)}"
            )

        return private_key


def main():
    """Main entry point."""
    parser = ArgumentParser(description="Pearl v1 Private Key Decryption Utility")
    parser.add_argument(
        "--key-file",
        type=str,
        default="ethereum_private_key.txt",
        help="Path to the private key file (default: ethereum_private_key.txt)",
    )
    parser.add_argument(
        "--password", type=str, default=None, help="Password to decrypt the Ethereum private key (Pearl v1)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the key is encrypted without decrypting it",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Decrypt and overwrite the key file with plaintext version",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output decrypted key to specified file instead of stdout",
    )

    args = parser.parse_args()
    key_file = Path(args.key_file)

    try:
        # Check mode - just report if encrypted
        if args.check:
            with open(key_file, "r") as f:
                content = f.read().strip()

            is_encrypted = is_encrypted_keystore(content)
            if is_encrypted:
                print(f"✅ Key file is encrypted (V3 keystore)")
                # Try to extract address from keystore
                try:
                    keystore = json.loads(content)
                    if "address" in keystore:
                        print(f"📍 Address: 0x{keystore['address']}")
                except:
                    pass
                sys.exit(0)
            else:
                print(f"ℹ️  Key file is plaintext (not encrypted)")
                # Try to get address
                try:
                    account = Account.from_key(content.strip())
                    print(f"📍 Address: {account.address}")
                except:
                    pass
                sys.exit(0)

        # Load and decrypt the key
        private_key = load_private_key(key_file, args.password)

        # Verify it's a valid key by creating an account
        account = Account.from_key(private_key)

        # In-place mode - overwrite the file
        if args.in_place:
            with open(key_file, "w") as f:
                f.write(private_key)
            print(f"✅ Successfully decrypted key and saved to {key_file}")
            print(f"📍 Ethereum Address: {account.address}")
            print(f"🔐 Key Type: Plaintext (decrypted)")

        # Output to file
        elif args.output:
            output_file = Path(args.output)
            with open(output_file, "w") as f:
                f.write(private_key)
            print(f"✅ Successfully decrypted key and saved to {output_file}")
            print(f"📍 Ethereum Address: {account.address}")

        # Output to stdout
        else:
            print(private_key)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print(f"💡 Make sure {args.key_file} exists", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
