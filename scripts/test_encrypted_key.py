#!/usr/bin/env python3
"""
Test script for encrypted key handling.

This script tests the Pearl v1 encrypted key support by:
1. Creating test keys (plaintext and encrypted)
2. Testing decryption with the utility script
3. Verifying backward compatibility
"""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from eth_account import Account


def run_test(name: str, test_func):
    """Run a test and report results."""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Test: {name}")
        print(f"{'='*60}")
        test_func()
        print(f"✅ PASSED: {name}")
        return True
    except AssertionError as e:
        print(f"❌ FAILED: {name}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   Unexpected error: {e}")
        return False


def test_plaintext_key():
    """Test handling of plaintext private key."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test account
        account = Account.create()
        private_key = account.key.hex()

        # Write plaintext key
        key_file = tmpdir / "ethereum_private_key.txt"
        with open(key_file, "w") as f:
            f.write(private_key)

        print(f"📝 Created plaintext key for address: {account.address}")

        # Test: Check without password should work
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file), "--check"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Check failed: {result.stderr}"
        assert "plaintext" in result.stdout.lower(), "Should detect plaintext key"
        print(f"✓ Detected as plaintext key")

        # Test: Load without password should work
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"
        loaded_key = result.stdout.strip()

        # Verify the loaded key is correct
        loaded_account = Account.from_key(loaded_key)
        assert loaded_account.address == account.address, "Address mismatch"
        print(f"✓ Loaded key correctly: {loaded_account.address}")


def test_encrypted_key():
    """Test handling of encrypted V3 keystore."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test account
        account = Account.create()
        password = "test_password_123"

        # Encrypt the key
        encrypted_keystore = Account.encrypt(account.key, password)

        # Write encrypted keystore
        key_file = tmpdir / "ethereum_private_key.txt"
        with open(key_file, "w") as f:
            json.dump(encrypted_keystore, f, indent=2)

        print(f"📝 Created encrypted key for address: {account.address}")
        print(f"🔐 Password: {password}")

        # Test: Check should detect encrypted
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file), "--check"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Check failed: {result.stderr}"
        assert "encrypted" in result.stdout.lower(), "Should detect encrypted key"
        print(f"✓ Detected as encrypted key")

        # Test: Load without password should fail
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail without password"
        assert "password" in result.stderr.lower(), "Should mention password in error"
        print(f"✓ Correctly rejected load without password")

        # Test: Load with wrong password should fail
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file), "--password", "wrong_password"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail with wrong password"
        print(f"✓ Correctly rejected wrong password")

        # Test: Load with correct password should work
        result = subprocess.run(
            ["python", "scripts/decrypt_key.py", "--key-file", str(key_file), "--password", password],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"
        loaded_key = result.stdout.strip()

        # Verify the loaded key is correct
        loaded_account = Account.from_key(loaded_key)
        assert loaded_account.address == account.address, "Address mismatch"
        print(f"✓ Decrypted key correctly: {loaded_account.address}")


def test_in_place_decryption():
    """Test in-place decryption of encrypted key."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test account
        account = Account.create()
        password = "test_password_456"

        # Encrypt the key
        encrypted_keystore = Account.encrypt(account.key, password)

        # Write encrypted keystore
        key_file = tmpdir / "ethereum_private_key.txt"
        with open(key_file, "w") as f:
            json.dump(encrypted_keystore, f, indent=2)

        print(f"📝 Created encrypted key for address: {account.address}")

        # Decrypt in-place
        result = subprocess.run(
            [
                "python",
                "scripts/decrypt_key.py",
                "--key-file",
                str(key_file),
                "--password",
                password,
                "--in-place",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"In-place decryption failed: {result.stderr}"
        print(f"✓ In-place decryption succeeded")

        # Read the file - should now be plaintext
        with open(key_file, "r") as f:
            content = f.read().strip()

        # Verify it's now plaintext
        assert not content.startswith("{"), "File should no longer be JSON"
        assert content.startswith("0x"), "Should be hex with 0x prefix"

        # Verify the key is correct
        decrypted_account = Account.from_key(content)
        assert decrypted_account.address == account.address, "Address mismatch after decryption"
        print(f"✓ File now contains plaintext key: {decrypted_account.address}")


def test_backward_compatibility():
    """Test that plaintext keys work exactly as before."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test keys both with and without 0x prefix
        account1 = Account.create()
        account2 = Account.create()

        # Key with 0x prefix
        key_file1 = tmpdir / "key_with_prefix.txt"
        with open(key_file1, "w") as f:
            f.write("0x" + account1.key.hex())  # account.key.hex() doesn't include 0x

        # Key without 0x prefix
        key_file2 = tmpdir / "key_without_prefix.txt"
        with open(key_file2, "w") as f:
            f.write(account1.key.hex())  # Just the raw hex, no 0x

        print(f"📝 Testing backward compatibility")

        # Test both formats
        for key_file, desc in [(key_file1, "with 0x"), (key_file2, "without 0x")]:
            result = subprocess.run(
                ["python", "scripts/decrypt_key.py", "--key-file", str(key_file)],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Failed for key {desc}"
            loaded_key = result.stdout.strip()
            loaded_account = Account.from_key(loaded_key)
            assert loaded_account.address == account1.address, f"Address mismatch for key {desc}"
            print(f"✓ Plaintext key {desc} works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 Pearl v1 Encrypted Key Support Test Suite")
    print("=" * 60)

    tests = [
        ("Plaintext Key Handling", test_plaintext_key),
        ("Encrypted Key Handling", test_encrypted_key),
        ("In-Place Decryption", test_in_place_decryption),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []
    for name, test_func in tests:
        passed = run_test(name, test_func)
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed_count}/{total_count} tests passed")
    print(f"{'=' * 60}\n")

    if passed_count == total_count:
        print("🎉 All tests passed! Pearl v1 encrypted key support is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
