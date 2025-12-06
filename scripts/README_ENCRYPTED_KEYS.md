# Encrypted Private Key Support (Pearl v1)

This directory contains utilities for handling encrypted Ethereum private keys as required by Pearl v1.

## Quick Start

### Check if your key is encrypted

```bash
python scripts/decrypt_key.py --check
```

### Decrypt an encrypted key (output to stdout)

```bash
python scripts/decrypt_key.py --password mypassword
```

### Decrypt and save in-place

```bash
python scripts/decrypt_key.py --password mypassword --in-place
```

## Usage Examples

### Example 1: Create an Encrypted Key

```python
from eth_account import Account
import json

# Create or load your private key
account = Account.from_key('0xYOUR_PRIVATE_KEY')

# Encrypt it with a password
password = "your_secure_password"
encrypted_keystore = Account.encrypt(account.key, password)

# Save to file
with open('ethereum_private_key.txt', 'w') as f:
    json.dump(encrypted_keystore, f, indent=2)

print(f"Encrypted keystore created for address: {account.address}")
```

### Example 2: Use with Agent Deployment

When deploying the agent with Pearl v1:

```bash
# Option 1: Decrypt before starting (recommended for testing)
python scripts/decrypt_key.py --password $AGENT_PASSWORD --in-place
# Now start your agent normally
make run

# Option 2: Keep encrypted and handle at orchestration layer
# The Pearl v1 orchestration will handle decryption
```

### Example 3: Migrate from Plaintext to Encrypted

```bash
# 1. Backup your current plaintext key
cp ethereum_private_key.txt ethereum_private_key.txt.backup

# 2. Create encrypted version
python -c "
from eth_account import Account
import json

# Read plaintext key
with open('ethereum_private_key.txt', 'r') as f:
    private_key = f.read().strip()

# Create account from key
account = Account.from_key(private_key)

# Encrypt with password
password = input('Enter encryption password: ')
encrypted = Account.encrypt(account.key, password)

# Save encrypted version
with open('ethereum_private_key.txt', 'w') as f:
    json.dump(encrypted, f, indent=2)

print(f'✅ Key encrypted for address: {account.address}')
"

# 3. Verify it works
python scripts/decrypt_key.py --password YOUR_PASSWORD --check
```

## Testing

Run the comprehensive test suite:

```bash
python scripts/test_encrypted_key.py
```

This tests:
- ✅ Plaintext key handling (backward compatibility)
- ✅ Encrypted V3 keystore handling
- ✅ In-place decryption
- ✅ Keys with and without 0x prefix

## File Formats Supported

### Plaintext (legacy, Quickstart compatible)

```
0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

or without 0x prefix:

```
1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

### Encrypted V3 Keystore (Pearl v1)

```json
{
  "address": "c9fe3494f6f96b57a3fe8c97055041f64c0b58d3",
  "crypto": {
    "cipher": "aes-128-ctr",
    "cipherparams": {
      "iv": "333c8d9d757fb91d7a1af36f388c8251"
    },
    "ciphertext": "e2f3f25bcfb4224e12ef7909cf885646b75828578def8bf92801a2b305a1bbe1",
    "kdf": "scrypt",
    "kdfparams": {
      "dklen": 32,
      "n": 262144,
      "r": 1,
      "p": 8,
      "salt": "89193d4941b92a45e8b4b99ce54c1d07"
    },
    "mac": "00ec6c5ca251adf5616598fe6ccd3eb9d4dced599124bf56e04d74c3fd1eeb1b"
  },
  "id": "100e1d76-f43f-42b1-abb9-2e2d2cbd96c9",
  "version": 3
}
```

## Command Reference

### decrypt_key.py

```
usage: decrypt_key.py [-h] [--key-file KEY_FILE] [--password PASSWORD]
                      [--check] [--in-place] [--output OUTPUT]

Pearl v1 Private Key Decryption Utility

optional arguments:
  -h, --help            show this help message and exit
  --key-file KEY_FILE   Path to the private key file (default: ethereum_private_key.txt)
  --password PASSWORD   Password to decrypt the Ethereum private key (Pearl v1)
  --check               Check if the key is encrypted without decrypting it
  --in-place            Decrypt and overwrite the key file with plaintext version
  --output OUTPUT       Output decrypted key to specified file instead of stdout
```

### test_encrypted_key.py

```bash
# Run all tests
python scripts/test_encrypted_key.py

# Tests included:
# - Plaintext key handling
# - Encrypted key handling
# - In-place decryption
# - Backward compatibility
```

## Security Notes

1. **Never commit plaintext private keys** to version control
2. **Use strong passwords** for encrypted keystores (minimum 12 characters recommended)
3. **Backup encrypted keystores** along with passwords in secure storage
4. **For production**, use Pearl v1 orchestration to handle passwords securely
5. **For development**, the `--in-place` decryption is convenient but less secure

## Integration with Pearl v1

Pearl v1 deployments will:
1. Provide encrypted V3 keystore files
2. Pass password via secure mechanism (not command line)
3. Handle decryption at orchestration layer
4. Never store plaintext keys on disk in production

This utility supports both the encrypted Pearl v1 format and maintains backward compatibility with plaintext keys for Quickstart deployments.

## Troubleshooting

### "Private key is encrypted but no password was provided"

You're trying to use an encrypted key without providing a password.

**Solution**: Provide the password with `--password` flag

### "Failed to decrypt private key: MAC mismatch"

The password is incorrect.

**Solution**: Double-check your password

### "Invalid private key length"

The key file is corrupted or in an unexpected format.

**Solution**:
- Check the file contains either valid hex (64 chars) or valid V3 keystore JSON
- Restore from backup if available

### "File not found: ethereum_private_key.txt"

The key file doesn't exist at the specified path.

**Solution**:
- Specify correct path with `--key-file`
- Generate a new key if needed

## Additional Resources

- [Pearl v1 Requirements](../PEARL_V1_ENCRYPTED_KEYS.md)
- [Implementation Summary](../PEARL_V1_IMPLEMENTATION_SUMMARY.md)
- [eth-account documentation](https://eth-account.readthedocs.io/)
