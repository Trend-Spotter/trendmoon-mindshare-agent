# Pearl v1 Encrypted Key Support

## Overview

The Mindshare agent supports both plaintext and encrypted V3 keystore formats for the Agent EOA private key, as required by Pearl v1.

## Current Implementation Status

### Framework Support

The agent uses the Open AEA framework which includes `eth-keyfile` (v0.8.1) as a dependency. This library provides native support for V3 encrypted Ethereum keystores.

### Key File Format

The `ethereum_private_key.txt` file can contain either:

1. **Plaintext hex** (Quickstart backward compatibility):
   ```
   0x1234567890abcdef...
   ```

2. **Encrypted V3 keystore JSON** (Pearl v1):
   ```json
   {
     "address": "...",
     "crypto": { ... },
     "id": "...",
     "version": 3
   }
   ```

## Creating Encrypted Keys

To create an encrypted private key for testing:

```python
from eth_account import Account
import json

# Create or load your account
account = Account.from_key('0xYOUR_PRIVATE_KEY')

# Encrypt it with a password
encrypted_keystore = Account.encrypt(account.key.hex(), 'your-password')

# Save to file
with open('ethereum_private_key.txt', 'w') as f:
    json.dump(encrypted_keystore, f, indent=2)

print(f"Encrypted keystore created for address: {account.address}")
```

## Password Handling

**Note**: Password handling for encrypted keys is expected to be managed at the deployment level (Pearl v1 orchestration/quickstart layer), not within the agent code itself.

The Open AEA framework and its connections (dcxt, ledger) should automatically detect and handle encrypted keystores when the appropriate password mechanism is provided by the deployment environment.

## Testing

To verify encrypted key support:

1. Generate an encrypted keystore using the script above
2. Deploy the agent using the encrypted `ethereum_private_key.txt`
3. Verify the agent starts correctly and can sign transactions

## Backward Compatibility

The agent maintains full backward compatibility with plaintext keys used by the Quickstart deployment method.

## References

- [Pearl v1 Requirements](../quickstart/pearl_v1_implementations/PEARL_V1_REQUIREMENTS.md)
- [Encrypted EOA Reference Implementation](../quickstart/pearl_v1_implementations/encrypted_eoa_handler.py)
- `eth-keyfile` library: https://github.com/ethereum/eth-keyfile
