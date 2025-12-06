# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrendMoon Mindshare Agent is an autonomous AI agent built on the Olas (formerly Autonolas) framework for decentralized multi-agent systems. It analyzes on-chain and off-chain sentiment signals to detect early crypto narratives and execute trades based on technical analysis and risk evaluation.

## Architecture

### Olas/AEA Framework
This project is built on the **Autonomous Economic Agent (AEA)** framework from Valory (Olas). Key architectural concepts:

- **Skills**: Self-contained modules with behaviors, handlers, models, and dialogues (located in `packages/xiuxiuxar/skills/`)
- **Agents**: Combinations of skills, connections, protocols, and contracts (located in `packages/xiuxiuxar/agents/`)
- **Services**: Multi-agent systems that can run as autonomous services (located in `packages/xiuxiuxar/services/`)
- **Connections**: Communication interfaces (HTTP, WebSockets, blockchain RPC, etc.)
- **Protocols**: Message formats for agent communication
- **Contracts**: Smart contract interfaces for blockchain interaction

### FSM-Based Trading Logic
The agent operates as a **Finite State Machine (FSM)** defined in `mindshare_fsm.yaml`. The FSM orchestrates the trading workflow through these rounds (states):

1. **SetupRound**: Initialize agent configuration and state
2. **CheckStakingKPIRound**: Verify staking requirements for Olas incentives
3. **DataCollectionRound**: Fetch market data, prices, and sentiment signals
4. **PositionMonitoringRound**: Check existing positions and exit conditions
5. **PortfolioValidationRound**: Validate portfolio constraints and trading capacity
6. **AnalysisRound**: Run technical indicators (RSI, MACD, ADX, moving averages)
7. **SignalAggregationRound**: Aggregate trading signals from multiple sources
8. **RiskEvaluationRound**: Assess risk and approve/reject trades
9. **TradeConstructionRound**: Build trade transactions for approved signals
10. **ExecutionRound**: Execute trades via CowSwap or DEX integrations
11. **HandleErrorRound**: Handle errors with retry logic
12. **PausedRound**: Pause operations when errors exceed retry limits

Each round is implemented as a behavior class in `packages/xiuxiuxar/skills/mindshare_app/behaviours/`.

### Main Skill: mindshare_app
The core logic is in `packages/xiuxiuxar/skills/mindshare_app/`:
- `behaviours/`: FSM round implementations (analysis, execution, data collection, etc.)
- `models.py`: Data models including Coingecko API client, Trendmoon API client, and technical indicators
- `handlers.py`: Message handlers for HTTP, contract calls, tickers, orders, ledger operations
- `dialogues.py`: Protocol dialogue management
- `skill.yaml`: Skill configuration including parameters, dependencies, and connections

### Trading Parameters
Key trading parameters are configured in `skill.yaml` under `models.params.args`:
- Portfolio constraints: `max_positions`, `max_total_exposure`, `max_exposure_per_position`
- Position sizing: `min_position_size_usdc`, `max_position_size_usdc`
- Risk management: `stop_loss_pct`, `trailing_stop_loss_pct`, `take_profit_risk_ratio`
- Technical indicators: RSI periods/thresholds, MACD periods, ADX thresholds, moving average lengths
- API configuration: CoinGecko and Trendmoon API keys, rate limits, monthly credits
- Staking: `staking_chain`, `on_chain_service_id`, staking contract addresses

### Allowed Trading Assets
Assets are whitelisted in `behaviours/base.py` under `ALLOWED_ASSETS`. Currently supports Base chain tokens including VIRTUAL, FLOCK, REI, CLANKER, etc. Each asset includes contract address, symbol, and CoinGecko ID.

## Development Commands

### Initial Setup
```shell
make install
```
This installs dependencies, sets up Git hooks, initializes AEA configuration, and syncs packages.

### Testing
```shell
make test              # Run all tests
poetry run adev -v test   # Run tests with verbose output
```

### Linting and Formatting
```shell
make lint              # Run ruff linter
make fmt               # Auto-format code with ruff
```

The project uses Ruff (configured in `ruff.toml`) with extensive rules for code quality, security, and style. Pre-commit hooks automatically format and lint code.

### Building
```shell
make all               # Format, lint, test, and update package hashes
make hashes            # Lock and push all packages (updates packages.json)
```

### Package Management
```shell
poetry run autonomy packages sync    # Sync packages from remote registries
make sync                             # Pull latest code and sync packages
```

### Agent Building
```shell
make build-agent-runner       # Build PyInstaller standalone binary (Linux)
make build-agent-runner-mac   # Build PyInstaller standalone binary (macOS)
```

## Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:
- `CONNECTION_LEDGER_CONFIG_LEDGER_APIS_ETHEREUM_ADDRESS`: Ethereum RPC URL
- `CONNECTION_LEDGER_CONFIG_LEDGER_APIS_BASE_ADDRESS`: Base chain RPC URL
- `CONNECTION_DCXT_CONFIG_EXCHANGES_0_RPC_URL`: Exchange RPC URL
- `CONNECTION_DCXT_CONFIG_EXCHANGES_0_ETHERSCAN_API_KEY`: Block explorer API key
- `SKILL_MINDSHARE_APP_MODELS_PARAMS_ARGS_COINGECKO_API_KEY`: CoinGecko API key
- `SKILL_MINDSHARE_APP_MODELS_PARAMS_ARGS_TRENDMOON_API_KEY`: TrendMoon API key
- `SKILL_MINDSHARE_APP_MODELS_PARAMS_ARGS_SAFE_CONTRACT_ADDRESSES`: Gnosis Safe addresses (JSON)
- `SKILL_MINDSHARE_APP_MODELS_PARAMS_ARGS_STORE_PATH`: Persistent data storage path

### Package Structure
- `packages/xiuxiuxar/`: Custom packages authored by xiuxiuxar
- `packages/packages.json`: Package registry with IPFS hashes
  - `dev`: Development packages (skill, agent, service)
  - `third_party`: External dependencies from Valory and other authors

## Technical Stack

- **Python**: 3.10-3.11 (managed via Poetry)
- **Frameworks**: Open AEA 2.x, Open Autonomy 0.21.1
- **Blockchain**: Ethereum, Base, Arbitrum, Optimism, Mode (via Gnosis Safe multi-sig)
- **Trading**: CowSwap protocol, Uniswap, DEX integrations
- **Data Sources**: CoinGecko API, TrendMoon API
- **Technical Analysis**: pandas-ta-classic for indicators
- **Testing**: pytest (via autonomy-dev)
- **Linting**: Ruff with comprehensive rule set

## Important Notes

### Security Considerations
- This is experimental software that handles real funds - code has not been audited
- Never commit sensitive data (.env files, private keys, credentials)
- All Safe transactions require multi-sig approval
- Review all trading parameters before deployment

### FSM State Transitions
When modifying the FSM, update both:
1. `mindshare_fsm.yaml`: State transition definitions
2. `packages/xiuxiuxar/skills/mindshare_app/behaviours/round_behaviour.py`: FSM behavior registration

### Package Fingerprinting
After modifying any package files, run `make hashes` to update fingerprints in `skill.yaml` and `packages.json`. This ensures package integrity in the Olas ecosystem.

### Olas Staking Integration
The agent participates in Olas staking programs:
- Must meet minimum transaction thresholds (`min_num_of_safe_tx_required`)
- Staking KPIs checked in `CheckStakingKPIRound`
- Checkpoint calls made to staking contracts in `CallCheckpointRound`

### AEA CLI Usage
Many operations use the AEA CLI:
```shell
poetry run autonomy fetch <hash_id>    # Fetch agent from IPFS
poetry run aea init --remote --author <name>   # Initialize AEA config
```

### Third-party Dependencies
The project includes a forked version of `autonomy-dev` in `third_party/forks/autodev`. This provides custom development tooling for the Olas ecosystem.
