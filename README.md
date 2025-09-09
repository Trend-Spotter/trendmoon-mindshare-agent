# trendmoon-mindshare-agent

An open-source AI agent built for the Olas ecosystem as part of the TrendMoon initiative. The Mindshare Agent analyzes on-chain and off-chain sentiment signals to detect early narratives, helping users stay ahead of trends in crypto.

## ⚠️ Disclaimer

This repository contains experimental software that interacts with blockchains, smart contracts, and external APIs. By running this code, you may put real funds and digital assets at risk.

The code has **not been audited** for security vulnerabilities.

The maintainers provide **no warranties, guarantees, or assurances of any kind**.

Bugs, logic errors, or exploits may result in loss of funds, compromised accounts, or permanent asset loss.

The software is provided “**AS IS**” under the terms of the LICENSE file.

You alone are responsible for reviewing, testing, and securing this code before use. Running this agent implies that you understand the risks and accept full liability for any consequences, including financial loss.

If you are not comfortable with these risks, **do not use this software in production or with real funds**.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Setup for Development](#setup-for-development)
- [Usage](#usage)
- [Commands](#commands)
  - [Testing](#testing)
  - [Linting](#linting)
  - [Formatting](#formatting)
  - [Releasing](#releasing)
- [License](#license)

## Getting Started

### Installation and Setup for Development

If you're looking to contribute or develop with `trendmoon-mindshare-agent`, get the source code and set up the environment:

```shell
git clone https://github.com/Trend-Spotter/trendmoon-mindshare-agent/ --recurse-submodules
cd trendmoon-mindshare-agent
make install
```

## Commands

Here are common commands you might need while working with the project:

### Formatting

```shell
make fmt
```

### Linting

```shell
make lint
```

### Testing

```shell
make test
```

### Locking

```shell
make hashes
```

### all

```shell
make all
```
