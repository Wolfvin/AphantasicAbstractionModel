# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.2.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability:

1. **Do NOT** open a public issue
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include: description, reproduction steps, affected versions, potential impact

We will:
- Acknowledge within 48 hours
- Provide an initial assessment within 7 days
- Release a fix as soon as possible
- Credit you in the security advisory (unless you prefer to remain anonymous)

## Security Features

- **Seed atom immutability**: 24 primitive atoms are locked at tier 1 with confidence 1.0 and cannot be modified or deleted
- **Policy engine**: Single-owner governance with hysteresis anti-flip-flop protection
- **Quarantine system**: Suspicious nodes are automatically quarantined before affecting the graph
- **Stability gate**: Global rollback mechanism prevents cascading corruption
- **Input validation**: All API endpoints validate request schemas and enforce constraints
- **DAG invariant**: The graph maintains a directed acyclic structure with circular reference detection
