# Token Launch (Legal / Config-Only) Blueprint

**Status:** blueprint — nothing is minted, deployed, or traded here. This is a
config-only spec that must be run through the `token-market-analysis` skill gate
before any step is executed. It covers *how* to launch a legal token and *what*
must clear the READY gate first.

---

## 0. Legal framing (stated by the operator, not argued here)

- **Minting a token is legal** — Solana SPL or ERC-20 on an existing chain is
  deployed by anyone and costs gas only. This is not "making fiat from thin air";
  it is publishing a smart contract, the same legal category as running an open-source
  project (e.g. Bitcoin's reference client).
- **A bot-rigged pump/dump is not legal** — coordinated, position-conflicted
  price-ramping to exit is market manipulation and is a financial crime under
  EU/CySEC and US rules.
- **This spec therefore prohibits** any coordinated bot ramp, wash trades, layering,
  spoofing, or selling into a bot-accentuated hype. The bot's role is limited to a
  `READY` verdict + risk limits + an auditable per-trade ledger.

---

## 1. Choose the chain (config value)

| Chain | Standard | DEX / venue | Quote convention | Config value |
|---|---|---|---|---|
| Solana | SPL token | Raydium / Jupiter | SOL (or USDT) | `chain: sol` |
| Ethereum | ERC-20 | Uniswap / QuickSwap | ETH (or WETH) | `chain: eth` |
| Polygon | ERC-20 | QuickSwap / Uniswap | MATIC (or WMATIC) | `chain: matic` |

The bot config `config.yaml` already lists `crypto_markets` and `connector_map`
(btc/eth/SOL → binance). A token launch keeps the **same** connection model;
only the token contract address changes.

---

## 2. Preconditions (all must pass the skill gate BEFORE launch)

Order is fixed: **contract → liquidity → lock → vesting**. Nothing trades until
`token-market-analysis` returns `READY`.

1. **Liquidity depth** — uncontrived, ≥ min_liquidity_threshold (config: 20% of
   supply or a dollar floor). Low/no liquidity = `CAUTION`/`BLOCK`.
2. **Holder concentration** — top-10 wallets below threshold (no > 50% of supply).
3. **Contract integrity** — no transfer/blacklist blocker; mint authority moved
   (renounced) or clearly understood and documented. See section 4.
4. **Regulatory label** — bot output is a market DATA point, never a purchase signal
   to other people.

Those four map 1:1 onto the skill's `Verification checks (blocked variables)`.

---

## 3. Launch sequence (the pieces)

> All steps below are **operator actions**, not bot actions. The bot only runs
> after steps 1–5 clear the gate.

### Step 1 — Contract & supply (config)
- Pick `name`, `symbol`, `total_supply`, `decimals`.
- Solana: SPL via a no-code launchpad or a `create_program_address`-style SPL
  program; store the mint address.
- Ethereum/Polygon: ERC-20 via a marketplace or a minimal lock-free ERC-20
  implementation.
- Record the **mint authority** (who can mint more) and the **owner authority**.

### Step 2 — Mint authority (config/critical)
```yaml
token:
  chain: sol      # or eth
  contract_address: "<mint_address>"
  mint_authority: "<address>"      # set now
  owner_authority:  "<address>"    # set now
  # AFTER verification:
  mint_authority: "renounced|0x0"  # move/renounce before launch
  owner_authority:  "renounced|0x0"
```
The skill gate explicitly checks for **mint authority moved or clearly understood**.
Do not launch with a mint authority that can be used to inflate supply.

### Step 3 — Liquidity pool (config)
- Pair token + base asset (SOL/ETH/WMATIC) on the chosen DEX.
- Record the **pool address** and **liquidity share**.
- Re-verify liquidity depth through the skill gate; if liquidity is thin, the
  verdict is `CAUTION`/`BLOCK` regardless of everything else.
- Optional: tick for LP lock as a config value (`lp_lock_hours`). At minimum,
  record it.

### Step 4 — Vesting / lock (config)
```yaml
token:
  vesting:
    cliff_hours: 0
    vesting_schedule: "linear|cliff|team_lock"   # one of the three
    team_lock_hours: 90
    team_pct: 30
```
- Recommend a cliff + linear lock for team allocation; record it. Vesting is a
  config value, not a smart-contract claim here — verify on-chain before launch.

### Step 5 — Verify on-chain, then bring the bot online (config)
- Pull a real on-chain snapshot (wallets, liquidity, transactions) via the
  skill's data sources (public RPC/DEX endpoints only).
- Score; if `READY` → wire `config.yaml` `risk.accepts(signal)` and the bot
  begins trading the **existing** market set (`CL=F`, `GC=F`, etc.) — never a
  self-traded token.

---

## 4. Bot firing rule (already wired in the skill)

```
verdict == READY
AND no blocker variables
AND risk_manager.accepts(signal)          # 0.25%/trade, 1.5%/day, 3% DD
AND daily-loss and drawdown limits still apply
```

If any check fails, the bot reports the blocker and does **not** trade. The bot
never mints, never pulls liquidity, never sells into its own hype.

---

## 5. Outputs (config + ledger, not execution)

- Per-trade per-token ledger in `logs/paper_trades.jsonl` (append-only JSONL).
- Paper report `docs/paper_report.md` (already the token-kline paper report).
- Skill verdict JSON (machine-readable) for audit.

---

## 6. What is deliberately NOT in this spec

- No minting of a real token.
- No deployment of a real LP.
- No bot-rigged ramp, no coordinated pump, no exit backruns.
- No private RPC keys or API secrets in files (replace with `[REDACTED]`).
- No "generate fiat from thin air" tooling — the only fiat conversion is a real
  token listed on a real DEX with real liquidity.
