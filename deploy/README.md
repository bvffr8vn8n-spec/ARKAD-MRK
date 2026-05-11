# Deploy — ARKAD MRK paper trader on a Linux server

Minimal recipe to run `paper_trader.py` 24/7 under systemd. Tested on
Ubuntu 22.04 / Debian 12. Python 3.11+ recommended.

## 1. Prepare the host

```bash
sudo useradd -m -s /bin/bash arkad
sudo mkdir -p /opt/arkad-mrk
sudo chown arkad:arkad /opt/arkad-mrk
```

## 2. Clone and install

```bash
sudo -u arkad -i
cd /opt/arkad-mrk
git clone https://github.com/bvffr8vn8n-spec/ARKAD-MRK.git .
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 3. Provide secrets and historical data

```bash
# 3a. API keys — never commit this file
cat > paper_trading/secrets.py <<'EOF'
BYBIT_API_KEY    = "YOUR_KEY"
BYBIT_API_SECRET = "YOUR_SECRET"
EOF
chmod 600 paper_trading/secrets.py

# 3b. Historical OHLCV (1H + 5m for each Tier 1 asset)
python data/download_all.py
```

## 4. First-time sanity check

```bash
. .venv/bin/activate
python paper_trader.py --status   # should show equity $10,000 and seeded timestamps
```

Run the loop once in the foreground to confirm Bybit reachability and model
training, then Ctrl-C to stop:

```bash
python paper_trader.py
# Wait until "All models trained.  Entering poll loop."
# Wait one or two poll cycles -- you should see "X new 15m bar(s) to process"
# Ctrl-C to stop -- expect the "Paper Trader stopped.  Uptime: ..." banner.
```

## 5. Install the systemd unit

```bash
# Edit User=, WorkingDirectory=, ExecStart= if your paths differ
sudo cp deploy/arkad-paper-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arkad-paper-trader
```

## 6. Operate

```bash
# Live tail of the log
sudo journalctl -u arkad-paper-trader -f

# OR tail the rotating file log directly
tail -f /opt/arkad-mrk/paper_trading/paper_trader.log

# Service status
sudo systemctl status arkad-paper-trader

# Graceful restart (SIGTERM -> drain current tick -> save state -> restart)
sudo systemctl restart arkad-paper-trader

# Graceful stop
sudo systemctl stop arkad-paper-trader

# Current trader state (read-only, safe to run any time)
sudo -u arkad /opt/arkad-mrk/.venv/bin/python /opt/arkad-mrk/paper_trader.py --status
```

## 7. Health checks to grep for

Greppable log signals worth alerting on:

| Pattern | Meaning | Action |
|---|---|---|
| `Shutdown requested` | SIGTERM/SIGINT received | Expected on stop/restart |
| `[circuit:bybit] N consecutive failures -> OPEN` | Bybit unreachable for ~N×retries | Investigate network/DDoS |
| `[circuit:bybit] success on HALF_OPEN -> CLOSED` | Recovery confirmed | No action |
| `Retry N/M for https://api.bybit.com/...` | Transient error, retried | No action unless frequent |
| `consecutive=50` in tick error log | Critical: 50 ticks failed in a row | On-call escalation |
| `TRADE OPEN` / `TRADE CLOSE` | Normal trade lifecycle | Periodic audit only |
| `Equity updated` | After each closed trade | None |

## 8. Backups

`paper_trades_tier1.csv` is append-only and the canonical trade log. Back it up
daily:

```cron
0 3 * * *  cp /opt/arkad-mrk/paper_trades_tier1.csv  /var/backups/arkad/paper_trades_$(date +\%F).csv
```

`paper_trading/state.json` is rewritten atomically every tick; do **not** edit
it manually. If it ever gets corrupted, stop the service, move it aside, and
let `state_store.load()` reseed on next start (equity will be lost — recover
manually from the CSV log).

## 9. Updating

```bash
sudo systemctl stop arkad-paper-trader
sudo -u arkad -i
cd /opt/arkad-mrk
git pull
. .venv/bin/activate
pip install -r requirements.txt  # if requirements changed
exit
sudo systemctl start arkad-paper-trader
```

Schema-compatible updates (added fields, etc.) survive restart automatically —
`from_dict()` ignores unknown keys and uses defaults for missing ones.

## 10. What this gives you

- HTTP retry with 1s / 2s / 4s backoff on 429 / 5xx / connection / timeout errors
- Global circuit breaker: 10 consecutive failures → 60 s cooldown, then probe
- Graceful SIGTERM shutdown: drains current tick, saves state, exits 0
- Atomic state writes (state.json is never corrupted by a kill or power loss)
- Idempotent bar processing: same bar never enters the system twice
- Restart recovery: monitors and open trades reload from state.json
- systemd auto-restart on crash (rate-limited to 10/hour)
