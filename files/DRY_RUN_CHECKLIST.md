# Prediction Bot — Pre-Flight Checklist
## From zero to first live trade safely

Work through this top to bottom. Do not skip steps.
Every item has a pass/fail test you can run right now.

---

## PHASE 1: Environment Setup

### 1.1 Python version
```bash
python --version
# Required: 3.10 or higher
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
# Watch for any failed installs — torch + transformers are large (~2GB)
# If torch fails: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 Create your .env file
```bash
cp .env.example .env
```
Open `.env` and fill in credentials (see Phase 2 for where to get each one).

---

## PHASE 2: Credential Acquisition

### 2.1 Polymarket — MOST IMPORTANT

Go to: https://polymarket.com

**Step 1: Create a wallet**
- Use a DEDICATED wallet for the bot. Never use your main wallet.
- Recommended: Create a new MetaMask wallet specifically for trading
- Fund it with USDC on Polygon network (not Ethereum mainnet)
- Start with a small amount — suggest $200-500 for first run

**Step 2: Get API credentials**
- Log into Polymarket
- Go to: https://polymarket.com/settings (or profile → API Keys)
- Generate API key, secret, and passphrase
- Copy all three to your .env:
  ```
  POLYMARKET_API_KEY=your_key
  POLYMARKET_API_SECRET=your_secret
  POLYMARKET_API_PASSPHRASE=your_passphrase
  POLYMARKET_WALLET_ADDRESS=0x...your_wallet_address
  POLYMARKET_PRIVATE_KEY=0x...your_private_key
  ```

⚠️  PRIVATE KEY SECURITY:
- Never commit .env to git (it's in .gitignore)
- Never share your private key with anyone
- The private key signs on-chain transactions — treat it like cash

**Verify Polymarket access:**
```bash
python -c "
import requests, os
from dotenv import load_dotenv
load_dotenv()
r = requests.get('https://clob.polymarket.com/markets?limit=1')
print('Polymarket API reachable:', r.status_code == 200)
"
```

### 2.2 Anthropic API Key

Go to: https://console.anthropic.com
- Create a new API key
- Add to .env: `ANTHROPIC_API_KEY=sk-ant-...`
- Recommended: Set a usage limit of $50/month to start

**Verify:**
```bash
python -c "
import anthropic, os
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()
msg = client.messages.create(model='claude-sonnet-4-20250514',
    max_tokens=10, messages=[{'role':'user','content':'ping'}])
print('Anthropic API OK:', msg.content[0].text)
"
```

### 2.3 Twitter / X API (optional but recommended)

Go to: https://developer.x.com/en/portal/dashboard
- Apply for Basic access (~$100/month) — Free tier has very low limits
- Create an app → generate Bearer Token
- Add to .env: `TWITTER_BEARER_TOKEN=...`

If you want to start without Twitter:
- The research_agent gracefully skips unavailable sources
- Set `TWITTER_BEARER_TOKEN=` (empty) and sentiment will use Reddit + RSS only

### 2.4 Reddit API (free, instant)

Go to: https://www.reddit.com/prefs/apps
- Click "create another app"
- Type: "script"
- Name: "prediction-bot" (anything)
- Redirect URI: http://localhost:8080
- Copy client_id (under app name) and secret
- Add to .env:
  ```
  REDDIT_CLIENT_ID=your_id
  REDDIT_CLIENT_SECRET=your_secret
  REDDIT_USER_AGENT=prediction-bot/1.0
  ```

---

## PHASE 3: Database & Model Setup

### 3.1 Initialize database
```bash
python -c "from data.db import init_db; init_db(); print('DB initialized')"
```
Expected: `DB initialized`
Check: `ls data/prediction_bot.db` — file should exist

### 3.2 Test market ingestion (read-only, safe to run anytime)
```bash
python scripts/run_bot.py ingest
```
Expected output:
```
Starting market ingestion (target: 300 markets)
Scanned 100 | Saved 23 | cursor=...
Scanned 200 | Saved 51 | cursor=...
...
Ingestion complete. XX/300 markets passed filter.
```
If saved count is 0: your filter thresholds in config.py may be too strict.
Temporarily lower `min_liquidity_usdc` to 500 and `min_volume_24h_usdc` to 100 to test.

### 3.3 Train XGBoost model
```bash
python models/xgboost_model.py train
```

**Interpreting results:**
```
✅ AUC > 0.65  → Model has predictive signal. Use it.
⚠️  AUC 0.55-0.65 → Marginal. Bot runs in heuristic mode (safer).
❌ AUC ~0.50  → Random. Bot uses heuristic fallback automatically.
```

If training fails with "not enough samples": Polymarket may not have enough
resolved markets in the API right now. Bot uses heuristic fallback — this is
fine for starting. Re-run training weekly as more markets resolve.

Check model status anytime:
```bash
python models/xgboost_model.py info
```

---

## PHASE 4: Dry Run (NO real money, full logic)

### 4.1 Run the full pipeline in dry-run mode
```bash
python scripts/run_bot.py dry-run
```

Watch the logs for one full cycle. You should see:
```
─── Starting trading cycle ───
Candidates for this cycle: 15
Edge found: +0.08 on 'Will X happen by...'
[DRY RUN] Would place YES $47.32 on ab12cd34
─── Cycle complete ───
```

**If you see NO edge found on any market:**
This is normal and actually healthy — it means your thresholds are working.
Real edge (>4%) is rare. Check back over multiple cycles across a day.

**If you see errors:**
- `ImportError`: re-run `pip install -r requirements.txt`
- `Connection refused` on Polymarket: check your API key in .env
- `Claude API error`: check ANTHROPIC_API_KEY and billing

### 4.2 Inspect what the bot would have traded
```bash
# After running dry-run for a few hours, check what was logged:
python -c "
import sqlite3
conn = sqlite3.connect('data/prediction_bot.db')
rows = conn.execute('''
    SELECT m.question, p.claude_prob, p.market_yes_price,
           p.edge_pct, p.confidence
    FROM predictions p
    JOIN markets m ON m.condition_id = p.condition_id
    ORDER BY p.predicted_at DESC LIMIT 20
''').fetchall()
for r in rows:
    print(f'Edge: {r[3]:+.1%} | Conf: {r[4]} | {r[0][:60]}')
conn.close()
"
```

Review these manually. Ask yourself:
- Do the edges make intuitive sense?
- Is the confidence rating reasonable?
- Are there obvious bad predictions? (If so, note the pattern for config tuning)

---

## PHASE 5: Risk Configuration Review

Before going live, confirm these settings in `config.py` match your actual bankroll:

```python
# For a $500 starting bankroll — conservative settings:
kelly_fraction          = 0.25   # quarter Kelly ✓
max_kelly_bet_pct       = 0.05   # max 5% = $25 per trade ✓
max_single_position_usdc = 25.0  # hard cap at $25 ✓
max_total_exposure_usdc  = 150.0 # max $150 open at once ✓
daily_loss_limit_usdc    = 50.0  # pause if down $50/day ✓

# For a $2000 bankroll:
max_single_position_usdc = 100.0
max_total_exposure_usdc  = 600.0
daily_loss_limit_usdc    = 200.0
```

⚠️  Golden rule: Set `daily_loss_limit_usdc` to 10% of your bankroll.
The bot will pause automatically if hit. You review and restart manually.

---

## PHASE 6: Go Live

### 6.1 Final checks before flipping to live
```
[ ] .env has all credentials filled in
[ ] Polymarket wallet funded with USDC (Polygon network)
[ ] Database initialized and ingestion tested
[ ] XGBoost trained (or heuristic fallback confirmed working)
[ ] Dry run ran for at least 2 hours with sensible predictions
[ ] Risk limits set to match actual bankroll
[ ] Running on a machine that will stay online (or set up as a service)
[ ] You have checked that prediction_bot.db is NOT in your git repo
```

### 6.2 Start live trading
```bash
python scripts/run_bot.py loop
```

First thing to watch:
- First cycle should run within 5 minutes
- Check logs/bot.log for any errors
- Confirm the first real order in Polymarket's UI before walking away

### 6.3 Monitor ongoing
```bash
# Check today's P&L
python -c "
from data.db import get_daily_pnl, get_open_exposure
print(f'Daily P&L: \${get_daily_pnl():.2f}')
print(f'Open exposure: \${get_open_exposure():.2f}')
"

# Check trade history
python -c "
import sqlite3
conn = sqlite3.connect('data/prediction_bot.db')
rows = conn.execute('''
    SELECT direction, size_usdc, status, pnl_usdc, placed_at
    FROM trades ORDER BY placed_at DESC LIMIT 10
''').fetchall()
for r in rows: print(r)
conn.close()
"
```

---

## TROUBLESHOOTING

**Bot places no trades over 24 hours:**
Normal if markets lack edge. Lower `min_edge_pct` to 0.03 temporarily to test execution pipeline, then raise back.

**Orders placed but never filled:**
Polymarket CLOB fills limit orders — check if your price is competitive. execute_agent places at best ask, which should fill immediately. Check `trades` table for `status='pending'` orders that are stuck.

**Claude API costs high:**
predict_agent calls Claude for every market with sufficient edge. If costs spike, raise `min_edge_pct` to 0.06 to reduce Claude calls.

**XGBoost predictions seem off:**
The model has no historical sentiment data (all trained with sentiment=0). This is expected. As learn_agent accumulates data and you retrain periodically, it improves. Run `python models/xgboost_model.py train` weekly.
