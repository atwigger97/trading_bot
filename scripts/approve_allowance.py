"""
scripts/approve_allowance.py — One-time USDC allowance approval for Polymarket.

Approves the Polymarket CTF Exchange contract to spend native USDC
from the bot wallet on Polygon mainnet.

Usage:
    cd /root/prediction-bot/trading_bot
    source venv/bin/activate
    python3 scripts/approve_allowance.py
"""

import os
import sys

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native USDC on Polygon
CLOB_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Polymarket CTF Exchange
MAX_UINT256 = 2**256 - 1

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("POLYMARKET_WALLET_ADDRESS")

if not PRIVATE_KEY or not WALLET_ADDRESS:
    print("ERROR: POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS must be set in .env")
    sys.exit(1)

# ─── Web3 setup ──────────────────────────────────────────────────────────────

w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

addr = Web3.to_checksum_address(WALLET_ADDRESS)

ERC20_ABI = [
    {
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
        "stateMutability": "nonpayable",
    },
    {
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
        "stateMutability": "view",
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
        "stateMutability": "view",
    },
]

usdc = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)

# ─── Pre-flight checks ──────────────────────────────────────────────────────

matic = w3.eth.get_balance(addr)
balance = usdc.functions.balanceOf(addr).call()
current_allowance = usdc.functions.allowance(addr, CLOB_EXCHANGE).call()

print(f"Wallet:            {addr}")
print(f"MATIC:             {w3.from_wei(matic, 'ether'):.4f}")
print(f"USDC balance:      {balance / 1e6:.2f}")
print(f"Current allowance: {'MAX' if current_allowance > 10**30 else f'{current_allowance / 1e6:.2f}'}")
print(f"Spender:           {CLOB_EXCHANGE} (Polymarket CTF Exchange)")

if current_allowance > 10**30:
    print("\nAllowance already set to MAX. Nothing to do.")
    sys.exit(0)

if matic == 0:
    print("\nERROR: No MATIC for gas fees. Send some POL/MATIC to the wallet first.")
    sys.exit(1)

# ─── Send approval TX ───────────────────────────────────────────────────────

print("\nSending approval transaction...")

base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
nonce = w3.eth.get_transaction_count(addr)

tx = usdc.functions.approve(CLOB_EXCHANGE, MAX_UINT256).build_transaction({
    "from": addr,
    "nonce": nonce,
    "gas": 80000,
    "maxFeePerGas": base_fee * 3,
    "maxPriorityFeePerGas": w3.to_wei(50, "gwei"),
    "chainId": 137,
})

signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
print(f"TX hash:           {tx_hash.hex()}")

receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
print(f"Block:             {receipt['blockNumber']}")
print(f"Status:            {'SUCCESS' if receipt['status'] == 1 else 'FAILED'}")

# ─── Verify ─────────────────────────────────────────────────────────────────

new_allowance = usdc.functions.allowance(addr, CLOB_EXCHANGE).call()
print(f"New allowance:     {'MAX' if new_allowance > 10**30 else new_allowance / 1e6:.2f}")

if receipt["status"] == 1:
    print("\nDone! The bot can now place orders on Polymarket.")
else:
    print("\nApproval transaction FAILED. Check the TX on Polygonscan.")
    sys.exit(1)
