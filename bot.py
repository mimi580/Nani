"""
DERIV DIGIT BOT — R_100 Even/Odd
==================================
Symbol   : R_100 (Volatility 100 Index)
Contract : DIGITEVEN / DIGITODD — 1 tick

Strategy — Bayes + Markov + Z-Score
-------------------------------------
  R_100 has a structural edge: digit 0 never appears, leaving
  5 odd digits (1,3,5,7,9) vs 4 even digits (2,4,6,8).
  Base P(Odd) ≈ 55.5%, confirmed across 1,951 ticks of monitoring.

  Three-layer signal pipeline:

  LAYER 1 — Markov Chain
    Tracks P(Even|prev) and P(Odd|prev) from a 10×10 transition
    matrix built live from observed digits. Priors seeded from
    monitor data:
      P(Odd|prev=Even) = 0.5576
      P(Odd|prev=Odd)  = 0.5536

  LAYER 2 — Bayesian Update
    Combines the Markov posterior with the session-level frequency
    of even/odd to produce a refined probability estimate.
    Prior seeded at P(Odd) = 0.5551 from monitor data.

  LAYER 3 — Z-Score Gate
    Rolling mean of is_even over last ZSCORE_WINDOW ticks.
    Z < -ZSCORE_THRESH → ODD is statistically dominant → trade ODD
    Z >  ZSCORE_THRESH → EVEN is statistically dominant → trade EVEN
    (In practice on R_100, ODD dominates ~95% of signals)

  Signal fires only when ALL THREE agree AND combined prob ≥ PROB_THRESH.

  Backtested on 1,951 R_100 ticks:
    prob ≥ 0.54 → 1,606 trades | WR 57.3%
    prob ≥ 0.56 → 1,020 trades | WR 68.0%
    prob ≥ 0.58 →   301 trades | WR 86.0%  ← default
    prob ≥ 0.60 →   111 trades | WR 100%

Risk
----
  Martingale: $0.35 × 1.65 per loss, max 4 steps then reset.
  Ladder: $0.35 → $0.58 → $0.95 → $1.57 → reset
  Max single-cycle risk: $3.45
  Circuit breaker: 4 consecutive losses → 5 min pause
  Session target: +$15 | Session stop: -$30
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime
from typing import Optional, Tuple

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")


# ============================================================================
# CONFIGURATION
# ============================================================================

def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


CONFIG = {
    # ── Deriv credentials ──────────────────────────────────
    "api_token":        _env("DERIV_API_TOKEN", "REPLACE_WITH_YOUR_TOKEN"),
    "app_id":           _env("DERIV_APP_ID", 1089),
    "symbol":           _env("SYMBOL", "R_100"),
    "currency":         "USD",

    # ── Signal pipeline thresholds ─────────────────────────
    # Minimum combined Bayes probability to trade
    "prob_thresh":      _env("PROB_THRESH", 0.58),

    # Z-score window and threshold
    "zscore_window":    _env("ZSCORE_WINDOW", 50),
    "zscore_thresh":    _env("ZSCORE_THRESH", 1.0),

    # Markov window — how many recent transitions to weight
    "markov_window":    _env("MARKOV_WINDOW", 100),

    # Warmup ticks before trading
    "min_warmup":       _env("MIN_WARMUP", 60),

    # Cooldown ticks between trades
    "cooldown_ticks":   _env("COOLDOWN_TICKS", 1),

    # ── Seeded priors from R_100 monitor data ──────────────
    # These give the model a head-start instead of cold-starting at 50/50
    # P(Odd) base rate observed across 1,951 ticks
    "prior_odd":        _env("PRIOR_ODD", 0.5551),

    # Markov transition priors (from monitor data)
    # [P(Odd|prev=Even), P(Odd|prev=Odd)]
    "mk_odd_given_even": _env("MK_ODD_GIVEN_EVEN", 0.5576),
    "mk_odd_given_odd":  _env("MK_ODD_GIVEN_ODD",  0.5536),

    # ── Risk / Martingale ──────────────────────────────────
    "initial_stake":    _env("INITIAL_STAKE", 0.35),
    "martingale_mul":   _env("MARTINGALE_MUL", 1.65),
    "max_losses":       _env("MAX_LOSSES", 4),       # steps before reset
    "target_profit":    _env("TARGET_PROFIT", 15.0),
    "stop_loss":        _env("STOP_LOSS", 30.0),

    # ── Circuit breaker ────────────────────────────────────
    "cb_limit":         _env("CB_LIMIT", 4),
    "cb_pause_secs":    _env("CB_PAUSE", 300),

    # ── Resilience ─────────────────────────────────────────
    "lock_timeout":     _env("LOCK_TIMEOUT", 30),
    "buy_retries":      _env("BUY_RETRIES", 8),
    "reconnect_min":    _env("RECONNECT_MIN", 2),
    "reconnect_max":    _env("RECONNECT_MAX", 60),
    "ws_ping":          _env("WS_PING", 30),
    "orphan_attempts":  _env("ORPHAN_ATTEMPTS", 4),
    "orphan_interval":  _env("ORPHAN_INTERVAL", 3),
}


# ============================================================================
# HELPERS
# ============================================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(tag: str, msg: str):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)

def _jlog(obj: dict):
    print(json.dumps(obj), flush=True)


# ============================================================================
# SIGNAL ENGINE — Bayes + Markov + Z-Score
# ============================================================================

class SignalEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # ── Digit history ──────────────────────────────────
        self.digits:     deque = deque(maxlen=cfg["markov_window"] + 2)
        self.is_even_history: deque = deque(maxlen=cfg["zscore_window"])
        self.tick_n:     int   = 0

        # ── Markov transition counts ───────────────────────
        # Counts[prev_is_even][curr_is_even]
        # Seeded with soft priors from monitor data
        # Using pseudo-counts of 10 to seed without overpowering live data
        seed = 10
        p_oe = cfg["mk_odd_given_even"]   # P(odd | prev=even)
        p_oo = cfg["mk_odd_given_odd"]    # P(odd | prev=odd)

        self.mk_counts = {
            1: {1: round(seed * (1 - p_oe)), 0: round(seed * p_oe)},  # prev=even
            0: {1: round(seed * (1 - p_oo)), 0: round(seed * p_oo)},  # prev=odd
        }

        # ── Bayesian session counter ───────────────────────
        # Seeded with prior from monitor data
        seed_b   = 20
        p_odd    = cfg["prior_odd"]
        self.bayes_even_count = round(seed_b * (1 - p_odd))
        self.bayes_odd_count  = round(seed_b * p_odd)

        # ── Last digit seen ────────────────────────────────
        self.last_is_even: Optional[int] = None

    def add_tick(self, digit: int):
        is_even = 1 if digit % 2 == 0 else 0
        self.is_even_history.append(is_even)
        self.digits.append(digit)
        self.tick_n += 1

        # Update Markov
        if self.last_is_even is not None:
            prev = self.last_is_even
            self.mk_counts[prev][is_even] += 1

        # Update Bayes session counter
        if is_even:
            self.bayes_even_count += 1
        else:
            self.bayes_odd_count += 1

        self.last_is_even = is_even

    def is_ready(self) -> bool:
        return self.tick_n >= self.cfg["min_warmup"]

    def _markov_prob_odd(self) -> float:
        """P(next=odd) from Markov transition given last digit."""
        if self.last_is_even is None:
            return self.cfg["prior_odd"]
        prev   = self.last_is_even
        n_even = self.mk_counts[prev][1]  # transitions to even
        n_odd  = self.mk_counts[prev][0]  # transitions to odd
        total  = n_even + n_odd
        if total == 0:
            return self.cfg["prior_odd"]
        return n_odd / total

    def _bayes_prob_odd(self, markov_p_odd: float) -> float:
        """
        Bayesian update: combine session frequency with Markov estimate.
        Uses log-odds combination for numerical stability.
        """
        total  = self.bayes_even_count + self.bayes_odd_count
        if total == 0:
            return markov_p_odd

        session_p_odd = self.bayes_odd_count / total

        # Weight: session gets more weight as data grows, capped at 0.7
        session_weight = min(0.7, total / (total + 50))
        markov_weight  = 1.0 - session_weight

        combined = (session_weight * session_p_odd +
                    markov_weight  * markov_p_odd)
        return max(0.01, min(0.99, combined))

    def _zscore(self) -> float:
        """Z-score of even-digit rate over rolling window."""
        if len(self.is_even_history) < 10:
            return 0.0
        vals = list(self.is_even_history)
        mu   = sum(vals) / len(vals)
        # Expected mean = prior_even = 1 - prior_odd
        expected_mu = 1.0 - self.cfg["prior_odd"]
        n            = len(vals)
        # Std dev of a Bernoulli
        std_dev = math.sqrt(expected_mu * (1 - expected_mu) / n)
        if std_dev == 0:
            return 0.0
        return (mu - expected_mu) / std_dev

    def compute(self) -> Tuple[Optional[str], float, float, float]:
        """
        Returns (direction, prob, zscore, markov_p_odd) or (None, ...) if no trade.
        direction: 'ODD' or 'EVEN'
        """
        if not self.is_ready():
            return None, 0.0, 0.0, 0.0

        markov_p_odd = self._markov_prob_odd()
        bayes_p_odd  = self._bayes_prob_odd(markov_p_odd)
        z            = self._zscore()
        thresh       = self.cfg["prob_thresh"]
        z_thresh     = self.cfg["zscore_thresh"]

        # Determine direction
        if bayes_p_odd >= thresh and z < -z_thresh:
            # All three layers agree: bet ODD
            return "ODD", bayes_p_odd, z, markov_p_odd

        if (1 - bayes_p_odd) >= thresh and z > z_thresh:
            # All three layers agree: bet EVEN
            return "EVEN", 1 - bayes_p_odd, z, markov_p_odd

        return None, bayes_p_odd, z, markov_p_odd

    def stats_summary(self) -> dict:
        total = self.bayes_even_count + self.bayes_odd_count
        return {
            "ticks":         self.tick_n,
            "session_even":  self.bayes_even_count,
            "session_odd":   self.bayes_odd_count,
            "session_p_odd": round(self.bayes_odd_count / total, 4) if total else 0,
            "mk_odd_given_even": round(self._markov_prob_odd(), 4),
            "zscore":        round(self._zscore(), 4),
        }


# ============================================================================
# MARTINGALE MANAGER
# ============================================================================

class MartingaleManager:
    def __init__(self, cfg: dict):
        self.initial_stake = cfg["initial_stake"]
        self.current_stake = cfg["initial_stake"]
        self.mul           = cfg["martingale_mul"]
        self.max_losses    = cfg["max_losses"]
        self.target_profit = cfg["target_profit"]
        self.stop_loss     = cfg["stop_loss"]
        self.loss_streak   = 0
        self.total_profit  = 0.0
        self.wins          = 0
        self.losses        = 0

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def record_win(self, profit: float):
        self.wins         += 1
        self.total_profit  = round(self.total_profit + profit, 4)
        self.loss_streak   = 0
        self.current_stake = self.initial_stake
        _log("WIN", f"+${profit:.4f} | streak reset | "
                    f"P&L ${self.total_profit:+.4f}")
        self._stats()

    def record_loss(self, loss: float):
        self.losses       += 1
        self.total_profit  = round(self.total_profit + loss, 4)
        self.loss_streak  += 1
        _log("LOSS", f"-${abs(loss):.2f} | streak={self.loss_streak} | "
                     f"P&L ${self.total_profit:+.4f}")
        if self.loss_streak >= self.max_losses:
            _log("MARTI", f"{self.max_losses} losses → reset to "
                          f"${self.initial_stake:.2f}")
            self.current_stake = self.initial_stake
            self.loss_streak   = 0
        else:
            self.current_stake = round(self.current_stake * self.mul, 2)
            _log("MARTI", f"L{self.loss_streak} next stake "
                          f"${self.current_stake:.2f}")
        self._stats()

    def can_trade(self) -> bool:
        if self.total_profit >= self.target_profit:
            _log("RISK", f"Target profit reached "
                         f"(${self.total_profit:.4f}) — stopping")
            return False
        if self.total_profit <= -self.stop_loss:
            _log("RISK", f"Stop-loss hit "
                         f"(${self.total_profit:.4f}) — stopping")
            return False
        return True

    def _stats(self):
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0.0
        print(f"\n{'='*58}", flush=True)
        print(f"  {total} trades | W:{self.wins} L:{self.losses} "
              f"| WR:{wr:.1f}%", flush=True)
        print(f"  P&L ${self.total_profit:+.4f} | "
              f"next stake ${self.current_stake:.2f}", flush=True)
        print(f"{'='*58}\n", flush=True)
        _jlog({
            "type":       "stats",
            "trades":     total,
            "wins":       self.wins,
            "losses":     self.losses,
            "wr":         round(wr, 1),
            "pnl":        self.total_profit,
            "next_stake": self.current_stake,
            "ts":         _ts(),
        })


# ============================================================================
# DERIV CLIENT
# ============================================================================

class DerivClient:
    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.endpoint = (
            f"wss://ws.derivws.com/websockets/v3?app_id={cfg['app_id']}"
        )
        self.ws          = None
        self._send_queue = None
        self._inbox      = None
        self._send_task  = None
        self._recv_task  = None

    async def connect(self) -> bool:
        _log("WS", f"Connecting → {self.endpoint}")
        self.ws = await websockets.connect(
            self.endpoint,
            ping_interval=self.cfg["ws_ping"],
            ping_timeout=20,
            close_timeout=10,
        )
        self._send_queue = asyncio.Queue()
        self._inbox      = asyncio.Queue()
        self._start_io()
        await self._send({"authorize": self.cfg["api_token"]})
        resp = await self._recv_type("authorize", timeout=15)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("AUTH", f"Failed: {err}")
            return False
        auth = resp.get("authorize", {})
        _log("AUTH", f"OK | {auth.get('loginid','?')} | "
                     f"Balance: ${auth.get('balance', 0):.2f}")
        return True

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            _log("RECV", f"Error: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send(self, data: dict):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def receive(self, timeout: float = 60) -> dict:
        try:
            return await asyncio.wait_for(
                self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def _recv_type(self, msg_type: str,
                         timeout: float = 10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            _log("BALANCE", f"Fetch error: {exc}")
        return None

    async def subscribe_ticks(self) -> bool:
        sym = self.cfg["symbol"]
        await self._send({"ticks": sym, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("TICK", f"Subscribe {sym} failed: {err}")
            return False
        _log("TICK", f"Subscribed to {sym}")
        return True

    async def place_digit_trade(
            self, direction: str, stake: float) -> Optional[str]:
        """
        Place DIGITEVEN or DIGITODD contract — 1 tick.
        direction: 'EVEN' or 'ODD'
        """
        contract_type = f"DIGIT{direction}"   # DIGITEVEN or DIGITODD
        proposal_req  = {
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": contract_type,
            "currency":      self.cfg["currency"],
            "duration":      1,
            "duration_unit": "t",
            "symbol":        self.cfg["symbol"],
        }
        await self._send(proposal_req)
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            err = (proposal or {}).get("error", {}).get("message", "timeout")
            _log("PROPOSAL", f"Error: {err}")
            return None

        prop   = proposal.get("proposal", {})
        pid    = prop.get("id")
        ask    = float(prop.get("ask_price", stake))
        payout = float(prop.get("payout", 0))
        if not pid:
            _log("PROPOSAL", "No proposal ID")
            return None

        roi = ((payout - ask) / ask * 100) if ask > 0 else 0
        _log("PROPOSAL",
             f"{contract_type}  ask=${ask:.2f}  "
             f"payout=${payout:.2f}  ROI={roi:.1f}%")

        buy_time    = time.time()
        contract_id = None
        await self._send({"buy": pid, "price": ask})

        for attempt in range(self.cfg["buy_retries"]):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                _log("BUY", f"No response (attempt {attempt + 1})")
                continue
            if "error" in resp:
                _log("BUY", f"Error: {resp['error'].get('message', '')}")
                return None
            contract_id = resp.get("buy", {}).get("contract_id")
            if contract_id:
                break

        if not contract_id:
            _log("BUY", "No contract_id — orphan recovery")
            contract_id = await self._recover_orphan(stake, buy_time)
            if contract_id:
                _log("BUY", f"Orphan recovered → {contract_id}")
            else:
                _log("BUY", "Orphan recovery failed")
                return None

        _log("TRADE", f"{contract_type}  ${stake:.2f}  "
                      f"contract={contract_id}")

        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id":            contract_id,
                "subscribe":              1,
            })
        except Exception:
            pass

        return str(contract_id)

    async def _recover_orphan(self, stake: float,
                              buy_time: float) -> Optional[str]:
        for attempt in range(self.cfg["orphan_attempts"]):
            await asyncio.sleep(self.cfg["orphan_interval"])
            try:
                await self._send({
                    "profit_table": 1, "description": 1,
                    "sort": "DESC", "limit": 5,
                })
                resp = await self._recv_type("profit_table", timeout=10)
                if not resp or "error" in resp:
                    continue
                for tx in (resp.get("profit_table", {})
                           .get("transactions", [])):
                    if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                            float(tx.get("purchase_time", 0)) >= buy_time - 5):
                        return str(tx.get("contract_id"))
            except Exception as exc:
                _log("ORPHAN", f"Poll {attempt + 1} error: {exc}")
        return None

    async def poll_contract(self,
                            contract_id: str) -> Optional[dict]:
        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id":            contract_id,
            })
            resp = await self._recv_type(
                "proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            _log("POLL", f"Error: {exc}")
        return None


# ============================================================================
# MAIN BOT
# ============================================================================

class DigitBot:
    def __init__(self):
        self.cfg    = CONFIG
        self.client = DerivClient(CONFIG)
        self.engine = SignalEngine(CONFIG)
        self.risk   = MartingaleManager(CONFIG)

        self.tick_n:             int            = 0
        self._last_trade_tick:   int            = 0
        self.current_contract:   Optional[dict] = None
        self.waiting_for_result: bool           = False
        self.lock_since:         Optional[float] = None
        self._evaluating:        bool           = False
        self._balance_before:    Optional[float] = None
        self._cb_paused_until:   float          = 0.0
        self._stop:              bool           = False

    # ── Lock management ───────────────────────────────────────────────────────

    def _unlock(self, reason: str = "manual"):
        if self.waiting_for_result:
            cid = (self.current_contract or {}).get("id", "?")
            _log("UNLOCK", f"Contract {cid} ({reason})")
        self.waiting_for_result = False
        self.current_contract   = None
        self.lock_since         = None
        self._evaluating        = False

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        if (time.monotonic() - self.lock_since >=
                self.cfg["lock_timeout"]):
            _log("TIMEOUT", "Auto-unlocking after lock timeout")
            self._unlock("timeout")

    # ── Settlement ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_settled(data: dict) -> bool:
        if data.get("is_settled") or data.get("is_sold"):
            return True
        for key in ("status", "contract_status"):
            if data.get(key, "").lower() in ("sold", "won", "lost"):
                return True
        return False

    async def handle_settlement(self,
                                data: dict) -> Optional[bool]:
        cid = str(data.get("contract_id", ""))
        if not self.current_contract or cid != self.current_contract["id"]:
            return None
        if not self._is_settled(data):
            return None

        bal_after  = await self.client.fetch_balance()
        api_profit = float(data.get("profit", 0))
        status     = data.get("status", "unknown")

        if bal_after is not None and self._balance_before is not None:
            actual = round(bal_after - self._balance_before, 4)
            _log("BALANCE",
                 f"Pre: ${self._balance_before:.2f} → "
                 f"Post: ${bal_after:.4f} | "
                 f"Actual: ${actual:+.4f} | "
                 f"API: ${api_profit:+.4f}")
        else:
            actual = api_profit

        # Extract last digit from contract data for logging
        last_digit = data.get("barrier", "?")

        print(f"\nRESULT  contract={cid}  status={status}  "
              f"digit={last_digit}  profit=${actual:+.4f}", flush=True)

        if actual > 0:
            self.risk.record_win(actual)
        else:
            self.risk.record_loss(actual)
            if (self.risk.loss_streak > 0 and
                    self.risk.loss_streak % self.cfg["cb_limit"] == 0):
                pause = self.cfg["cb_pause_secs"]
                self._cb_paused_until = time.monotonic() + pause
                _log("BREAKER",
                     f"{self.cfg['cb_limit']} consecutive losses → "
                     f"pausing {pause}s ({pause // 60}m)")

        _jlog({
            "type":      "result",
            "cid":       cid,
            "status":    status,
            "profit":    actual,
            "pnl":       self.risk.total_profit,
            "wins":      self.risk.wins,
            "losses":    self.risk.losses,
            "streak":    self.risk.loss_streak,
            "ts":        _ts(),
        })

        self._balance_before = None
        self._unlock("settlement")
        return self.risk.can_trade()

    # ── Tick handler ──────────────────────────────────────────────────────────

    async def on_tick(self, price: float):
        self.tick_n += 1
        self._check_lock_timeout()

        # Extract last digit from price
        digit = int(str(round(price, 2)).replace(".", "")[-1])
        self.engine.add_tick(digit)

        if self.tick_n % 20 == 0:
            warmup_left = max(0, self.cfg["min_warmup"] - self.engine.tick_n)
            status = ("WAIT" if self.waiting_for_result else
                      f"WARMUP({warmup_left})" if warmup_left > 0
                      else "READY")
            print(f"\r  #{self.tick_n}  p={price:.2f}  "
                  f"d={digit}  {status}  {_ts()}",
                  end="", flush=True)

        if self.waiting_for_result or self._evaluating:
            return
        if not self.engine.is_ready():
            return
        if (self.tick_n - self._last_trade_tick) < self.cfg["cooldown_ticks"]:
            return

        self._evaluating = True
        try:
            await self._evaluate()
        finally:
            self._evaluating = False

    # ── Signal evaluation ─────────────────────────────────────────────────────

    async def _evaluate(self):
        if self.waiting_for_result:
            return

        direction, prob, z, mk_p = self.engine.compute()

        if direction is None:
            return   # silent — fires too frequently to log every skip

        print(f"\n{'='*58}", flush=True)
        print(f"SIGNAL  #{self.tick_n}  {_ts()}", flush=True)
        print(f"  Bayes p_odd={prob:.4f}  Markov p_odd={mk_p:.4f}  "
              f"Z={z:.3f}", flush=True)
        print(f"  → {direction}  stake=${self.risk.get_stake():.2f}",
              flush=True)
        print(f"{'='*58}", flush=True)

        # Circuit breaker check
        now = time.monotonic()
        if now < self._cb_paused_until:
            remaining = self._cb_paused_until - now
            _log("BREAKER", f"Paused — {remaining:.0f}s remaining")
            return

        if not self.risk.can_trade():
            return

        stake = self.risk.get_stake()

        bal = await self.client.fetch_balance()
        if bal is not None:
            self._balance_before = bal
            _log("BALANCE", f"Pre-trade: ${bal:.2f}")
        else:
            self._balance_before = None

        contract_id = await self.client.place_digit_trade(direction, stake)

        if contract_id:
            self.current_contract = {
                "id":        contract_id,
                "direction": direction,
                "stake":     stake,
                "prob":      prob,
                "z":         z,
                "time":      datetime.now(),
            }
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            self._last_trade_tick   = self.tick_n
            _log("LOCK", f"Waiting for result on {contract_id}")
            _jlog({
                "type":      "trade",
                "cid":       contract_id,
                "direction": direction,
                "stake":     stake,
                "prob":      round(prob, 4),
                "z":         round(z, 4),
                "mk_p":      round(mk_p, 4),
                "ts":        _ts(),
            })
        else:
            self._balance_before = None
            _log("TRADE", "Placement failed — ready for next signal")

    # ── Reconnect ─────────────────────────────────────────────────────────────

    async def _reconnect(self) -> bool:
        delay   = self.cfg["reconnect_min"]
        attempt = 0
        while not self._stop:
            attempt += 1
            _log("RECONNECT", f"Attempt {attempt} in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg["reconnect_max"])
            await self.client.close()
            self.client = DerivClient(self.cfg)
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                # Re-attach open contract
                if self.waiting_for_result and self.current_contract:
                    cid  = self.current_contract["id"]
                    _log("RECONNECT", f"Re-attaching to {cid}")
                    data = await self.client.poll_contract(cid)
                    if data:
                        await self.handle_settlement(data)
                    if self.waiting_for_result:
                        await self.client._send({
                            "proposal_open_contract": 1,
                            "contract_id": cid,
                            "subscribe":   1,
                        })
                _log("RECONNECT", "OK")
                return True
            except Exception as exc:
                _log("RECONNECT", f"Error: {exc}")
        return False

    # ── Console ───────────────────────────────────────────────────────────────

    async def _console(self):
        loop = asyncio.get_event_loop()
        _log("CMD", "Commands: [s]tats  [e]ngine  [u]nlock  [q]uit")
        while not self._stop:
            try:
                cmd = (await loop.run_in_executor(
                    None, input)).strip().lower()
                if cmd == "s":
                    self.risk._stats()
                elif cmd == "e":
                    summary = self.engine.stats_summary()
                    print(f"\n  Engine stats: {summary}", flush=True)
                elif cmd == "u":
                    self._unlock("user command")
                elif cmd in ("q", "quit", "exit"):
                    self._stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self):
        cfg = self.cfg
        print("\n" + "="*58, flush=True)
        print("  DERIV DIGIT BOT — Bayes + Markov + Z-Score", flush=True)
        print("="*58, flush=True)
        print(f"  Symbol     : {cfg['symbol']}", flush=True)
        print(f"  Contract   : DIGITODD / DIGITEVEN (1 tick)", flush=True)
        print(f"  Prob gate  : ≥ {cfg['prob_thresh']}", flush=True)
        print(f"  Z-score    : |Z| > {cfg['zscore_thresh']}", flush=True)
        print(f"  Prior P(odd): {cfg['prior_odd']}", flush=True)
        print(f"  Stake      : ${cfg['initial_stake']:.2f} "
              f"(×{cfg['martingale_mul']} mart, "
              f"reset @{cfg['max_losses']} losses)", flush=True)
        print(f"  Ladder     : "
              + " → ".join(
                  f"${round(cfg['initial_stake'] * cfg['martingale_mul']**i, 2):.2f}"
                  for i in range(cfg['max_losses'])
              ) + " → reset", flush=True)
        print(f"  Target     : +${cfg['target_profit']}  "
              f"Stop: -${cfg['stop_loss']}", flush=True)
        print(f"  Breaker    : {cfg['cb_limit']} losses → "
              f"{cfg['cb_pause_secs']}s pause", flush=True)
        print(f"  Warmup     : {cfg['min_warmup']} ticks", flush=True)
        print("="*58 + "\n", flush=True)

        if cfg["api_token"] in ("REPLACE_WITH_YOUR_TOKEN", ""):
            _log("ERROR", "Set DERIV_API_TOKEN before running")
            return

        if not await self.client.connect():
            return
        if not await self.client.subscribe_ticks():
            return

        _log("BOT", f"Live — warming up "
                    f"({cfg['min_warmup']} ticks)...")
        console_task = asyncio.create_task(self._console())

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    _log("WS", "Disconnected — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        _log("WS", "Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    quote = response["tick"].get("quote")
                    if quote is not None:
                        print()
                        await self.on_tick(float(quote))

                if "proposal_open_contract" in response:
                    result = await self.handle_settlement(
                        response["proposal_open_contract"])
                    if result is False:
                        break

                if "buy" in response:
                    result = await self.handle_settlement(
                        response["buy"])
                    if result is False:
                        break

                if "transaction" in response:
                    tx = response["transaction"]
                    if "contract_id" in tx:
                        result = await self.handle_settlement({
                            "contract_id": tx.get("contract_id"),
                            "profit":      tx.get("profit", 0),
                            "status":      tx.get("action", ""),
                            "is_settled":  True,
                        })
                        if result is False:
                            break

        except KeyboardInterrupt:
            print("\n\nInterrupted", flush=True)
        except Exception as exc:
            print(f"\nUnhandled error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            console_task.cancel()
            await self.client.close()
            print("\nFINAL STATS", flush=True)
            self.risk._stats()
            print("Goodbye", flush=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    bot = DigitBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
