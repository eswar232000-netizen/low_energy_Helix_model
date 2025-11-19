#!/usr/bin/env python3
"""
HelixDAG Real-time Simulator — Full Regenerated Version

Save as: helixdag_realtime_sim_full.py
Run: python helixdag_realtime_sim_full.py
"""

import threading
import time
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------------- CONFIG ----------------
NUM_MINERS = 5
SIM_DURATION = 20.0            # seconds (wall-clock)
TARGET_INTERVAL = 6.0          # seconds
ALPHA = 0.2                    # EWMA smoothing
INITIAL_DIFFICULTY = 2 ** 22
MIN_DIFFICULTY = 2 ** 20
ENERGY_PER_HASH = 5e-9         # joules per hash
BLAKE_OUT_BITS = 256
GAMMA = 2.0                    # CMDA strength
TX_PER_SEC_PER_NODE = 2
PRINT_PROGRESS = True

# ---------------- DATA MODELS ----------------
@dataclass
class Transaction:
    txid: str
    sender: str
    receiver: str
    amount: float
    timestamp: float

@dataclass
class Block:
    height: int
    prev_hash: str
    miner_id: str
    timestamp: float
    nonce: int
    txs: List[Transaction]
    difficulty: int
    block_hash: Optional[str] = None

@dataclass
class MinerStats:
    mined_blocks: int = 0
    hashes: int = 0
    energy_joules: float = 0.0
    last_block_time: Optional[float] = None
    ewma_interval: float = TARGET_INTERVAL
    local_difficulty: int = INITIAL_DIFFICULTY

# ---------------- UTILS ----------------
def blake2b_hash(b: bytes) -> bytes:
    return hashlib.blake2b(b, digest_size=BLAKE_OUT_BITS // 8).digest()

def header_to_bytes(h, p, m, t, n) -> bytes:
    return f"{h}:{p}:{m}:{t:.6f}:{n}".encode()

def hash_meets_target(digest: bytes, difficulty: int) -> bool:
    value = int.from_bytes(digest, 'big')
    max_int = 2 ** BLAKE_OUT_BITS - 1
    target = max_int // difficulty
    return value <= target

def now() -> float:
    return time.time()

# ---------------- MINER THREAD ----------------
class Miner(threading.Thread):
    def __init__(self, miner_id: str, global_state: dict, mode: str = 'cmda'):
        super().__init__()
        self.miner_id = miner_id
        self.global_state = global_state
        self.mode = mode  # 'cmda', 'fixed', 'lowcmda'
        self.stats = MinerStats()
        self.stop_event = threading.Event()
        self.tx_pool: List[Transaction] = []
        self.random = random.Random(hash(miner_id) & 0xffffffff)

    def run(self):
        threading.Thread(target=self._generate_transactions_loop, daemon=True).start()

        while not self.stop_event.is_set() and not self.global_state['stop_all']:
            # get tip & height
            with self.global_state['lock']:
                height = self.global_state['height']
                prev_hash = self.global_state['tip_hash']

            txs = self._grab_transactions()
            difficulty = self._current_difficulty()

            nonce = 0
            found = False

            while not found and not self.stop_event.is_set() and not self.global_state['stop_all']:
                header = header_to_bytes(height + 1, prev_hash, self.miner_id, now(), nonce)
                digest = blake2b_hash(header)
                self.stats.hashes += 1

                if hash_meets_target(digest, difficulty):
                    # found block
                    found = True
                    timestamp = now()
                    block = Block(height + 1, prev_hash, self.miner_id,
                                  timestamp, nonce, txs, difficulty, digest.hex())
                    self._broadcast_block(block)
                    self.stats.mined_blocks += 1

                    # CMDA: predictive EWMA (level + trend) -> multiplicative update
                    if self.stats.last_block_time is not None:
                        observed = timestamp - self.stats.last_block_time
                        prev_level = self.stats.ewma_interval
                        level = (1 - ALPHA) * prev_level + ALPHA * observed
                        self.stats.ewma_interval = level

                        trend_prev = getattr(self.stats, 'trend', 0.0)
                        beta = 0.1
                        trend = beta * (level - prev_level) + (1 - beta) * trend_prev
                        self.stats.trend = trend

                        predicted_interval = level + trend
                        ratio = max(0.1, predicted_interval / TARGET_INTERVAL)

                        if self.mode == 'lowcmda':
                            new_diff = int(self.stats.local_difficulty * pow(ratio, GAMMA * 1.5))
                        else:
                            new_diff = int(self.stats.local_difficulty * pow(ratio, GAMMA))

                        self.stats.local_difficulty = max(MIN_DIFFICULTY, new_diff)

                    self.stats.last_block_time = timestamp
                    break

                # throttle to reduce CPU
                nonce += 1
                if self.mode == 'lowcmda' and nonce % 500 == 0:
                    time.sleep(0.002)
                if nonce % 1000 == 0:
                    time.sleep(0)

            # update energy estimate
            self.stats.energy_joules = self.stats.hashes * ENERGY_PER_HASH
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()

    def _current_difficulty(self) -> int:
        if self.mode in ('cmda', 'lowcmda'):
            return self.stats.local_difficulty
        return INITIAL_DIFFICULTY

    def _broadcast_block(self, block: Block):
        with self.global_state['lock']:
            if block.height > self.global_state['height']:
                self.global_state['height'] = block.height
                self.global_state['tip_hash'] = block.block_hash
                self.global_state['chain'].append(block)
                self.global_state['accepted_blocks'].append(block)
                self.global_state['miner_map'][block.block_hash] = block.miner_id
                if PRINT_PROGRESS:
                    print(f"[ACCEPT] {block.miner_id} mined block {block.height} diff={block.difficulty}")
            else:
                self.global_state['orphaned_blocks'].append(block)
                self.global_state['miner_map'][block.block_hash] = block.miner_id
                if PRINT_PROGRESS:
                    print(f"[ORPHAN] {block.miner_id} mined orphan block {block.height}")

    def _generate_transactions_loop(self):
        while not self.stop_event.is_set() and not self.global_state['stop_all']:
            for _ in range(TX_PER_SEC_PER_NODE):
                tx = Transaction(
                    txid=f"{self.miner_id}-{time.time_ns()}-{self.random.randint(0,1<<30)}",
                    sender=self.miner_id,
                    receiver=f"N{self.random.randint(0, NUM_MINERS-1)}",
                    amount=round(self.random.random()*10, 4),
                    timestamp=now(),
                )
                self.tx_pool.append(tx)
            time.sleep(1)

    def _grab_transactions(self, max_txs: int = 20) -> List[Transaction]:
        txs: List[Transaction] = []
        while self.tx_pool and len(txs) < max_txs:
            txs.append(self.tx_pool.pop(0))
        return txs


# -------------- EXPERIMENT ----------------
def run_experiment(mode: str, record_time_series: bool = True) -> Dict:
    global_state = {
        'lock': threading.Lock(),
        'height': 0,
        'tip_hash': '0' * 64,
        'chain': [],
        'accepted_blocks': [],
        'orphaned_blocks': [],
        'miner_map': {},
        'stop_all': False
    }

    miners: List[Miner] = [Miner(f"M{i}", global_state, mode) for i in range(NUM_MINERS)]
    for m in miners:
        m.stats.local_difficulty = INITIAL_DIFFICULTY

    for m in miners:
        m.start()

    start_time = now()

    diff_series: Dict[str, List[int]] = {m.miner_id: [] for m in miners}
    hash_series: Dict[str, List[int]] = {m.miner_id: [] for m in miners}
    energy_series: Dict[str, List[float]] = {m.miner_id: [] for m in miners}
    timestamp_series: List[float] = []

    try:
        while now() - start_time < SIM_DURATION:
            time.sleep(1.0)
            if record_time_series:
                t = now() - start_time
                timestamp_series.append(t)
                for m in miners:
                    diff_series[m.miner_id].append(m.stats.local_difficulty)
                    hash_series[m.miner_id].append(m.stats.hashes)
                    energy_series[m.miner_id].append(m.stats.energy_joules)
    except KeyboardInterrupt:
        pass

    # stop miners cleanly
    global_state['stop_all'] = True
    for m in miners:
        m.stop()
    for m in miners:
        m.join()

    # collect metrics
    total_hashes = sum(m.stats.hashes for m in miners)
    total_energy = sum(m.stats.energy_joules for m in miners)
    total_blocks = sum(m.stats.mined_blocks for m in miners)
    accepted = len(global_state['accepted_blocks'])
    orphaned = len(global_state['orphaned_blocks'])

    per_miner = {
        m.miner_id: {
            'blocks': m.stats.mined_blocks,
            'hashes': m.stats.hashes,
            'energy': m.stats.energy_joules,
            'final_diff': m.stats.local_difficulty
        } for m in miners
    }

    return {
        'mode': mode,
        'total_hashes': total_hashes,
        'total_energy': total_energy,
        'total_blocks': total_blocks,
        'accepted_blocks': accepted,
        'orphaned_blocks': orphaned,
        'per_miner': per_miner,
        'diff_series': diff_series,
        'hash_series': hash_series,
        'energy_series': energy_series,
        'timestamp_series': timestamp_series,
        'chain': global_state['chain']
    }


# ------------------ PLOTTING HELPERS ----------------
def align_series(ts: List[float], series_dict: Dict[str, List]) -> Dict[str, List]:
    """Trim or pad miner series so lengths match timestamp length."""
    target_len = len(ts)
    aligned: Dict[str, List] = {}
    for mid, s in series_dict.items():
        if len(s) > target_len:
            aligned[mid] = s[:target_len]
        elif len(s) < target_len:
            # pad with last observed value (or zero if empty)
            pad_val = s[-1] if len(s) > 0 else 0
            aligned[mid] = s + [pad_val] * (target_len - len(s))
        else:
            aligned[mid] = s
    return aligned

def plot_difficulty(metrics_list: List[Dict], labels: List[str]):
    ts = metrics_list[0]['timestamp_series']
    plt.figure(figsize=(12,6))
    for metrics, label in zip(metrics_list, labels):
        aligned = align_series(ts, metrics['diff_series'])
        for mid, series in aligned.items():
            plt.plot(ts, series, label=f"{label}-{mid}")
    plt.title("Difficulty Evolution per Miner")
    plt.xlabel("Time (s)")
    plt.ylabel("Difficulty")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("difficulty_evolution.png")
    plt.show()
    plt.close()
    print("Saved difficulty_evolution.png")

def plot_hashrate(metrics_list: List[Dict], labels: List[str]):
    ts = metrics_list[0]['timestamp_series']
    plt.figure(figsize=(12,6))
    for metrics, label in zip(metrics_list, labels):
        aligned = align_series(ts, metrics['hash_series'])
        for mid, series in aligned.items():
            hps = [0] + [series[i] - series[i-1] for i in range(1, len(series))]
            plt.plot(ts, hps, label=f"{label}-{mid}")
    plt.title("Hashes/sec per Miner")
    plt.xlabel("Time (s)")
    plt.ylabel("Hashes/sec")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("hashrate_evolution.png")
    plt.show()
    plt.close()
    print("Saved hashrate_evolution.png")

def plot_energy(metrics_list: List[Dict], labels: List[str]):
    ts = metrics_list[0]['timestamp_series']
    plt.figure(figsize=(12,6))
    for metrics, label in zip(metrics_list, labels):
        aligned = align_series(ts, metrics['energy_series'])
        for mid, series in aligned.items():
            plt.plot(ts, series, label=f"{label}-{mid}")
    plt.title("Energy Consumption Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("energy_over_time.png")
    plt.show()
    plt.close()
    print("Saved energy_over_time.png")

def save_energy_table(cmda: Dict, fixed: Dict, low: Dict):
    with open("energy_comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Mode", "Total Energy (J)", "Total Hashes", "Accepted Blocks", "Orphans", "Avg Difficulty"])

        def avgdiff(m):
            vals = [d['final_diff'] for d in m['per_miner'].values()]
            return float(np.mean(vals)) if vals else 0.0

        w.writerow(["CMDA", cmda['total_energy'], cmda['total_hashes'], cmda['accepted_blocks'], cmda['orphaned_blocks'], avgdiff(cmda)])
        w.writerow(["Fixed", fixed['total_energy'], fixed['total_hashes'], fixed['accepted_blocks'], fixed['orphaned_blocks'], avgdiff(fixed)])
        w.writerow(["LowCMDA", low['total_energy'], low['total_hashes'], low['accepted_blocks'], low['orphaned_blocks'], avgdiff(low)])
    print("Saved energy_comparison_table.csv")

# ------------------ MAIN DRIVER ----------------
def main():
    print("=== HelixDAG Real-time Simulator ===")
    print(f"NUM_MINERS={NUM_MINERS}, SIM_DURATION={SIM_DURATION}s, TARGET_INTERVAL={TARGET_INTERVAL}s")

    print("Running CMDA mode...")
    metrics_cmda = run_experiment("cmda")

    print("Running Fixed mode...")
    metrics_fixed = run_experiment("fixed")

    print("Running Low-Energy CMDA mode...")
    metrics_low = run_experiment("lowcmda")

    metrics_list = [metrics_cmda, metrics_fixed, metrics_low]
    labels = ["CMDA", "Fixed", "LowCMDA"]

    plot_difficulty(metrics_list, labels)
    plot_hashrate(metrics_list, labels)
    plot_energy(metrics_list, labels)
    save_energy_table(metrics_cmda, metrics_fixed, metrics_low)

    print("Simulation complete — generated files:")
    print(" - difficulty_evolution.png")
    print(" - hashrate_evolution.png")
    print(" - energy_over_time.png")
    print(" - energy_comparison_table.csv")

if __name__ == "__main__":
    main()
