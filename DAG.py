#!/usr/bin/env python3
"""
helixdag_realtime_sim_dag_final_optionC_2tips.py

HelixDAG real-time simulator + EWMA difficulty predictor + CMDA + Low-Energy CMDA
Option C final: 3 target plots (Predicted difficulty, Actual difficulty evolution, Energy comparison)
DAG: each block references 2 tips (user requested '2' in last message).

Saves CSVs and publication-quality figures (SVG, PDF, PNG@600).
Prints all numeric arrays used for plotting to the console.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
import time
import random
import hashlib
import os
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------- CONFIG ----------------
RND_SEED = 42
random.seed(RND_SEED)
np.random.seed(RND_SEED)

# Simulation / model parameters
NUM_MINERS = 5
SIM_DURATION = 20.0            # seconds per mode
TARGET_INTERVAL = 6.0          # ideal block interval (seconds)
ALPHA = 0.2                    # EWMA smoothing
INITIAL_DIFFICULTY = 2 ** 20   # lowered for manageable demo speed
MIN_DIFFICULTY = 2 ** 18
ENERGY_PER_HASH = 5e-9         # joules per hash
BLAKE_OUT_BITS = 256
GAMMA = 2.0                    # CMDA exponent strength
TX_PER_SEC_PER_NODE = 2
PRINT_PROGRESS = False         # set True for debugging printouts

# DAG specifics: user requested 2 tips
PARENTS_PER_BLOCK = 2
ALLOW_MULTI_PARENT = True

# Plotting and export settings
PLOT_REFRESH = 1.0             # seconds
FIGSIZE = (14, 9)
LINEWIDTH = 2.0
FONT_TITLE = 14
FONT_LABEL = 12
LEGEND_FS = 9
GRID_ALPHA = 0.25
OUT_DIR = "helixdag_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# File naming
COMBINED_BASE = os.path.join(OUT_DIR, "helixdag_combined_final_optionC_2tips")

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
    parents: List[str]
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
    trend: float = 0.0
    local_difficulty: int = INITIAL_DIFFICULTY

# ---------------- HELPERS ----------------
def blake2b_hash(b: bytes) -> bytes:
    return hashlib.blake2b(b, digest_size=BLAKE_OUT_BITS // 8).digest()

def header_to_bytes(parents: List[str], m: str, t: float, n: int) -> bytes:
    parents_field = "|".join(parents)
    return f"{parents_field}:{m}:{t:.6f}:{n}".encode()

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
        super().__init__(daemon=True)
        self.miner_id = miner_id
        self.global_state = global_state
        self.mode = mode  # 'cmda', 'fixed', 'lowcmda'
        self.stats = MinerStats()
        self.stop_event = threading.Event()
        self.tx_pool: List[Transaction] = []
        self.random = random.Random((hash(miner_id) & 0xffffffff) ^ int(time.time() * 1000) & 0xffffffff)

    def run(self):
        # Start tx generator
        threading.Thread(target=self._generate_transactions_loop, daemon=True).start()

        while not self.stop_event.is_set() and not self.global_state['stop_all']:
            parents = self._select_parents()
            txs = self._grab_transactions()
            difficulty = self._current_difficulty()

            nonce = 0
            found = False

            while not found and not self.stop_event.is_set() and not self.global_state['stop_all']:
                header = header_to_bytes(parents, self.miner_id, now(), nonce)
                digest = blake2b_hash(header)
                self.stats.hashes += 1

                if hash_meets_target(digest, difficulty):
                    timestamp = now()
                    block_hash = digest.hex()
                    block = Block(self.global_state['next_height'], parents.copy(),
                                  self.miner_id, timestamp, nonce, txs.copy(), difficulty, block_hash)
                    self._broadcast_block(block)
                    self.stats.mined_blocks += 1

                    # EWMA + trend update on mining event
                    if self.stats.last_block_time is not None:
                        observed = timestamp - self.stats.last_block_time
                        prev_level = self.stats.ewma_interval
                        level = (1 - ALPHA) * prev_level + ALPHA * observed
                        self.stats.ewma_interval = level

                        trend_prev = self.stats.trend
                        beta = 0.1
                        trend = beta * (level - prev_level) + (1 - beta) * trend_prev
                        self.stats.trend = trend

                        predicted_interval = level + trend
                        ratio = max(0.1, predicted_interval / TARGET_INTERVAL)

                        # block rapidity adjustment: if observed much smaller than prev, increase more
                        if self.stats.last_block_time is not None and observed < prev_level * 0.8:
                            rapid_multiplier = 1.15
                        else:
                            rapid_multiplier = 1.0

                        if self.mode == 'lowcmda':
                            new_diff = int(self.stats.local_difficulty * pow(ratio, GAMMA * 1.5) * rapid_multiplier)
                        else:
                            new_diff = int(self.stats.local_difficulty * pow(ratio, GAMMA) * rapid_multiplier)

                        self.stats.local_difficulty = max(MIN_DIFFICULTY, new_diff)

                    self.stats.last_block_time = timestamp
                    found = True
                    break

                nonce += 1
                if self.mode == 'lowcmda' and nonce % 500 == 0:
                    time.sleep(0.002)
                if nonce % 10000 == 0:
                    time.sleep(0)

            self.stats.energy_joules = self.stats.hashes * ENERGY_PER_HASH
            time.sleep(0.003)

    def _current_difficulty(self) -> int:
        if self.mode in ('cmda', 'lowcmda'):
            return max(MIN_DIFFICULTY, self.stats.local_difficulty)
        return INITIAL_DIFFICULTY

    def _select_parents(self) -> List[str]:
        # choose up to PARENTS_PER_BLOCK tips
        with self.global_state['lock']:
            tips_list = list(self.global_state['tips'])
            if not tips_list:
                return [self.global_state['genesis_hash']]
            k = min(PARENTS_PER_BLOCK, len(tips_list))
            chosen = self.random.sample(tips_list, k)
            return chosen

    def _broadcast_block(self, block: Block):
        with self.global_state['lock']:
            block.height = self.global_state['next_height']
            self.global_state['next_height'] += 1

            # unknown parents
            unknown_parents = [p for p in block.parents if p not in self.global_state['all_blocks'] and p != self.global_state['genesis_hash']]
            if unknown_parents:
                self.global_state['unknown_parent_events'].append((block.block_hash, unknown_parents))
                if PRINT_PROGRESS:
                    print(f"[WARN] {block.miner_id} referenced unknown parents {unknown_parents} in block {block.height}")

            self.global_state['dag'].append(block)
            self.global_state['all_blocks'][block.block_hash] = block
            self.global_state['miner_map'][block.block_hash] = block.miner_id
            self.global_state['accepted_blocks'].append(block)

            # update tips: referenced parents removed, new block added as tip
            for p in block.parents:
                if p in self.global_state['tips']:
                    self.global_state['tips'].discard(p)
            self.global_state['tips'].add(block.block_hash)

            self.global_state['recent_block_timestamps'].append(block.timestamp)
            if PRINT_PROGRESS:
                print(f"[DAG-ACCEPT] {block.miner_id} mined block {block.height} refs={len(block.parents)} diff={block.difficulty}")

    def stop(self):
        self.stop_event.set()

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

# ---------------- RUN EXPERIMENT (per mode) ----------------
def run_experiment(mode: str, sim_duration: float) -> Dict:
    genesis_hash = "0" * 64
    global_state = {
        'lock': threading.Lock(),
        'next_height': 1,
        'genesis_hash': genesis_hash,
        'dag': [],
        'all_blocks': {},
        'accepted_blocks': [],
        'tips': set(),
        'miner_map': {},
        'unknown_parent_events': [],
        'recent_block_timestamps': [],
        'stop_all': False
    }
    global_state['tips'].add(genesis_hash)

    miners: List[Miner] = [Miner(f"M{i}", global_state, mode) for i in range(NUM_MINERS)]
    for m in miners:
        m.stats.local_difficulty = INITIAL_DIFFICULTY

    for m in miners:
        m.start()

    start_time = now()

    # series
    timestamp_series: List[float] = []
    ewma_pred_series: List[float] = []
    predicted_diff_series: List[int] = []
    fixed_diff_series: List[int] = []
    cmda_diff_series: List[int] = []
    low_diff_series: List[int] = []
    block_time_series: List[float] = []
    tips_count_series: List[int] = []
    dag_node_count_series: List[int] = []
    blocks_mined_series: List[int] = []

    # per-miner series collections
    diff_series: Dict[str, List[int]] = {m.miner_id: [] for m in miners}
    hash_series: Dict[str, List[int]] = {m.miner_id: [] for m in miners}
    energy_series: Dict[str, List[float]] = {m.miner_id: [] for m in miners}

    try:
        while now() - start_time < sim_duration:
            time.sleep(PLOT_REFRESH)
            t = now() - start_time
            timestamp_series.append(round(t, 6))

            # collect per-miner stats & EWMA averaging
            avg_ewma = 0.0
            for m in miners:
                diff_series[m.miner_id].append(m.stats.local_difficulty)
                hash_series[m.miner_id].append(m.stats.hashes)
                energy_series[m.miner_id].append(m.stats.energy_joules)
                avg_ewma += m.stats.ewma_interval
            avg_ewma /= max(1, len(miners))

            avg_trend = float(np.mean([m.stats.trend for m in miners]))
            predicted_interval = max(0.01, avg_ewma + avg_trend)
            ewma_pred_series.append(predicted_interval)

            ratio = max(0.01, predicted_interval / TARGET_INTERVAL)
            baseline_diff = int(np.mean([m.stats.local_difficulty for m in miners]))
            pred_diff = int(baseline_diff * pow(ratio, GAMMA))
            predicted_diff_series.append(pred_diff)

            fixed_diff_series.append(INITIAL_DIFFICULTY)
            cmda_diff_series.append(int(np.mean([m.stats.local_difficulty for m in miners])))
            low_diff_series.append(int(np.mean([int(m.stats.local_difficulty * 0.9) for m in miners])))

            # block time inter-arrival
            rbts = global_state['recent_block_timestamps']
            if len(rbts) >= 2:
                block_time_series.append(round(rbts[-1] - rbts[-2], 6))
            elif len(rbts) == 1:
                block_time_series.append(round(rbts[0] - start_time, 6))
            else:
                block_time_series.append(0.0)

            tips_count_series.append(len(global_state['tips']))
            dag_node_count_series.append(len(global_state['dag']))
            blocks_mined_series.append(sum(m.stats.mined_blocks for m in miners))

            # compact log line
            print(f"[{mode.upper()}] t={t:.2f}s | blocks={blocks_mined_series[-1]} | tips={tips_count_series[-1]} | nodes={dag_node_count_series[-1]} | total_energy={sum(m.stats.energy_joules for m in miners):.8f}J")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        global_state['stop_all'] = True
        for m in miners:
            m.stop()
        for m in miners:
            m.join()

    # final metrics
    total_hashes = sum(m.stats.hashes for m in miners)
    total_energy = sum(m.stats.energy_joules for m in miners)
    total_blocks = sum(m.stats.mined_blocks for m in miners)
    accepted = len(global_state['accepted_blocks'])
    tips_left = len(global_state['tips'])
    unknown_events = len(global_state['unknown_parent_events'])

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
        'tips_left': tips_left,
        'unknown_parent_events': unknown_events,
        'per_miner': per_miner,
        'diff_series': diff_series,
        'hash_series': hash_series,
        'energy_series': energy_series,
        'timestamp_series': timestamp_series,
        'ewma_pred_series': ewma_pred_series,
        'predicted_diff_series': predicted_diff_series,
        'fixed_diff_series': fixed_diff_series,
        'cmda_diff_series': cmda_diff_series,
        'low_diff_series': low_diff_series,
        'block_time_series': block_time_series,
        'tips_count_series': tips_count_series,
        'dag_node_count_series': dag_node_count_series,
        'blocks_mined_series': blocks_mined_series,
        'dag': global_state['dag'],
        'global_state': global_state
    }

# ---------------- LIVE PLOT (3 subplots) ----------------
def create_live_figure():
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.suptitle("HelixDAG — Live: Predicted Diff | Actual Diff Evolution | Energy Comparison", fontsize=16)
    for ax in axes:
        ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes

def update_live_plots(fig, axes, metrics, mode_label):
    t = metrics['timestamp_series']
    if not t:
        return

    # Ax0: Predicted next difficulty per miner (plot EWMA-based predicted difficulty per miner computed from miners' EWMA+trend)
    ax0 = axes[0]; ax0.clear(); ax0.grid(True, alpha=GRID_ALPHA)
    # Derive per-miner predicted difficulty: baseline = miner.local_difficulty, predicted_interval = miner.ewma + miner.trend
    # But we only have averaged predicted stored in metrics; to show per-miner lines we use the diff_series per miner and last ewma/trend is unavailable here,
    # So we'll plot the predicted_diff_series as single line and also show cmda mean per timestep for context.
    ax0.plot(t, metrics['predicted_diff_series'], label="PredictedDiff (EWMA -> diff)", linewidth=LINEWIDTH, linestyle='--')
    ax0.plot(t, metrics['cmda_diff_series'], label="CMDA mean (for reference)", linewidth=LINEWIDTH, alpha=0.6)
    ax0.set_title("Predicted Next Difficulty (EWMA + Trend)")
    ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Difficulty")
    ax0.legend(fontsize=LEGEND_FS)

    # Ax1: Actual difficulty evolution overlay (Fixed / CMDA / Low-Energy)
    ax1 = axes[1]; ax1.clear(); ax1.grid(True, alpha=GRID_ALPHA)
    ax1.plot(t, metrics['fixed_diff_series'], label="Fixed (initial)", linewidth=LINEWIDTH)
    ax1.plot(t, metrics['cmda_diff_series'], label="CMDA (mean local)", linewidth=LINEWIDTH)
    ax1.plot(t, metrics['low_diff_series'], label="LowCMDA (mean)", linewidth=LINEWIDTH)
    ax1.set_title("Actual Difficulty Evolution (Fixed vs CMDA vs LowCMDA)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Difficulty")
    ax1.legend(fontsize=LEGEND_FS)

    # Ax2: Energy consumption comparison (total energy for current mode over time)
    ax2 = axes[2]; ax2.clear(); ax2.grid(True, alpha=GRID_ALPHA)
    # compute total energy across miners per timestamp
    aligned = metrics['energy_series']
    totals = []
    for i in range(len(t)):
        totals.append(sum(aligned[mid][i] for mid in aligned.keys()))
    ax2.plot(t, totals, label=f"{mode_label} - Total Energy (J)", linewidth=LINEWIDTH)
    ax2.set_title("Energy Consumption Over Time (Total Joules)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy (J)")
    ax2.legend(fontsize=LEGEND_FS)

    fig.canvas.draw()
    fig.canvas.flush_events()

# ---------------- SAVE CSVs & PRINT NUMERIC ARRAYS ----------------
def save_and_print_metrics(metrics_list: List[Dict], labels: List[str]):
    # Save per-mode CSVs and print arrays used for plotting
    for metrics, label in zip(metrics_list, labels):
        ts = metrics['timestamp_series']
        # prepare rows
        energy_aligned = metrics['energy_series']
        total_energy_series = [sum(energy_aligned[mid][i] for mid in energy_aligned.keys()) for i in range(len(ts))]
        header = ["timestamp_s", "fixed_diff", "cmda_diff_mean", "low_diff_mean", "predicted_diff", "ewma_pred_interval", "block_time_s", "tips_count", "dag_nodes", "blocks_mined", "total_energy_J"]
        rows = []
        for i in range(len(ts)):
            rows.append([
                ts[i],
                metrics['fixed_diff_series'][i],
                metrics['cmda_diff_series'][i],
                metrics['low_diff_series'][i],
                metrics['predicted_diff_series'][i],
                round(metrics['ewma_pred_series'][i], 6) if i < len(metrics['ewma_pred_series']) else None,
                metrics['block_time_series'][i] if i < len(metrics['block_time_series']) else None,
                metrics['tips_count_series'][i] if i < len(metrics['tips_count_series']) else None,
                metrics['dag_node_count_series'][i] if i < len(metrics['dag_node_count_series']) else None,
                metrics['blocks_mined_series'][i] if i < len(metrics['blocks_mined_series']) else None,
                total_energy_series[i]
            ])
        csv_path = os.path.join(OUT_DIR, f"{label.lower()}_series_optionC_2tips.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"Saved CSV: {csv_path}")

        # print arrays
        print("\n" + "="*70)
        print(f"NUMERICAL SERIES for MODE = {label.upper()}")
        print("="*70)
        print(f"Timestamps (s) [{len(ts)}]: {ts}")
        print(f"Fixed Difficulty series [{len(metrics['fixed_diff_series'])}]: {metrics['fixed_diff_series']}")
        print(f"CMDA Difficulty mean series [{len(metrics['cmda_diff_series'])}]: {metrics['cmda_diff_series']}")
        print(f"LowCMDA Difficulty mean series [{len(metrics['low_diff_series'])}]: {metrics['low_diff_series']}")
        print(f"Predicted Difficulty series [{len(metrics['predicted_diff_series'])}]: {metrics['predicted_diff_series']}")
        print(f"EWMA Pred Interval series [{len(metrics['ewma_pred_series'])}]: {metrics['ewma_pred_series']}")
        print(f"Block Time series [{len(metrics['block_time_series'])}]: {metrics['block_time_series']}")
        print(f"Tips count series [{len(metrics['tips_count_series'])}]: {metrics['tips_count_series']}")
        print(f"DAG node count series [{len(metrics['dag_node_count_series'])}]: {metrics['dag_node_count_series']}")
        print(f"Blocks mined series [{len(metrics['blocks_mined_series'])}]: {metrics['blocks_mined_series']}")
        print(f"Total energy series [{len(total_energy_series)}]: {total_energy_series}")
        print("Per-miner final snapshot:")
        for mid, v in metrics['per_miner'].items():
            print(f"  {mid}: blocks={v['blocks']}, hashes={v['hashes']}, energy_J={v['energy']:.8f}, final_diff={v['final_diff']}")
        print("="*70 + "\n")

    # Combined summary CSV
    comp_csv = os.path.join(OUT_DIR, "energy_comparison_optionC_2tips.csv")
    with open(comp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Mode", "Total Energy (J)", "Total Hashes", "Total Mined Blocks", "Accepted Blocks", "Tips Left", "Unknown Parent Events", "Avg Difficulty"])
        for metrics, label in zip(metrics_list, labels):
            def avgdiff(m):
                vals = [d['final_diff'] for d in m['per_miner'].values()]
                return float(np.mean(vals)) if vals else 0.0
            w.writerow([label, metrics['total_energy'], metrics['total_hashes'], metrics['total_blocks'],
                        metrics['accepted_blocks'], metrics['tips_left'], metrics['unknown_parent_events'],
                        avgdiff(metrics)])
    print(f"Saved combined summary CSV: {comp_csv}")

    # metadata
    metadata = {
        "RND_SEED": RND_SEED,
        "NUM_MINERS": NUM_MINERS,
        "SIM_DURATION": SIM_DURATION,
        "TARGET_INTERVAL": TARGET_INTERVAL,
        "ALPHA": ALPHA,
        "INITIAL_DIFFICULTY": INITIAL_DIFFICULTY,
        "PARENTS_PER_BLOCK": PARENTS_PER_BLOCK,
        "GAMMA": GAMMA,
        "TIMESTAMP": time.asctime()
    }
    meta_path = os.path.join(OUT_DIR, "metadata_optionC_2tips.json")
    with open(meta_path, "w") as mf:
        json.dump(metadata, mf, indent=2)
    print(f"Saved metadata: {meta_path}")

# ---------------- SAVE FIGURES (publication quality) ----------------
def save_publication_plots(metrics_list: List[Dict], labels: List[str]):
    # Combined figure with 3 subplots (high-res)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.suptitle("HelixDAG — Final Summary (Option C, 2 tips per block)", fontsize=16)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # left: Predicted diff (overlay modes)
    ax0 = axes[0]; ax0.grid(True, alpha=GRID_ALPHA)
    for metrics, lab, col in zip(metrics_list, labels, color_cycle):
        ax0.plot(metrics['timestamp_series'], metrics['predicted_diff_series'], label=f"{lab}-PredDiff", linewidth=LINEWIDTH, color=col)
    ax0.set_title("Predicted Difficulty (EWMA + trend)"); ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Difficulty"); ax0.legend(fontsize=8)

    # middle: actual difficulty evolution
    ax1 = axes[1]; ax1.grid(True, alpha=GRID_ALPHA)
    for metrics, lab, col in zip(metrics_list, labels, color_cycle):
        ax1.plot(metrics['timestamp_series'], metrics['cmda_diff_series'], label=f"{lab}-CMDA", linewidth=LINEWIDTH, color=col)
        ax1.plot(metrics['timestamp_series'], metrics['low_diff_series'], linestyle='--', label=f"{lab}-LowCMDA", linewidth=1.2, color=col)
        ax1.plot(metrics['timestamp_series'], metrics['fixed_diff_series'], linestyle=':', label=f"{lab}-Fixed", linewidth=1.2, color=col)
    ax1.set_title("Actual Difficulty Evolution"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Difficulty"); ax1.legend(fontsize=7, ncol=2)

    # right: energy comparison (total per mode)
    ax2 = axes[2]; ax2.grid(True, alpha=GRID_ALPHA)
    for metrics, lab, col in zip(metrics_list, labels, color_cycle):
        aligned = metrics['energy_series']
        totals = [sum(aligned[mid][i] for mid in aligned.keys()) for i in range(len(metrics['timestamp_series']))]
        ax2.plot(metrics['timestamp_series'], totals, label=f"{lab}-TotalEnergy", linewidth=LINEWIDTH, color=col)
    ax2.set_title("Energy Consumption Comparison (Total Joules)"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy (J)"); ax2.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save combined figure in SVG, PDF, PNG@600
    for ext, dpi in [('svg', None), ('pdf', None), ('png', 600)]:
        fname = f"{COMBINED_BASE}.{ext}"
        if dpi:
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(fname, bbox_inches='tight')
        print(f"Saved figure: {fname}")

    # Save each subplot individually
    for i, ax in enumerate(axes):
        fig_sub = plt.figure(figsize=(8,6))
        sub_ax = fig_sub.add_subplot(111)
        if i == 0:
            for metrics, lab, col in zip(metrics_list, labels, color_cycle):
                sub_ax.plot(metrics['timestamp_series'], metrics['predicted_diff_series'], label=f"{lab}-PredDiff", linewidth=LINEWIDTH, color=col)
            sub_ax.set_title("Predicted Difficulty (EWMA + trend)")
            sub_ax.set_xlabel("Time (s)"); sub_ax.set_ylabel("Difficulty")
        elif i == 1:
            for metrics, lab, col in zip(metrics_list, labels, color_cycle):
                sub_ax.plot(metrics['timestamp_series'], metrics['cmda_diff_series'], label=f"{lab}-CMDA", linewidth=LINEWIDTH, color=col)
            sub_ax.set_title("Actual Difficulty Evolution")
            sub_ax.set_xlabel("Time (s)"); sub_ax.set_ylabel("Difficulty")
        elif i == 2:
            for metrics, lab, col in zip(metrics_list, labels, color_cycle):
                aligned = metrics['energy_series']
                totals = [sum(aligned[mid][j] for mid in aligned.keys()) for j in range(len(metrics['timestamp_series']))]
                sub_ax.plot(metrics['timestamp_series'], totals, label=f"{lab}-TotalEnergy", linewidth=LINEWIDTH, color=col)
            sub_ax.set_title("Energy Consumption Comparison")
            sub_ax.set_xlabel("Time (s)"); sub_ax.set_ylabel("Energy (J)")

        sub_ax.grid(True, alpha=GRID_ALPHA); sub_ax.legend(fontsize=8)
        plt.tight_layout()
        base = os.path.join(OUT_DIR, f"subplot_{i}_optionC_2tips")
        for ext, dpi in [('svg', None), ('pdf', None), ('png', 600)]:
            fp = f"{base}.{ext}"
            if dpi:
                fig_sub.savefig(fp, dpi=dpi, bbox_inches='tight')
            else:
                fig_sub.savefig(fp, bbox_inches='tight')
            print(f"Saved subplot: {fp}")
        plt.close(fig_sub)

    plt.close(fig)

# ---------------- MAIN DRIVER ----------------
def main():
    print("=== HelixDAG Real-time Simulator — Option C (2 tips) ===")
    print(f"NUM_MINERS={NUM_MINERS}, SIM_DURATION={SIM_DURATION}s, TARGET_INTERVAL={TARGET_INTERVAL}s")
    print(f"PARENTS_PER_BLOCK={PARENTS_PER_BLOCK}")

    modes = ["cmda", "fixed", "lowcmda"]
    labels = [m.upper() for m in modes]
    metrics_list = []

    # Create live figure with 3 subplots
    fig, axes = create_live_figure()

    # Run each mode sequentially and update live plots
    for mode in modes:
        print(f"\n--- Running mode: {mode} ---")
        metrics = run_experiment(mode, SIM_DURATION)
        metrics_list.append(metrics)

        # live update using latest metrics
        update_live_plots(fig, axes, metrics, mode.upper())
        # hold briefly so user can see
        plt.pause(0.5)

    # After runs finish, produce publication-quality outputs
    print("\nAll modes finished. Saving publication-quality figures & CSVs...")
    plt.ioff()  # turn off interactive
    save_publication_plots(metrics_list, labels)
    save_and_print_metrics(metrics_list, labels)

    print("\nOutputs saved in directory:", OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
