#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_run.py — Run SUMO using a trained RL policy (DQN / Double / Dueling) and log KPIs to CSV.
Detects network architecture from checkpoint config["algo"].
"""
import os, time, argparse, subprocess, csv, datetime
import numpy as np
import traci
import torch
import torch.nn as nn

# ------------------------ Nets ------------------------
class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)

class DuelingQNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        z = self.feature(x)
        A = self.advantage(z)
        V = self.value(z)
        return V + A - A.mean(dim=1, keepdim=True)

# ------------------------ SUMO helpers -----------------
def launch_sumo(sumo_bin: str, net: str, rou: str, port: int, begin: int, end: int, extra=None):
    cmd = [sumo_bin, "--remote-port", str(port), "-n", net, "-r", rou, "--begin", str(begin), "--end", str(end)]
    if extra: cmd += extra
    return subprocess.Popen(cmd)

def choose_tls():
    ids = traci.trafficlight.getIDList()
    if not ids: raise RuntimeError("No traffic lights found")
    return ids[0]

def groups_by_approach(tls_id: str):
    links = traci.trafficlight.getControlledLinks(tls_id)
    g = {"N":[], "S":[], "E":[], "W":[]}
    for i, link in enumerate(links):
        if not link or not link[0]: continue
        fe = link[0][0]
        if   fe.startswith("in_N"): g["N"].append(i)
        elif fe.startswith("in_S"): g["S"].append(i)
        elif fe.startswith("in_E"): g["E"].append(i)
        elif fe.startswith("in_W"): g["W"].append(i)
    return g

def label_phase(state: str, g):
    ns = g["N"] + g["S"]; ew = g["E"] + g["W"]
    ns_g = sum(1 for i in ns if i < len(state) and state[i] in ("G","g"))
    ew_g = sum(1 for i in ew if i < len(state) and state[i] in ("G","g"))
    if ns_g>0 and ew_g==0: return "NS"
    if ew_g>0 and ns_g==0: return "EW"
    return "OTHER"

def halting_edges():
    return [f"in_{d}" for d in ("N","S","E","W")]

# ------------------------ State builder ----------------
def build_state(tls_id: str, g, max_green: float, clip_q: float) -> np.ndarray:
    qs = []
    for e in halting_edges():
        try: q = float(traci.edge.getLastStepHaltingNumber(e))
        except Exception: q = 0.0
        qs.append(min(q, clip_q) / clip_q)
    prog = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    cur  = traci.trafficlight.getPhase(tls_id)
    lab  = label_phase(prog.phases[cur].state, g)
    onehot = [1.0, 0.0] if lab=="NS" else ([0.0, 1.0] if lab=="EW" else [0.0, 0.0])
    tnow = traci.simulation.getTime()
    rem  = max(0.0, traci.trafficlight.getNextSwitch(tls_id) - tnow)
    elapsed_norm = (max_green - min(max_green, rem)) / max_green
    return np.array(qs + onehot + [elapsed_norm], dtype=np.float32)

# --------------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo-bin", default="sumo")
    ap.add_argument("--net", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--model", required=True)
    ap.add_argument("--decision", type=int, default=5)
    ap.add_argument("--extend-s", type=int, default=5)
    ap.add_argument("--min-green", type=int, default=10)
    ap.add_argument("--max-green", type=int, default=60)
    ap.add_argument("--end", type=int, default=3600)
    ap.add_argument("--metrics-csv", default="rl_run_metrics.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clip-q", type=float, default=15.0)
    ap.add_argument("--sumo-extra", nargs="*", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location=args.device)
    cfg = ckpt.get("config", {})
    hidden = int(cfg.get("hidden", 128))
    algo = cfg.get("algo", "dqn")
    state_dim = 4 + 2 + 1
    n_actions = 2

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    if algo == "dueling":
        qnet = DuelingQNet(state_dim, n_actions, hidden=hidden).to(device)
    else:
        qnet = QNet(state_dim, n_actions, hidden=hidden).to(device)
    qnet.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    qnet.eval()

    proc = launch_sumo(args.sumo_bin, args.net, args.routes, args.port, 0, args.end, args.sumo_extra)
    time.sleep(0.5)
    traci.init(args.port)

    tls = choose_tls()
    g = groups_by_approach(tls)
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]

    ns_idx = ew_idx = None
    for i, ph in enumerate(logic.phases):
        lab = label_phase(ph.state, g)
        if lab=="NS" and ns_idx is None: ns_idx=i
        if lab=="EW" and ew_idx is None: ew_idx=i
    if ns_idx is None or ew_idx is None:
        raise RuntimeError("Could not find NS/EW-only phases")
    traci.trafficlight.setPhase(tls, ns_idx)
    traci.trafficlight.setPhaseDuration(tls, args.min_green)

    departed_total = 0
    switches = 0
    try:
        all_edges = set(traci.edge.getIDList())
    except Exception:
        all_edges = set()
    upstream_edges = [e for e in all_edges if e.startswith("src_") or e.startswith("in_")]
    if not upstream_edges: upstream_edges = halting_edges()
    out_edges = [f"out_{d}" for d in ("N","S","E","W") if f"out_{d}" in all_edges]
    wait_sec, spd_sum, spd_cnt = {}, {}, {}
    passed_ids = set()

    def kpi_step():
        nonlocal departed_total
        departed_total += len(traci.simulation.getDepartedIDList())
        for e in upstream_edges:
            try:
                vids = traci.edge.getLastStepVehicleIDs(e)
            except Exception:
                vids = []
            for vid in vids:
                if vid in passed_ids: continue
                v = traci.vehicle.getSpeed(vid)
                if v < 0.1:
                    wait_sec[vid] = wait_sec.get(vid, 0) + 1
                spd_sum[vid] = spd_sum.get(vid, 0.0) + max(0.0, v)
                spd_cnt[vid] = spd_cnt.get(vid, 0) + 1
        for e in out_edges:
            try:
                for vid in traci.edge.getLastStepVehicleIDs(e):
                    passed_ids.add(vid)
            except Exception:
                pass

    t = 0.0
    next_decision = args.decision

    with torch.no_grad():
        while t < args.end:
            traci.simulationStep()
            t = traci.simulation.getTime()
            kpi_step()

            if t >= next_decision:
                s = build_state(tls, g, args.max_green, args.clip_q)
                q = qnet(torch.from_numpy(s).to(device).unsqueeze(0))
                a = int(torch.argmax(q, dim=1).item())  # greedy

                if a == 0:
                    rem = traci.trafficlight.getNextSwitch(tls) - t
                    new_rem = min(max(0.1, rem + args.extend_s), float(args.max_green))
                    traci.trafficlight.setPhaseDuration(tls, new_rem)
                else:
                    traci.trafficlight.setPhaseDuration(tls, 0.1)
                    switches += 1

                next_decision = t + args.decision

    try: sim_t = traci.simulation.getTime()
    except Exception: sim_t = float(args.end)

    passed_list = list(passed_ids)
    if passed_list:
        waits = [float(wait_sec.get(v, 0)) for v in passed_list]
        speeds = []
        for v in passed_list:
            n = spd_cnt.get(v, 0); s = spd_sum.get(v, 0.0)
            speeds.append((s/n)*3.6 if n>0 else 0.0)
        avg_wait = sum(waits)/len(waits)
        avg_spd  = sum(speeds)/len(speeds)
    else:
        avg_wait = 0.0; avg_spd = 0.0
    still_up = max(0, departed_total - len(passed_list))

    outp = os.path.abspath(args.metrics_csv)
    hdr = not os.path.exists(outp)
    with open(outp, "a", newline="") as fh:
        w = csv.writer(fh)
        if hdr:
            w.writerow(["timestamp","sim_time_s","departed_total","passed_count","still_upstream",
                        "avg_waiting_s","avg_approach_speed_kmh","switches"])
        w.writerow([datetime.datetime.now().isoformat(), f"{sim_t:.2f}",
                    departed_total, len(passed_list), still_up,
                    f"{avg_wait:.3f}", f"{avg_spd:.3f}", switches])
    print(f"[RL run KPI] departed={departed_total} passed={len(passed_list)} still_up={still_up} "
          f"avg_wait={avg_wait:.1f}s avg_spd={avg_spd:.1f}km/h switches={switches} -> {outp}")

    traci.close()
    try: proc.terminate()
    except Exception: pass

if __name__ == "__main__":
    main()
