#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_train_resume.py — Unified trainer for DQN, Double DQN, and Dueling DQN with resume support.
Prints per-episode progress lines: "[ep XXX] reward=..." (flush=True).
"""
import os, time, argparse, subprocess, collections
from typing import Deque, Tuple

import numpy as np
import traci

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- SUMO helpers -----------------------------
def launch_sumo(sumo_bin: str, net: str, rou: str, port: int, begin: int, end: int, extra=None):
    cmd = [sumo_bin, "--remote-port", str(port), "-n", net, "-r", rou, "--begin", str(begin), "--end", str(end)]
    if extra: cmd += extra
    return subprocess.Popen(cmd)

def choose_tls() -> str:
    ids = traci.trafficlight.getIDList()
    if not ids:
        raise RuntimeError("No traffic lights found in network")
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

# ----------------------------- Q-networks -------------------------------
class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

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

# ----------------------------- Replay buffer ---------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = collections.deque(maxlen=capacity)
    def push(self, s, a, r, s2, d): self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size: int):
        import random as _r
        batch = _r.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))
    def __len__(self): return len(self.buf)

# ----------------------------- Utilities --------------------------------
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

def dqn_update(qnet, tgt, opt, batch, gamma: float, device: str, algo: str):
    s, a, r, s2, d = batch
    s  = torch.from_numpy(s).to(device)
    a  = torch.from_numpy(a).long().to(device).unsqueeze(1)
    r  = torch.from_numpy(r).to(device).unsqueeze(1)
    s2 = torch.from_numpy(s2).to(device)
    d  = torch.from_numpy(d).to(device).unsqueeze(1)

    q = qnet(s).gather(1, a)
    with torch.no_grad():
        if algo == "double":
            a2 = qnet(s2).argmax(dim=1, keepdim=True)
            q2 = tgt(s2).gather(1, a2)
        else:
            q2 = tgt(s2).max(dim=1, keepdim=True)[0]
        y  = r + gamma * (1.0 - d) * q2
    loss = torch.nn.functional.smooth_l1_loss(q, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
    opt.step()
    return float(loss.item())

# ------------------------------ Training ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    # SUMO / episode
    ap.add_argument("--sumo-bin", default="sumo")
    ap.add_argument("--net", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--ep-seconds", type=int, default=3600)
    ap.add_argument("--decision", type=int, default=5)
    ap.add_argument("--extend-s", type=int, default=5)
    ap.add_argument("--min-green", type=int, default=10)
    ap.add_argument("--max-green", type=int, default=60)
    ap.add_argument("--switch-penalty", type=float, default=0.05)
    ap.add_argument("--sumo-extra", nargs="*", default=None, help="extra flags for sumo, e.g., --step-length 1.0 --meso")
    # Algo
    ap.add_argument("--algo", choices=["dqn","double","dueling"], default="dqn")
    # DQN
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--buffer-size", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target-update", type=int, default=2000)
    # Exploration
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.05)
    ap.add_argument("--epsilon-decay-steps", type=int, default=80000)
    # Misc
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--save", default="rl_policy.pt")
    ap.add_argument("--load", default=None, help="optional checkpoint to resume from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clip-q", type=float, default=15.0)
    args = ap.parse_args()

    import random as pyrandom
    pyrandom.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    n_actions = 2
    state_dim = 4 + 2 + 1
    if args.algo == "dueling":
        qnet = DuelingQNet(state_dim, n_actions, hidden=args.hidden).to(device)
        tgt  = DuelingQNet(state_dim, n_actions, hidden=args.hidden).to(device)
    else:
        qnet = QNet(state_dim, n_actions, hidden=args.hidden).to(device)
        tgt  = QNet(state_dim, n_actions, hidden=args.hidden).to(device)
    opt  = optim.Adam(qnet.parameters(), lr=args.lr)
    rb   = ReplayBuffer(args.buffer_size)

    global_step = 0
    eps = args.epsilon_start
    eps_decay = (args.epsilon_start - args.epsilon_end) / max(1, args.epsilon_decay_steps)

    # Optional resume
    if args.load and os.path.exists(args.load):
        ckpt = torch.load(args.load, map_location=device)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        try:
            qnet.load_state_dict(sd)
        except Exception as e:
            raise SystemExit(f"Checkpoint arch mismatch for --algo={args.algo}. Did you load a model from a different algo?") from e
        if "target" in ckpt:
            try: tgt.load_state_dict(ckpt["target"])
            except Exception: pass
        if "steps" in ckpt: global_step = int(ckpt["steps"])

    for ep in range(1, args.episodes+1):
        proc = launch_sumo(args.sumo_bin, args.net, args.routes, args.port, 0, args.ep_seconds, args.sumo_extra)
        time.sleep(0.5)
        traci.init(args.port)

        tls = choose_tls()
        g = groups_by_approach(tls)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
        ns_idx = ew_idx = None
        for i, ph in enumerate(logic.phases):
            lab = label_phase(ph.state, g)
            if lab=="NS" and ns_idx is None: ns_idx = i
            if lab=="EW" and ew_idx is None: ew_idx = i
        if ns_idx is None or ew_idx is None:
            raise RuntimeError("Could not find NS/EW-only phases")
        traci.trafficlight.setPhase(tls, ns_idx)
        traci.trafficlight.setPhaseDuration(tls, args.min_green)

        t = 0.0
        next_decision = args.decision
        ep_reward = 0.0
        losses = []
        s = None

        while t < args.ep_seconds:
            traci.simulationStep()
            t = traci.simulation.getTime()

            if t >= next_decision:
                if s is None:
                    s = build_state(tls, g, args.max_green, args.clip_q)
                import random as pyrand
                if pyrand.random() < eps:
                    a = pyrand.randrange(n_actions)
                else:
                    with torch.no_grad():
                        q = qnet(torch.from_numpy(s).to(device).unsqueeze(0))
                        a = int(torch.argmax(q, dim=1).item())

                queues = 0.0
                for e in halting_edges():
                    try: queues += traci.edge.getLastStepHaltingNumber(e)
                    except Exception: pass
                r = -float(queues) - (args.switch_penalty if a==1 else 0.0)
                ep_reward += r

                if a == 0:
                    rem = traci.trafficlight.getNextSwitch(tls) - t
                    new_rem = min(max(0.1, rem + args.extend_s), float(args.max_green))
                    traci.trafficlight.setPhaseDuration(tls, new_rem)
                else:
                    traci.trafficlight.setPhaseDuration(tls, 0.1)

                la = max(1, int(args.decision/2))
                for _ in range(la):
                    traci.simulationStep()
                t = traci.simulation.getTime()

                s2 = build_state(tls, g, args.max_green, args.clip_q)
                done = (t >= args.ep_seconds - 1e-3)
                rb.push(s, a, r, s2, done)
                s = s2
                next_decision = t + args.decision

                if len(rb) >= args.warmup:
                    batch = rb.sample(args.batch_size)
                    loss = dqn_update(qnet, tgt, opt, batch, args.gamma, device, args.algo)
                    losses.append(loss)
                    global_step += 1
                    if global_step % args.target_update == 0:
                        tgt.load_state_dict(qnet.state_dict())

                if eps > args.epsilon_end:
                    eps = max(args.epsilon_end, eps - eps_decay)

        traci.close()
        try: proc.terminate()
        except Exception: pass

        avg_loss = (sum(losses)/len(losses)) if losses else 0.0
        print(f"[ep {ep:03d}] reward={ep_reward:.1f}  eps={eps:.3f}  avg_loss={avg_loss:.4f}  rb={len(rb)}", flush=True)

        ckpt = {"model": qnet.state_dict(), "target": tgt.state_dict(), "config": vars(args), "steps": global_step}
        torch.save(ckpt, args.save)

    ckpt = {"model": qnet.state_dict(), "target": tgt.state_dict(), "config": vars(args), "steps": global_step}
    torch.save(ckpt, args.save)
    print(f"[done] saved {args.save}")

if __name__ == "__main__":
    main()
