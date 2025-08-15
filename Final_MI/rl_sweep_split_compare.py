#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_sweep_split_compare.py — Train/val/test split for three algos: DQN, Double DQN, Dueling DQN.
- Short sweep (A/B/C) across TRAIN -> pick best on VAL (mean avg_wait).
- Long-train best on TRAIN.
- TEST evaluation on held-out scenarios.
- Generates comparison plots across algos.
"""
import os, csv, argparse, subprocess, re
from pathlib import Path
import matplotlib.pyplot as plt

EP_LINE = re.compile(r"\[ep\s+(\d+)\]\s+reward=([-\d\.]+)")

def read_manifest(path: Path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def run_train_resume(tag, model_path:Path, routes_list, episodes:int, common_args, log_path:Path):
    total_eps = episodes
    K = len(routes_list)
    base = episodes // K
    rem  = episodes % K
    alloc = [base + (1 if i < rem else 0) for i in range(K)]
    ep_offset = 0
    rewards_all = []
    for idx, (rou, epc) in enumerate(zip(routes_list, alloc)):
        if epc == 0: continue
        print(f"\n▶ Training {tag} on scenario {idx+1}/{K} for {epc} episodes")
        cmd = ["python","-u","rl_train_resume.py"] + common_args + ["--routes", rou, "--episodes", str(epc), "--save", str(model_path)]
        if model_path.exists():
            cmd += ["--load", str(model_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with proc.stdout as out:
            for line in out:
                m = EP_LINE.search(line)
                if m:
                    ep = int(m.group(1)); rw = float(m.group(2))
                    rewards_all.append(rw)
                    print(f"episode {ep_offset + ep}/{total_eps}")
        proc.wait()
        ep_offset += epc
    return rewards_all

def eval_avg(tag_name, model_path:Path, routes_list, run_common, outdir:Path, tag: str):
    rows = []
    for i, rou in enumerate(routes_list, 1):
        csvp = outdir / f"eval_{tag_name}_{tag}_{i:02d}.csv"
        cmd = ["python","-u","rl_run.py"] + run_common + ["--model", str(model_path), "--routes", rou, "--metrics-csv", str(csvp)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with p.stdout as out:
            for _ in out: pass
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"Eval failed for {tag_name} on {rou} (rc={rc})")
        last=None
        with open(csvp,"r") as fh:
            rd=csv.DictReader(fh)
            for r in rd: last=r
        if last: rows.append(last)
    import statistics as st
    def fmean(vals, cast=float):
        arr=[cast(v) for v in vals if v is not None]
        return sum(arr)/len(arr) if arr else float("nan")
    agg = {
        "avg_waiting_s_mean": fmean([r["avg_waiting_s"] for r in rows], float),
        "avg_waiting_s_std":  st.pstdev([float(r["avg_waiting_s"]) for r in rows]) if len(rows)>1 else 0.0,
        "passed_mean":        fmean([r["passed_count"] for r in rows], int),
        "still_up_mean":      fmean([r["still_upstream"] for r in rows], int),
        "speed_mean":         fmean([r["avg_approach_speed_kmh"] for r in rows], float),
        "switches_mean":      fmean([r.get("switches","0") for r in rows], int),
        "n_eval": len(rows)
    }
    print(f"{tag_name} | {tag} avg_wait={agg['avg_waiting_s_mean']:.1f}s ±{agg['avg_waiting_s_std']:.1f}  "
          f"passed~{agg['passed_mean']:.0f}  still~{agg['still_up_mean']:.0f}  "
          f"speed~{agg['speed_mean']:.1f}km/h  switches~{agg['switches_mean']:.0f}  (n={agg['n_eval']})")
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo-train-bin", default="sumo")
    ap.add_argument("--sumo-eval-bin", default="sumo")
    ap.add_argument("--net", required=True)
    ap.add_argument("--routes-train-list", required=True)
    ap.add_argument("--routes-val-list", required=True)
    ap.add_argument("--routes-test-list", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--short-episodes", type=int, default=40)
    ap.add_argument("--long-episodes", type=int, default=200)
    ap.add_argument("--ep-seconds", type=int, default=3600)
    ap.add_argument("--decision", type=int, default=5)
    ap.add_argument("--extend-s", type=int, default=5)
    ap.add_argument("--min-green", type=int, default=10)
    ap.add_argument("--max-green", type=int, default=60)
    ap.add_argument("--clip-q", type=float, default=15.0)
    ap.add_argument("--switch-penalty", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--outdir", default="compare_out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sumo-extra", nargs="*", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    train_routes = read_manifest(Path(args.routes_train_list))
    val_routes   = read_manifest(Path(args.routes_val_list))
    test_routes  = read_manifest(Path(args.routes_test_list))

    algos = ["dqn","double","dueling"]
    algo_names = {"dqn":"DQN","double":"DoubleDQN","dueling":"DuelingDQN"}

    grids = {
        "A": {"lr":1e-3,  "gamma":0.99,  "batch":128, "hidden":128, "target":2000, "eps_decay":80000},
        "B": {"lr":5e-4,  "gamma":0.995, "batch":256, "hidden":256, "target":3000, "eps_decay":120000},
        "C": {"lr":2e-3,  "gamma":0.985, "batch":64,  "hidden":128, "target":1500, "eps_decay":60000},
    }

    train_common = [
        "--sumo-bin", args.sumo_train_bin, "--net", args.net, "--port", str(args.port),
        "--ep-seconds", str(args.ep_seconds), "--decision", str(args.decision),
        "--extend-s", str(args.extend_s), "--min-green", str(args.min_green), "--max-green", str(args.max_green),
        "--switch-penalty", str(args.switch_penalty), "--device", args.device, "--clip-q", str(args.clip_q)
    ]
    if args.sumo_extra: train_common += ["--sumo-extra"] + args.sumo_extra
    run_common = [
        "--sumo-bin", args.sumo_eval_bin, "--net", args.net, "--port", str(args.port),
        "--decision", str(args.decision), "--extend-s", str(args.extend_s),
        "--min-green", str(args.min_green), "--max-green", str(args.max_green),
        "--end", str(args.ep_seconds), "--device", args.device, "--clip-q", str(args.clip_q)
    ]

    val_summary = outdir / "compare_val_summary.csv"
    test_summary = outdir / "compare_test_summary.csv"
    with val_summary.open("w", newline="") as fh:
        csv.writer(fh).writerow(["algo","best_grid","val_avg_wait_mean","val_avg_wait_std","val_passed_mean","val_still_mean","val_speed_mean","val_switches_mean","model_path"])
    with test_summary.open("w", newline="") as fh:
        csv.writer(fh).writerow(["algo","test_avg_wait_mean","test_avg_wait_std","test_passed_mean","test_still_mean","test_speed_mean","test_switches_mean","model_path"])

    val_metrics = {}; test_metrics = {}; train_curves = {}

    print("===== COMPARISON: DQN vs Double vs Dueling =====")
    for algo in algos:
        tag = algo_names[algo]
        out_algo = outdir / f"{algo}"; out_algo.mkdir(parents=True, exist_ok=True)
        best = None  # (score, grid_key, model_path, agg_val)
        curves = {}

        print(f"\n=== {tag}: short sweep across TRAIN scenarios ===")
        for key, cfg in grids.items():
            model_path = out_algo / f"{algo}_{key}.pt"
            args_list = train_common + [
                "--algo", algo,
                "--gamma", str(cfg["gamma"]), "--lr", str(cfg["lr"]), "--hidden", str(cfg["hidden"]),
                "--batch-size", str(cfg["batch"]), "--target-update", str(cfg["target"]),
                "--epsilon-decay-steps", str(cfg["eps_decay"]), "--save", str(model_path),
                "--seed", str(args.seed)
            ]
            rewards = run_train_resume(f"{tag}-{key}", model_path, train_routes, args.short_episodes, args_list, out_algo / f"train_{key}.log")
            curves[key] = rewards

            agg_val = eval_avg(f"{tag}-{key}", model_path, val_routes, run_common, out_algo, tag="val")
            score = (agg_val["avg_waiting_s_mean"], -agg_val["passed_mean"])
            if (best is None) or (score < best[0]):
                best = (score, key, model_path, agg_val)

        # Plot sweep rewards for this algo
        plt.figure()
        for key, curve in curves.items():
            xs = list(range(1, len(curve)+1))
            plt.plot(xs, curve, label=f"{tag}-{key}")
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"{tag}: Sweep rewards (TRAIN)")
        plt.legend(); plt.savefig(out_algo / f"{algo}_sweep_rewards.png", dpi=160); plt.close()

        # Long train best
        grid_key = best[1]; start_model = best[2]; agg_val = best[3]
        best_grid = grids[grid_key]
        best_model_long = out_algo / f"{algo}_{grid_key}_long.pt"
        args_long = train_common + [
            "--algo", algo,
            "--gamma", str(best_grid["gamma"]), "--lr", str(best_grid["lr"]), "--hidden", str(best_grid["hidden"]),
            "--batch-size", str(best_grid["batch"]), "--target-update", str(best_grid["target"]),
            "--epsilon-decay-steps", str(best_grid["eps_decay"]), "--save", str(best_model_long),
            "--seed", str(args.seed), "--load", str(start_model)
        ]
        rewards_long = run_train_resume(f"{tag}-{grid_key}*", best_model_long, train_routes, args.long_episodes, args_long, out_algo / f"train_{grid_key}_long.log")
        plt.figure(); plt.plot(range(1, len(rewards_long)+1), rewards_long)
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"{tag}-{grid_key}: Long training rewards")
        plt.savefig(out_algo / f"{algo}_{grid_key}_long_rewards.png", dpi=160); plt.close()

        # Test evaluation
        agg_test = eval_avg(f"{tag}-{grid_key}(long)", best_model_long, test_routes, run_common, out_algo, tag="test")

        # Save summaries
        with val_summary.open("a", newline="") as fh:
            csv.writer(fh).writerow([tag, grid_key, f"{agg_val['avg_waiting_s_mean']:.3f}", f"{agg_val['avg_waiting_s_std']:.3f}",
                                     f"{agg_val['passed_mean']:.1f}", f"{agg_val['still_up_mean']:.1f}",
                                     f"{agg_val['speed_mean']:.3f}", f"{agg_val['switches_mean']:.1f}", str(best_model_long)])
        with test_summary.open("a", newline="") as fh:
            csv.writer(fh).writerow([tag, f"{agg_test['avg_waiting_s_mean']:.3f}", f"{agg_test['avg_waiting_s_std']:.3f}",
                                     f"{agg_test['passed_mean']:.1f}", f"{agg_test['still_up_mean']:.1f}",
                                     f"{agg_test['speed_mean']:.3f}", f"{agg_test['switches_mean']:.1f}", str(best_model_long)])

        val_metrics[tag] = agg_val
        test_metrics[tag] = agg_test
        train_curves[tag] = curves

    # ---- Comparison plots ----
    # VAL avg_wait
    labels = list(val_metrics.keys())
    means = [val_metrics[k]["avg_waiting_s_mean"] for k in labels]
    stds  = [val_metrics[k]["avg_waiting_s_std"] for k in labels]
    plt.figure(); plt.bar(labels, means, yerr=stds)
    plt.ylabel("Avg waiting time (s)"); plt.title("Validation: Average waiting time")
    plt.savefig(outdir / "cmp_val_avg_wait.png", dpi=160); plt.close()

    # TEST avg_wait
    labels = list(test_metrics.keys())
    means = [test_metrics[k]["avg_waiting_s_mean"] for k in labels]
    stds  = [test_metrics[k]["avg_waiting_s_std"] for k in labels]
    plt.figure(); plt.bar(labels, means, yerr=stds)
    plt.ylabel("Avg waiting time (s)"); plt.title("Test: Average waiting time")
    plt.savefig(outdir / "cmp_test_avg_wait.png", dpi=160); plt.close()

    # TEST throughput (passed)
    vals = [test_metrics[k]["passed_mean"] for k in labels]
    plt.figure(); plt.bar(labels, vals)
    plt.ylabel("Passed vehicles"); plt.title("Test: Throughput (passed count)")
    plt.savefig(outdir / "cmp_test_passed.png", dpi=160); plt.close()

    # TEST speed
    vals = [test_metrics[k]["speed_mean"] for k in labels]
    plt.figure(); plt.bar(labels, vals)
    plt.ylabel("Avg approach speed (km/h)"); plt.title("Test: Average approach speed")
    plt.savefig(outdir / "cmp_test_speed.png", dpi=160); plt.close()

    # TEST switches
    vals = [test_metrics[k]["switches_mean"] for k in labels]
    plt.figure(); plt.bar(labels, vals)
    plt.ylabel("Signal switches"); plt.title("Test: Switch count")
    plt.savefig(outdir / "cmp_test_switches.png", dpi=160); plt.close()

    # TEST scatter: avg_wait vs passed
    xs = [test_metrics[k]["avg_waiting_s_mean"] for k in labels]
    ys = [test_metrics[k]["passed_mean"] for k in labels]
    plt.figure(); plt.scatter(xs, ys)
    for i, lab in enumerate(labels): plt.annotate(lab, (xs[i], ys[i]))
    plt.xlabel("Avg waiting time (s)"); plt.ylabel("Passed vehicles"); plt.title("Test: Avg wait vs Throughput")
    plt.savefig(outdir / "cmp_test_wait_vs_passed.png", dpi=160); plt.close()

    print("\n=== Comparison complete ===")
    print("VAL summary  :", outdir / "compare_val_summary.csv")
    print("TEST summary :", outdir / "compare_test_summary.csv")
    print("Plots in     :", outdir)

if __name__ == "__main__":
    main()
