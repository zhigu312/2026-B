#!/usr/bin/env python3
"""B题安全残差PPO v6离线训练器。

同种子几何基线对照、多局批量GAE、分层源数、固定验证集、
全清硬门槛、可行域信息增益奖励、P90/最坏时间最优检查点和早停。
默认单进程低内存。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sim = _load("b_offline_simulator", SCRIPT_DIR / "local_offline_simulator.py")
ppo = _load("b_ppo_agent", SCRIPT_DIR / "ceshi_actor_critic.py")
base = ppo.base


class _DiscardOutput:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def aggregate(rows: List[dict]) -> dict:
    valid = [row for row in rows if not row.get("error")]
    times = [float(row["virtual_time_s"]) for row in valid]
    regrets = [
        float(row["virtual_time_s"]) - float(row["baseline_time_s"])
        for row in valid if row.get("baseline_time_s") not in (None, "")
    ]
    cleared = sum(int(row["cleared_count"]) for row in valid)
    total = sum(int(row["jammer_count"]) for row in valid)
    decisions = sum(int(row.get("ppo_decisions", 0)) for row in valid)
    deviations = sum(int(row.get("policy_deviations", 0)) for row in valid)
    proposed = sum(int(row.get("policy_proposed_deviations", 0)) for row in valid)
    gate_blocks = sum(int(row.get("safety_gate_blocks", 0)) for row in valid)
    entropy_weight = sum(
        float(row.get("mean_policy_entropy", 0.0))
        * int(row.get("ppo_decisions", 0)) for row in valid
    )
    observations = sum(
        float(row.get("mean_observations_per_cleared", 0.0))
        * int(row.get("cleared_count", 0)) for row in valid
    )
    jung_ready = sum(
        int(row.get("geometry_jung_certified_channels", 0)) for row in valid
    )
    mec_ready = sum(
        int(row.get("geometry_mec_ready_channels", 0)) for row in valid
    )
    feasible_radius_weight = sum(
        float(row.get("mean_final_feasible_radius_m", 0.0))
        * int(row.get("cleared_count", 0)) for row in valid
    )
    return {
        "episodes": len(rows),
        "successful_episodes": len(valid),
        "all_cleared_episode_rate": (
            sum(row["cleared_count"] == row["jammer_count"] for row in valid)
            / max(1, len(valid))
        ),
        "source_clear_rate": cleared / max(1, total),
        "virtual_time_mean_s": statistics.mean(times) if times else None,
        "virtual_time_median_s": statistics.median(times) if times else None,
        "virtual_time_p90_s": percentile(times, 0.90),
        "virtual_time_max_s": max(times) if times else None,
        "virtual_time_min_s": min(times) if times else None,
        "mean_regret_vs_baseline_s": statistics.mean(regrets) if regrets else None,
        "median_regret_vs_baseline_s": statistics.median(regrets) if regrets else None,
        "p90_regret_vs_baseline_s": percentile(regrets, 0.90),
        "max_regret_vs_baseline_s": max(regrets) if regrets else None,
        "policy_changed_rate": deviations / max(1, decisions),
        "policy_proposed_rate": proposed / max(1, decisions),
        "safety_gate_block_rate": gate_blocks / max(1, proposed),
        "mean_policy_entropy": entropy_weight / max(1, decisions),
        "mean_observations_per_cleared": observations / max(1, cleared),
        "geometry_jung_certified_rate": jung_ready / max(1, cleared),
        "geometry_mec_ready_rate": mec_ready / max(1, cleared),
        "mean_final_feasible_radius_m": feasible_radius_weight / max(1, cleared),
    }


def make_scheduler(path: Path, training: bool, args, baseline=False):
    return ppo.MaskedPPOPolicy(
        path, training=training,
        min_policy_episodes=(10**9 if baseline else 0),
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio, actor_lr=args.actor_lr,
        critic_lr=args.critic_lr, update_epochs=args.update_epochs,
        target_kl=args.target_kl, batch_episodes=args.batch_episodes,
        minibatch_size=args.minibatch_size, entropy_coef=args.entropy_coef,
        geometry_prior=args.geometry_prior,
        residual_warmup_episodes=args.residual_warmup_episodes,
    )


def make_agent(client, scheduler, problem: int, args,
               baseline_time_s: Optional[float], true_jammer_count: int):
    q_policy = base.QLearningPolicy(
        SCRIPT_DIR.parent / "rl_policy.json",
        epsilon=0.0, alpha=0.0, training=False
    )
    return ppo.PPOSearchAgent(
        client, q_policy, problem, scheduler=scheduler,
        max_detour_s=args.max_detour_s, candidate_limit=args.candidate_limit,
        safety_probability=args.safety_probability,
        safety_margin=args.safety_margin,
        safety_two_step_extra_s=args.safety_two_step_extra_s,
        max_consecutive_deviations=args.max_consecutive_deviations,
        tail_threshold_s=args.tail_threshold_s,
        tail_risk_coef=args.tail_risk_coef,
        baseline_time_s=baseline_time_s, true_jammer_count=true_jammer_count,
        max_probes=args.max_probes,
        joint_coverage_localization=True,
        joint_coverage_warmup=args.joint_warmup,
        directed_plan_order=True, joint_probe_variants=True,
        immediate_refine=True, information_gated_coverage=True,
        information_rate_planning=True,
    )


def run_episode(problem: int, seed: int, count: int, scheduler, args,
                baseline_time_s: Optional[float] = None,
                verbose: bool = False) -> dict:
    client = sim.LocalSimulatorClient(
        problem=problem, seed=seed, robot_id="offline-train",
        distribution=sim.CaseDistribution(mode=args.distribution), count=count,
    )
    true_count = len(client.core.jammers)
    agent = make_agent(client, scheduler, problem, args,
                       baseline_time_s, true_count)
    start = time.perf_counter()
    try:
        output = (contextlib.nullcontext() if verbose
                  else contextlib.redirect_stdout(_DiscardOutput()))
        with output:
            summary = agent.run()
        truth = client.truth()
        return {
            "seed": seed, "jammer_count": truth["jammer_count"],
            "cleared_count": truth["cleared_count"],
            "virtual_time_s": float(summary["virtual_time_s"]),
            "baseline_time_s": baseline_time_s,
            "path_distance_m": float(summary["path_distance_m"]),
            "action_count": int(summary["action_count"]),
            "ppo_decisions": agent.rl_decisions,
            "policy_deviations": agent.policy_deviations,
            "policy_proposed_deviations": agent.policy_proposed_deviations,
            "safety_gate_blocks": agent.safety_gate_blocks,
            "mean_policy_entropy": agent.policy_entropy_sum / max(1, agent.rl_decisions),
            "reward": agent.rl_reward_total,
            "mean_observations_per_cleared": float(
                summary.get("mean_observations_per_cleared", 0.0) or 0.0
            ),
            "geometry_jung_certified_channels": int(
                summary.get("geometry_jung_certified_channels", 0)
            ),
            "geometry_mec_ready_channels": int(
                summary.get("geometry_mec_ready_channels", 0)
            ),
            "mean_final_feasible_radius_m": float(
                summary.get("mean_final_feasible_radius_m", 0.0) or 0.0
            ),
            "wall_time_s": time.perf_counter() - start,
        }
    except Exception as exc:
        if scheduler.training:
            scheduler.abort_episode()
        return {
            "seed": seed, "jammer_count": true_count,
            "cleared_count": client.truth()["cleared_count"],
            "virtual_time_s": client.virtual_time_s,
            "baseline_time_s": baseline_time_s,
            "path_distance_m": sum(r["move_distance_m"] for r in client.records),
            "action_count": len(client.records),
            "ppo_decisions": agent.rl_decisions,
            "policy_deviations": agent.policy_deviations,
            "policy_proposed_deviations": agent.policy_proposed_deviations,
            "safety_gate_blocks": agent.safety_gate_blocks,
            "mean_policy_entropy": agent.policy_entropy_sum / max(1, agent.rl_decisions),
            "reward": agent.rl_reward_total,
            "wall_time_s": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }


def paired_episode(problem: int, seed: int, count: int, scheduler,
                   baseline_scheduler, args, verbose=False) -> dict:
    baseline = run_episode(problem, seed, count, baseline_scheduler, args,
                           verbose=verbose)
    if baseline.get("error"):
        return baseline
    result = run_episode(
        problem, seed, count, scheduler, args,
        baseline_time_s=float(baseline["virtual_time_s"]), verbose=verbose,
    )
    result["regret_vs_baseline_s"] = (
        float(result["virtual_time_s"]) - float(baseline["virtual_time_s"])
    )
    return result


def evaluate_model(model_path: Path, cases: List[tuple], baselines: List[dict],
                   args) -> List[dict]:
    evaluator = make_scheduler(model_path, training=False, args=args)
    rows = []
    for (seed, count), baseline in zip(cases, baselines):
        if baseline.get("error"):
            rows.append(dict(baseline))
            continue
        row = run_episode(
            args.problem, seed, count, evaluator, args,
            baseline_time_s=float(baseline["virtual_time_s"]),
            verbose=args.verbose,
        )
        row["regret_vs_baseline_s"] = (
            float(row["virtual_time_s"]) - float(baseline["virtual_time_s"])
        )
        rows.append(row)
    return rows


def safe_checkpoint_score(metrics: dict, max_regret_budget_s: float) -> tuple:
    clear_rate = float(metrics["all_cleared_episode_rate"])
    def value(name: str) -> float:
        item = metrics.get(name)
        return math.inf if item is None else float(item)
    mean_regret = value("mean_regret_vs_baseline_s")
    p90_regret = value("p90_regret_vs_baseline_s")
    max_regret = value("max_regret_vs_baseline_s")
    # 平均节时为主体，同时惩罚正向尾部；另设单局回退硬预算。
    # 这避免“基线的P90/最大后悔均为0，任何探索模型永远无法入选”。
    risk_adjusted_regret = (
        mean_regret + 0.75 * max(0.0, p90_regret)
        + 0.25 * max(0.0, max_regret)
    )
    return (
        0 if clear_rate >= 1.0 - 1e-12 else 1,
        -clear_rate,
        0 if max_regret <= max_regret_budget_s else 1,
        risk_adjusted_regret,
        p90_regret,
        max_regret,
        mean_regret,
    )


def mean_checkpoint_score(metrics: dict) -> tuple:
    """实验模型：全清优先，其次直接优化平均时间。"""
    clear_rate = float(metrics["all_cleared_episode_rate"])
    def value(name: str) -> float:
        item = metrics.get(name)
        return math.inf if item is None else float(item)
    return (
        0 if clear_rate >= 1.0 - 1e-12 else 1,
        -clear_rate,
        value("mean_regret_vs_baseline_s"),
        value("p90_regret_vs_baseline_s"),
        value("max_regret_vs_baseline_s"),
    )


def deploy_checkpoint_score(
    metrics: dict, p90_budget_s: float, max_budget_s: float,
) -> tuple:
    """部署模型：100%全清硬门，兼顾均值收益和可控尾部回退。"""
    clear_rate = float(metrics["all_cleared_episode_rate"])
    def value(name: str) -> float:
        item = metrics.get(name)
        return math.inf if item is None else float(item)
    mean_regret = value("mean_regret_vs_baseline_s")
    p90_regret = value("p90_regret_vs_baseline_s")
    max_regret = value("max_regret_vs_baseline_s")
    return (
        0 if clear_rate >= 1.0 - 1e-12 else 1,
        -clear_rate,
        0 if p90_regret <= p90_budget_s else 1,
        0 if max_regret <= max_budget_s else 1,
        mean_regret,
        p90_regret,
        max_regret,
    )


def write_curve(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_algorithm(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("algorithm")
    except (OSError, ValueError, TypeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B题安全接管残差PPO v6离线训练")
    parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=42)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--max-regret-budget-s", type=float, default=30.0,
        help="固定验证集中任一地图相对几何基线允许的最大回退秒数",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--distribution", choices=("official_estimate", "robust"),
                        default="official_estimate")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--safe-best-model-path", "--best-model-path",
        dest="safe_best_model_path", default=None,
    )
    parser.add_argument("--mean-best-model-path", default=None)
    parser.add_argument("--deploy-best-model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="继续训练v6工作模型")
    parser.add_argument("--deploy-p90-regret-budget-s", type=float, default=10.0)
    parser.add_argument("--deploy-max-regret-budget-s", type=float, default=150.0)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--joint-warmup", type=int, default=2)
    parser.add_argument("--max-probes", type=int, default=4)
    parser.add_argument("--max-detour-s", type=float, default=55.0)
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument("--safety-probability", type=float, default=0.54)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    parser.add_argument("--safety-two-step-extra-s", type=float, default=12.0)
    parser.add_argument("--max-consecutive-deviations", type=int, default=1)
    parser.add_argument("--tail-threshold-s", type=float, default=20.0)
    parser.add_argument("--tail-risk-coef", type=float, default=1.0)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--geometry-prior", type=float, default=0.40)
    parser.add_argument("--residual-warmup-episodes", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--actor-lr", type=float, default=0.0007)
    parser.add_argument("--critic-lr", type=float, default=0.002)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--target-kl", type=float, default=0.025)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.eval_episodes <= 0:
        raise SystemExit("episodes和eval-episodes必须>0")
    if args.validation_episodes <= 0 or args.validation_every <= 0:
        raise SystemExit("验证局数和验证间隔必须>0")
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        SCRIPT_DIR / "offline_training_v6" / f"problem_{args.problem}" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else (
        SCRIPT_DIR / "offline_models"
        / f"ppo_problem_{args.problem}_residual_v6_working.json"
    )
    safe_best_path = Path(args.safe_best_model_path) if args.safe_best_model_path else (
        SCRIPT_DIR / "offline_models"
        / f"ppo_problem_{args.problem}_residual_v6_safe_best.json"
    )
    mean_best_path = Path(args.mean_best_model_path) if args.mean_best_model_path else (
        SCRIPT_DIR / "offline_models"
        / f"ppo_problem_{args.problem}_residual_v6_mean_best.json"
    )
    deploy_best_path = Path(args.deploy_best_model_path) if args.deploy_best_model_path else (
        SCRIPT_DIR / "offline_models"
        / f"ppo_problem_{args.problem}_residual_v6_deploy_best.json"
    )
    for path in (model_path, safe_best_path, mean_best_path, deploy_best_path):
        incompatible = path.exists() and model_algorithm(path) != ppo.ALGORITHM_VERSION
        replace_old = path.exists() and not args.resume
        if incompatible or replace_old:
            archived = path.with_name(f"{path.stem}_before_{timestamp}{path.suffix}")
            path.replace(archived)
            print(f"旧模型已备份：{archived}")

    scheduler = make_scheduler(model_path, training=True, args=args)
    baseline_scheduler = make_scheduler(
        output_dir / "_baseline_never_saved.json",
        training=False, args=args, baseline=True,
    )
    wall_start = time.perf_counter()
    eval_cases = [
        (args.seed + 1_000_000 + i, 10 + i % 7)
        for i in range(args.eval_episodes)
    ]
    print(f"预计算{len(eval_cases)}个固定评估地图的几何基线……")
    baseline_rows = [
        run_episode(args.problem, seed, count, baseline_scheduler, args,
                    verbose=args.verbose)
        for seed, count in eval_cases
    ]
    validation_count = min(args.validation_episodes, len(eval_cases))
    validation_cases = eval_cases[:validation_count]
    validation_baselines = baseline_rows[:validation_count]

    scheduler.save()
    current_rows = evaluate_model(model_path, validation_cases, validation_baselines, args)
    current_metrics = aggregate(current_rows)
    safe_candidates = [(
        safe_checkpoint_score(current_metrics, args.max_regret_budget_s),
        "initial_working", current_rows,
    )]
    if safe_best_path.exists():
        rows = evaluate_model(
            safe_best_path, validation_cases, validation_baselines, args
        )
        safe_candidates.append((
            safe_checkpoint_score(aggregate(rows), args.max_regret_budget_s),
            "existing_safe", rows,
        ))
    safe_best_score, safe_source, _ = min(
        safe_candidates, key=lambda item: item[0]
    )
    if safe_source == "initial_working":
        scheduler.save(safe_best_path)

    mean_candidates = [(
        mean_checkpoint_score(current_metrics), "initial_working", current_rows,
    )]
    if mean_best_path.exists():
        rows = evaluate_model(
            mean_best_path, validation_cases, validation_baselines, args
        )
        mean_candidates.append((
            mean_checkpoint_score(aggregate(rows)), "existing_mean", rows,
        ))
    mean_best_score, mean_source, _ = min(
        mean_candidates, key=lambda item: item[0]
    )
    if mean_source == "initial_working":
        scheduler.save(mean_best_path)

    deploy_candidates = [(
        deploy_checkpoint_score(
            current_metrics, args.deploy_p90_regret_budget_s,
            args.deploy_max_regret_budget_s,
        ),
        "initial_working", current_rows,
    )]
    if deploy_best_path.exists():
        rows = evaluate_model(
            deploy_best_path, validation_cases, validation_baselines, args
        )
        deploy_candidates.append((
            deploy_checkpoint_score(
                aggregate(rows), args.deploy_p90_regret_budget_s,
                args.deploy_max_regret_budget_s,
            ),
            "existing_deploy", rows,
        ))
    deploy_best_score, deploy_source, _ = min(
        deploy_candidates, key=lambda item: item[0]
    )
    if deploy_source == "initial_working":
        scheduler.save(deploy_best_path)

    initial_metrics = current_metrics
    print(
        f"初始验证：全清={initial_metrics['all_cleared_episode_rate']:.1%} | "
        f"平均后悔={initial_metrics['mean_regret_vs_baseline_s']:+.1f}s | "
        f"P90后悔={initial_metrics['p90_regret_vs_baseline_s']:+.1f}s"
    )

    training_rows: List[dict] = []
    checkpoint_history: List[dict] = []
    stale_validations = 0
    stopped_early = False
    starting_model_episodes = scheduler.episodes
    for iteration in range(1, args.episodes + 1):
        # resume时接着使用新地图，避免从第1个seed重复训练。
        episode = starting_model_episodes + iteration
        seed = args.seed + episode - 1
        count = 10 + (episode - 1) % 7
        row = paired_episode(args.problem, seed, count, scheduler,
                             baseline_scheduler, args, args.verbose)
        row["episode"] = episode
        training_rows.append(row)
        if iteration % max(1, args.report_every) == 0 or iteration == args.episodes:
            recent = aggregate(training_rows[-min(28, len(training_rows)):])
            print(
                f"train {iteration}/{args.episodes} (模型累计{episode}) | "
                f"全清={recent['all_cleared_episode_rate']:.1%} | "
                f"平均后悔={recent['mean_regret_vs_baseline_s']:+.1f}s | "
                f"改策率={recent['policy_changed_rate']:.1%} | "
                f"熵={recent['mean_policy_entropy']:.3f}"
            )
        if iteration % args.validation_every == 0 or iteration == args.episodes:
            scheduler.flush_update()
            scheduler.save()
            rows = evaluate_model(model_path, validation_cases,
                                  validation_baselines, args)
            write_curve(output_dir / f"validation_{episode}.csv", rows)
            metrics = aggregate(rows)
            safe_score = safe_checkpoint_score(
                metrics, args.max_regret_budget_s
            )
            mean_score = mean_checkpoint_score(metrics)
            deploy_score = deploy_checkpoint_score(
                metrics, args.deploy_p90_regret_budget_s,
                args.deploy_max_regret_budget_s,
            )
            safe_improved = safe_score < safe_best_score
            mean_improved = mean_score < mean_best_score
            deploy_improved = deploy_score < deploy_best_score
            if safe_improved:
                safe_best_score = safe_score
                scheduler.save(safe_best_path)
            if mean_improved:
                mean_best_score = mean_score
                scheduler.save(mean_best_path)
            if deploy_improved:
                deploy_best_score = deploy_score
                scheduler.save(deploy_best_path)
            if safe_improved or mean_improved or deploy_improved:
                stale_validations = 0
            else:
                stale_validations += 1
            checkpoint_history.append({
                "episode": episode,
                "safe_improved": safe_improved,
                "mean_improved": mean_improved,
                "deploy_improved": deploy_improved,
                "metrics": metrics,
                "ppo_update": dict(scheduler.last_update_stats),
            })
            print(
                f"validation {episode} | 全清={metrics['all_cleared_episode_rate']:.1%} | "
                f"平均后悔={metrics['mean_regret_vs_baseline_s']:+.1f}s | "
                f"P90后悔={metrics['p90_regret_vs_baseline_s']:+.1f}s | "
                f"最坏后悔={metrics['max_regret_vs_baseline_s']:+.1f}s | "
                f"安全门拦截={metrics['safety_gate_block_rate']:.1%} | "
                f"{'SAVE SAFE ' if safe_improved else ''}"
                f"{'SAVE MEAN' if mean_improved else ''}"
                f"{' SAVE DEPLOY' if deploy_improved else ''}"
                f"{'no improvement' if not (safe_improved or mean_improved or deploy_improved) else ''}"
            )
            if stale_validations >= args.early_stop_patience:
                print(f"连续{stale_validations}次验证无改善，提前停止。")
                stopped_early = True
                break

    scheduler.flush_update()
    scheduler.save()
    safe_evaluation_rows = evaluate_model(
        safe_best_path, eval_cases, baseline_rows, args
    )
    mean_evaluation_rows = evaluate_model(
        mean_best_path, eval_cases, baseline_rows, args
    )
    deploy_evaluation_rows = evaluate_model(
        deploy_best_path, eval_cases, baseline_rows, args
    )
    write_curve(output_dir / "training_curve.csv", training_rows)
    write_curve(output_dir / "baseline_evaluation.csv", baseline_rows)
    write_curve(output_dir / "safe_best_evaluation.csv", safe_evaluation_rows)
    write_curve(output_dir / "mean_best_evaluation.csv", mean_evaluation_rows)
    write_curve(output_dir / "deploy_best_evaluation.csv", deploy_evaluation_rows)
    report = {
        "algorithm": ppo.ALGORITHM_VERSION,
        "problem": args.problem, "seed": args.seed,
        "distribution": args.distribution, "rules": asdict(sim.RULES),
        "distribution_disclosure": (
            "官方未公开案例随机生成器；official_estimate使用圆盘面积均匀、"
            "10..16分层数量、频道无放回、接收半径1000..1500均匀。"
        ),
        "working_model_path": str(model_path.resolve()),
        "safe_best_model_path": str(safe_best_path.resolve()),
        "mean_best_model_path": str(mean_best_path.resolve()),
        "deploy_best_model_path": str(deploy_best_path.resolve()),
        "model_episodes": scheduler.episodes,
        "model_updates": scheduler.updates,
        "stopped_early": stopped_early,
        "train": aggregate(training_rows),
        "baseline_evaluation": aggregate(baseline_rows),
        "safe_best_evaluation": aggregate(safe_evaluation_rows),
        "mean_best_evaluation": aggregate(mean_evaluation_rows),
        "deploy_best_evaluation": aggregate(deploy_evaluation_rows),
        "best_evaluation": aggregate(deploy_evaluation_rows),
        "checkpoint_history": checkpoint_history,
        "wall_time_s": time.perf_counter() - wall_start,
        "parameters": vars(args),
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("\n离线训练完成，安全最佳模型评估：")
    print(json.dumps(report["safe_best_evaluation"], ensure_ascii=False, indent=2))
    print("\n均值最佳模型评估：")
    print(json.dumps(report["mean_best_evaluation"], ensure_ascii=False, indent=2))
    print("\n部署最佳模型评估：")
    print(json.dumps(report["deploy_best_evaluation"], ensure_ascii=False, indent=2))
    print(f"安全最佳模型：{safe_best_path.resolve()}")
    print(f"均值最佳模型：{mean_best_path.resolve()}")
    print(f"部署最佳模型：{deploy_best_path.resolve()}")
    print(f"工作模型：{model_path.resolve()}")
    print(f"报告：{report_path.resolve()}")


if __name__ == "__main__":
    main()
