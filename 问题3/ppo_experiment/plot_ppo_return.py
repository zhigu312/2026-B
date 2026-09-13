"""从PPO训练报告绘制检查点平均回报曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="training_report.json路径")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    records = [
        item for item in data["checkpoint_history"]
        if item.get("ppo_update") is not None
    ]
    if not records:
        raise SystemExit("报告中没有包含ppo_update的检查点")

    episodes = [item["episode"] for item in records]
    returns = [item["ppo_update"]["mean_return"] for item in records]
    output = args.output or args.report.with_name("ppo_return_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, returns, marker="o", linewidth=2.2,
             color="#2878B5", label="PPO mean return")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    for x, y in zip(episodes, returns):
        plt.annotate(f"{y:.1f}", (x, y), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=8)
    plt.xlabel("Training Episodes")
    plt.ylabel("Mean Discounted Return")
    plt.title("PPO Training Return — Problem 3")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    print(f"图片已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
