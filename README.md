# 2026 数学建模 B 题代码

本仓库整理了 B 题问题 1—4 的建模、数值实验、机器狗搜索清除算法，以及问题 3 的 PPO 离线训练实验代码。

## 创作团队

于汶琳、汤家豪、杜元润

## 环境准备

建议使用 Python 3.10 及以上版本。在仓库根目录安装绘图与数值计算依赖：

```bash
python -m pip install numpy matplotlib shapely
```

问题 3、问题 4 的在线测试需要先启动官方机器狗模拟器，并取得模拟器分配的 `ROBOT_ID`。仓库中未保存参赛团队号，运行时必须通过 `--robot-id` 传入。

## 目录与运行方法

### 问题 1

```powershell
python ".\问题1\problem1_geometric_localization.py"
```

### 问题 2

问题 2 包含保证接收候选圆、定位性能空间分布、Pareto 前沿和推荐点定位结果四组实验。进入对应实验目录后运行其中的 Python 文件，例如：

```powershell
python ".\问题2\实验1_保证接受候选圆\experiment1_guaranteed_reception_circle.py"
python ".\问题2\实验2_定位性能空间分布\experiment2_worst_case_heatmap.py"
python ".\问题2\实验3_Pareto前沿\experiment3_pareto_front.py"
python ".\问题2\实验4_推荐点定位结果\experiment4_recommended_point_result.py"
```

### 问题 3

在 `问题3` 目录运行正式策略：

```powershell
cd ".\问题3"
python .\ceshi.py --problem 3 --formal --robot-id "<ROBOT_ID>" --no-show
```

问题 3 的 PPO 离线训练与回报绘图代码位于 `问题3/ppo_experiment`。常用训练命令：

```powershell
cd ".\问题3\ppo_experiment"
python .\train_ppo_offline.py --problem 3 --episodes 1000 --eval-episodes 100
```

具体参数见该目录脚本的 `--help` 输出。

### 问题 4

在 `问题4` 目录运行正式策略：

```powershell
cd ".\问题4"
python .\ceshi_problem4.py --formal --robot-id "<ROBOT_ID>"
```

需要同时生成轨迹图时添加 `--plot`：

```powershell
python .\ceshi_problem4.py --formal --robot-id "<ROBOT_ID>" --plot
```

问题 3、问题 4 的每次运行都会保存 CSV/JSON 日志；详细参数和输出说明分别见对应目录下的 `README.md`。

## 注意事项

- 请勿把真实团队号、账号凭据或正式比赛密钥提交到公开仓库。
- `rl_policy.json` 是算法策略参数文件，运行问题 3、问题 4 时请与主程序保持在同一目录。
- 正式测试建议使用 `--formal`，关闭随机探索并避免测试过程改写策略。
