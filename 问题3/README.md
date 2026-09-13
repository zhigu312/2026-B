# B题问题3代码说明

本目录保存问题3正式算法及PPO离线实验代码。

## 文件结构

```text
问题3/
├─ ceshi.py
├─ rl_policy.json
└─ ppo_experiment/
   ├─ ceshi_actor_critic.py
   ├─ local_offline_simulator.py
   ├─ train_ppo_offline.py
   └─ plot_ppo_return.py
```

- `ceshi.py`：问题3正式算法与模拟器通信入口。
- `rl_policy.json`：Q-learning辅助策略表，与 `ceshi.py` 保持同目录。
- `ppo_experiment`：安全残差Masked PPO离线训练、验证和绘图代码。

## 运行环境

- Windows 10/11；
- Python 3.9或更高版本；
- 官方模拟器已启动并显示接口就绪；
- 默认接口为 `http://127.0.0.1:2026`。

统一依赖文件位于支撑材料根目录。在 `B题支撑材料` 目录执行：

```powershell
python -m pip install -r requirements.txt
```

## 正式运行

在本目录执行，将 `<ROBOT_ID>` 替换为模拟器使用的机器狗编号：

```powershell
python .\ceshi.py --problem 3 --formal --robot-id '<ROBOT_ID>' --no-show
```

自定义接口地址：

```powershell
python .\ceshi.py --problem 3 --formal `
    --robot-id '<ROBOT_ID>' `
    --base-url 'http://127.0.0.1:2026' `
    --no-show
```

程序默认在本目录的 `runs/时间戳/` 下写入：

- `robot_records.csv`：逐动作位置、频道和检测清除结果；
- `run_summary.json`：清除数量、虚拟时间、单位干扰源时间和阶段统计；
- `robot_path_enhanced.png`：完整运动轨迹图。

## PPO离线实验

```powershell
python .\ppo_experiment\train_ppo_offline.py `
    --problem 3 `
    --episodes 1000 `
    --eval-episodes 100 `
    --validation-episodes 70 `
    --validation-every 100 `
    --early-stop-patience 6 `
    --batch-episodes 16 `
    --report-every 20
```

根据训练报告绘制回报曲线：

```powershell
python .\ppo_experiment\plot_ppo_return.py `
    '.\ppo_experiment\offline_training_v6\problem_3\时间戳\training_report.json'
```

正式运行使用 `--formal` 关闭随机探索和在线策略改写；机器狗编号通过命令行传入。
