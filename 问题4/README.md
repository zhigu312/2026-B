# B题问题4代码说明

本目录保存问题4正式算法、复用模块及策略文件。

## 文件结构

```text
问题4/
├─ ceshi_problem4.py
├─ ceshi.py
└─ rl_policy.json
```

- `ceshi_problem4.py`：问题4方向感知搜索与定位清除入口。
- `ceshi.py`：问题4复用的模拟器接口、几何计算和绘图函数。
- `rl_policy.json`：随公共算法归档的策略文件。

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
python .\ceshi_problem4.py --formal --robot-id '<ROBOT_ID>'
```

生成轨迹图：

```powershell
python .\ceshi_problem4.py --formal --robot-id '<ROBOT_ID>' --plot
```

程序默认在本目录的 `runs_problem_4/时间戳/` 下保存问题4运行记录和摘要；使用 `--plot` 时同时生成轨迹图。

## 算法概要

问题4在问题3几何定位与安全清除框架上加入方向感知搜索、位置—半径—类型—发射方向联合假设集、定向源非对称无信号更新、多尺度补测和长尾预算接管。清除操作继续使用20米安全判据。

`ceshi_problem4.py` 与 `ceshi.py` 需要保持在同一目录。正式运行使用 `--formal`，机器狗编号通过命令行传入。
