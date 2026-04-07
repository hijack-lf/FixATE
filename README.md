# Fixate 实验仓库

## 目录约定

| 路径 | 职责 |
|------|------|
| `config/` | 配置：`common.py` 为数据路径与共用训练流程；`glimpse.py` / `attnlrp.py` / `rollout.py` 为各注意力方法的输出目录、超参与 `TrainerConfig`。 |
| `fixate/` | Fixate 框架包（soft prompt 优化）；训练入口在 `fixate/probing_operator/`。 |
| `fixate/probing_operator/` | 各方法训练脚本：`train_att_glimpse_new.py`、`train_att_attnlrp.py`、`train_att_rollout.py`。 |
| `preprocessing/` | 数据预处理；界面图生成见 `preprocessing/generate_interface_iamge.py`。 |

## 运行训练

在仓库根目录执行，并保证根目录在 `PYTHONPATH` 中（脚本已用 `_repo_root.py` 自动加入）：

```bash
cd /path/to/fixate_github
PYTHONPATH=. python fixate/probing_operator/train_att_glimpse_new.py
```

修改实验参数时优先编辑 `config/` 下对应文件，而非在训练脚本内写死路径。
