# Bat-mind Sleep Onset OS · MVP Demo

这个仓库是 Bat-mind「睡前 20 分钟入睡导航」的第一代 MVP 骨架，用来做 **Sleep Onset OS** 的离线模拟和算法验证。

当前状态（2026 V1.0）：

- 有一个完整的 Python 脚本：`sim_run_demo.py`
- 这个脚本会做几件事：
  1. 模拟一条 20 分钟的心率曲线（HR 从 75 bpm 缓慢下降到约 60 bpm）
  2. 使用简单的状态模型把 HR 映射到紧张度场 `x_hat(t) ∈ [0,1]`
  3. 生成一个目标轨迹 `Y*(t)`（从 0.8 慢慢滑到 0.28 的「入睡轨迹」）
  4. 按 60 秒窗口计算误差，并通过 `intensity`（护航强度）做闭环调整
  5. 用规则估计入睡时间 `t_onset_est`
  6. 把一次完整 session 的 CIEU 五元组写到 `logs/cieu_demo.json`

换句话说：这已经是一条「从 HR 波形 → 场状态 → Y*(t) → 控制 → 入睡估计 → CIEU 日志」的完整流水线，只是现在用的是模拟 HR，而不是 Apple Watch 的真实数据。

---

## 目录结构（当前）

- `README.md`：项目说明（你现在看到的这个文件）
- `sim_run_demo.py`：单文件版 MVP Demo（离线模拟 + CIEU 输出）

运行结束后会自动生成：

- `logs/cieu_demo.json`：一次 Sleep Onset episode 的 CIEU 五元组（X, U, Y*, Y, R）

---

## 给程序员 / 合作者：如何本地运行

（下面这段是给真正会用 Python 的人看的，你可以把这一段原样发给对方，让 TA 照着做。）

1. 克隆或下载本仓库：

   ```bash
   git clone https://github.com/liuhaotian2024-prog/bat-mind-sleep-onset-os.git
   cd bat-mind-sleep-onset-os
