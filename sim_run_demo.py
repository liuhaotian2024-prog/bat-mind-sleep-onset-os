"""
Bat-mind Sleep Onset OS · 超简化离线 MVP Demo (单文件版)

功能：
- 模拟一条 20 分钟的 HR 曲线（从 75 bpm 平滑降到 60 bpm）
- 用简单方式把 HR 映射到紧张度 x_hat(t)
- 生成一个目标轨迹 Y*(t)
- 按窗口计算误差并调整强度 intensity（控制逻辑简化）
- 用规则估计入睡时间 t_onset_est
- 打印结果，并把一次 CIEU 五元组写入 logs/cieu_demo.json
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import random
import statistics as stats
import time
import json


# ========== 1. HR -> 状态模型 ==========

@dataclass
class HRStateConfig:
    hr_min: float = 50.0
    hr_max: float = 100.0
    alpha: float = 0.25


class StateModel:
    def __init__(self, config: HRStateConfig | None = None) -> None:
        self.config = config or HRStateConfig()
        self.x_hat: Optional[float] = None

    def hr_to_state(self, hr: float) -> float:
        c = self.config
        hr_clamped = max(c.hr_min, min(c.hr_max, hr))
        x_meas = (hr_clamped - c.hr_min) / (c.hr_max - c.hr_min)
        return x_meas

    def update(self, hr: float) -> float:
        x_meas = self.hr_to_state(hr)
        if self.x_hat is None:
            self.x_hat = x_meas
        else:
            self.x_hat = (1 - self.config.alpha) * self.x_hat + self.config.alpha * x_meas
        return self.x_hat


# ========== 2. 目标轨迹 Y*(t) ==========

@dataclass
class YStarConfig:
    T_sec: float = 1200.0
    Y_start: float = 0.8
    Y_mid: float = 0.6
    Y_end: float = 0.28
    t1: float = 300.0
    t2: float = 900.0


class YStar:
    def __init__(self, config: YStarConfig | None = None) -> None:
        self.config = config or YStarConfig()
        assert 0.0 < self.config.t1 < self.config.t2 <= self.config.T_sec

    def get(self, t: float) -> float:
        c = self.config
        if t <= 0.0:
            return c.Y_start
        if t >= c.T_sec:
            return c.Y_end

        if t <= c.t1:
            ratio = t / c.t1
            return c.Y_start + (c.Y_mid - c.Y_start) * ratio
        elif t <= c.t2:
            ratio = (t - c.t1) / (c.t2 - c.t1)
            return c.Y_mid + (c.Y_end - c.Y_mid) * ratio
        else:
            return c.Y_end


# ========== 3. 控制：waveflow 档位 ==========

@dataclass
class WaveProfile:
    beat_freq: float
    volume: float
    intensity: float
    token: str


@dataclass
class ControlConfig:
    error_up_threshold: float = 0.15
    error_down_threshold: float = 0.08
    intensity_min: float = 0.05
    intensity_max: float = 0.5
    intensity_step: float = 0.05
    beat_base: float = 3.0
    volume_base: float = 0.6


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choose_wave_profile(avg_error: float, current: WaveProfile, cfg: ControlConfig) -> WaveProfile:
    """根据窗口平均误差调整 intensity / beat / volume。"""
    intensity = current.intensity
    if avg_error > cfg.error_up_threshold:
        intensity = clamp(intensity + cfg.intensity_step, cfg.intensity_min, cfg.intensity_max)
    elif avg_error < cfg.error_down_threshold:
        intensity = clamp(intensity - cfg.intensity_step, cfg.intensity_min, cfg.intensity_max)

    rel = (intensity - cfg.intensity_min) / max(1e-6, (cfg.intensity_max - cfg.intensity_min))
    beat_freq = cfg.beat_base + 1.0 * rel
    volume = clamp(cfg.volume_base + 0.2 * rel, 0.0, 1.0)

    return WaveProfile(
        beat_freq=beat_freq,
        volume=volume,
        intensity=intensity,
        token=current.token,
    )


# ========== 4. 入睡时间估计 ==========

@dataclass
class OnsetConfig:
    hr_base_window_sec: float = 180.0   # 前 3 分钟
    hold_window_sec: float = 180.0      # 判定窗口 3 分钟
    hr_drop_ratio: float = 0.9
    error_eps: float = 0.1
    intensity_threshold: float = 0.2


def _window_indices(times: List[float], start_t: float, end_t: float) -> Optional[tuple[int, int]]:
    n = len(times)
    start_idx = None
    end_idx = None
    for i, t in enumerate(times):
        if t >= start_t and start_idx is None:
            start_idx = i
        if t <= end_t:
            end_idx = i
    if start_idx is None or end_idx is None or start_idx > end_idx:
        return None
    return start_idx, end_idx


def estimate_onset_time(
    times: List[float],
    hr_values: List[float],
    x_hat_values: List[float],
    y_star_values: List[float],
    intensity_values: List[float],
    cfg: OnsetConfig,
) -> Optional[float]:
    if not times:
        return None

    base_end_t = cfg.hr_base_window_sec
    base_idx = _window_indices(times, 0.0, base_end_t)
    if not base_idx:
        return None

    b_start, b_end = base_idx
    hr_base = stats.fmean(hr_values[b_start:b_end + 1])

    T_total = times[-1]
    t = base_end_t
    while t + cfg.hold_window_sec <= T_total:
        w_idx = _window_indices(times, t, t + cfg.hold_window_sec)
        if not w_idx:
            break

        w_start, w_end = w_idx
        hr_mean = stats.fmean(hr_values[w_start:w_end + 1])
        err_abs = [abs(x_hat_values[i] - y_star_values[i]) for i in range(w_start, w_end + 1)]
        err_mean = stats.fmean(err_abs)
        intensity_mean = stats.fmean(intensity_values[w_start:w_end + 1])

        hr_ok = hr_mean <= cfg.hr_drop_ratio * hr_base
        err_ok = err_mean < cfg.error_eps
        intensity_ok = intensity_mean <= cfg.intensity_threshold

        if hr_ok and err_ok and intensity_ok:
            return t + cfg.hold_window_sec

        t += 30.0

    return None


# ========== 5. CIEU 简化结构 ==========

@dataclass
class XState:
    session_id: str
    user_id: str
    device: str
    hr_source: str
    session_start_ts: float
    session_duration_sec: float
    HR_base: float
    HR_var_base: float
    x0: float
    hr_min: float
    hr_max: float


@dataclass
class UPolicy:
    wave_token_set: list[str]
    micro_window_sec: float
    initial_profile: dict
    control_params: dict


@dataclass
class YStarParams:
    T_sec: float
    Y_start: float
    Y_mid: float
    Y_end: float
    t1: float
    t2: float


@dataclass
class YOutcome:
    t_onset_est_sec: Optional[float]
    onset_duration_sec: Optional[float]
    HR_drop: Optional[float]
    HR_mean_onset_window: Optional[float]
    session_completed: bool


@dataclass
class RReward:
    R_phys: float
    R_components: dict


# ========== 6. 模拟 HR 序列 ==========

def simulate_hr_series(
    T_sec: float = 1200.0,
    dt: float = 5.0,
    hr_start: float = 75.0,
    hr_end: float = 60.0,
    noise_std: float = 0.8,
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    hr_values: list[float] = []
    n_steps = int(T_sec / dt) + 1
    for i in range(n_steps):
        t = i * dt
        ratio = t / T_sec
        base_hr = hr_start + (hr_end - hr_start) * ratio
        hr = base_hr + random.gauss(0.0, noise_std)
        if hr < 45:
            hr = 45.0
        times.append(t)
        hr_values.append(hr)
    return times, hr_values


# ========== 7. 运行一次模拟并写 CIEU JSON ==========

def main() -> None:
    # 0. 配置
    T_sec = 1200.0
    dt = 5.0

    # 1. 模拟 HR
    times, hr_values = simulate_hr_series(T_sec=T_sec, dt=dt)

    # 2. 初始化状态 & 目标
    hr_cfg = HRStateConfig()
    state_model = StateModel(hr_cfg)
    ystar_cfg = YStarConfig(T_sec=T_sec)
    ystar = YStar(ystar_cfg)

    x_hat_values: list[float] = []
    y_star_values: list[float] = []
    intensity_values: list[float] = []

    # 初始 wave profile
    ctrl_cfg = ControlConfig()
    current_profile = WaveProfile(
        beat_freq=3.5,
        volume=0.7,
        intensity=0.4,
        token="binaural_breathing_v1",
    )

    window_sec = 60.0
    window_samples = int(window_sec / dt)

    # 3. 逐步模拟
    for idx, (t, hr) in enumerate(zip(times, hr_values)):
        x_hat = state_model.update(hr)
        y_star_t = ystar.get(t)

        x_hat_values.append(x_hat)
        y_star_values.append(y_star_t)
        intensity_values.append(current_profile.intensity)

        if idx > 0 and idx % window_samples == 0:
            start_idx = max(0, idx - window_samples)
            win_x = x_hat_values[start_idx:idx]
            win_y = y_star_values[start_idx:idx]
            errors = [xh - ys for xh, ys in zip(win_x, win_y)]
            avg_error = sum(errors) / len(errors)
            current_profile = choose_wave_profile(avg_error, current_profile, ctrl_cfg)

    # 4. 入睡时间估计
    onset_cfg = OnsetConfig()
    t_onset = estimate_onset_time(
        times=times,
        hr_values=hr_values,
        x_hat_values=x_hat_values,
        y_star_values=y_star_values,
        intensity_values=intensity_values,
        cfg=onset_cfg,
    )

    if t_onset is not None:
        print(f"Estimated onset time: {t_onset:.1f} s (~{t_onset/60:.1f} min)")
    else:
        print("No stable onset detected.")

    # 5. 基础 HR 统计
    base_end_t = onset_cfg.hr_base_window_sec
    base_idx = _window_indices(times, 0.0, base_end_t)
    if base_idx:
        b_start, b_end = base_idx
        base_hr_window = hr_values[b_start:b_end + 1]
        HR_base = stats.fmean(base_hr_window)
        HR_var_base = stats.pvariance(base_hr_window, mu=HR_base) if len(base_hr_window) > 1 else 0.0
    else:
        HR_base = 0.0
        HR_var_base = 0.0

    x0 = x_hat_values[0] if x_hat_values else 0.0

    # 6. 组装 CIEU
    session_id = "demo-session-001"
    X = XState(
        session_id=session_id,
        user_id="local-demo-user",
        device="offline-sim",
        hr_source="simulated",
        session_start_ts=time.time(),
        session_duration_sec=T_sec,
        HR_base=HR_base,
        HR_var_base=HR_var_base,
        x0=x0,
        hr_min=hr_cfg.hr_min,
        hr_max=hr_cfg.hr_max,
    )

    U = UPolicy(
        wave_token_set=[current_profile.token],
        micro_window_sec=window_sec,
        initial_profile={
            "beat_freq": 3.5,
            "volume": 0.7,
            "intensity": 0.4,
            "token": "binaural_breathing_v1",
        },
        control_params={
            "error_up_threshold": ctrl_cfg.error_up_threshold,
            "error_down_threshold": ctrl_cfg.error_down_threshold,
            "intensity_min": ctrl_cfg.intensity_min,
            "intensity_max": ctrl_cfg.intensity_max,
            "intensity_step": ctrl_cfg.intensity_step,
        },
    )

    if t_onset is not None:
        onset_duration = t_onset
        onset_start_t = max(0.0, t_onset - onset_cfg.hold_window_sec)
        onset_end_t = t_onset
        w_idx = _window_indices(times, onset_start_t, onset_end_t)
        if w_idx:
            s, e = w_idx
            onset_hr_window = hr_values[s:e + 1]
            HR_mean_onset = stats.fmean(onset_hr_window)
        else:
            HR_mean_onset = None
        HR_drop = HR_base - HR_mean_onset if HR_mean_onset is not None else None
    else:
        onset_duration = None
        HR_mean_onset = None
        HR_drop = None

    Y_outcome = YOutcome(
        t_onset_est_sec=t_onset,
        onset_duration_sec=onset_duration,
        HR_drop=HR_drop,
        HR_mean_onset_window=HR_mean_onset,
        session_completed=True,
    )

    mse = sum((xh - ys) ** 2 for xh, ys in zip(x_hat_values, y_star_values)) / len(x_hat_values)
    avg_intensity = sum(intensity_values) / len(intensity_values)
    if onset_duration is not None:
        onset_penalty = onset_duration / 60.0
    else:
        onset_penalty = 30.0
    R_phys = -mse - 0.1 * onset_penalty - 0.5 * avg_intensity

    R_components = {
        "traj_mse": mse,
        "onset_duration_sec": onset_duration,
        "HR_drop": HR_drop,
        "avg_intensity": avg_intensity,
    }
    R = RReward(R_phys=R_phys, R_components=R_components)

    Y_star_params = YStarParams(
        T_sec=ystar_cfg.T_sec,
        Y_start=ystar_cfg.Y_start,
        Y_mid=ystar_cfg.Y_mid,
        Y_end=ystar_cfg.Y_end,
        t1=ystar_cfg.t1,
        t2=ystar_cfg.t2,
    )

    cieu_obj = {
        "CIEU_version": "1.0",
        "session_id": session_id,
        "X": asdict(X),
        "U": asdict(U),
        "Y_star": asdict(Y_star_params),
        "Y": asdict(Y_outcome),
        "R": asdict(R),
    }

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "cieu_demo.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cieu_obj, f, ensure_ascii=False, indent=2)

    print(f"CIEU JSON written to: {out_path}")


if __name__ == "__main__":
    main()
