#! /usr/bin/env python3

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from audio_utils_msgs.msg import AudioFrame
from odas_ros_msgs.msg import OdasSstArrayStamped
from std_msgs.msg import Bool, Int32


LOW_BAND = (200.0, 2000.0)
HIGH_BAND = (2000.0, 8000.0)
VOICE_BAND = (300.0, 3400.0)
NON_VOICE_LOW = (50.0, 300.0)
NON_VOICE_HIGH = (3400.0, 8000.0)
MID_HIGH_BAND = (1000.0, 2000.0)
UPPER_HIGH_BAND = (4000.0, 8000.0)


def azimuth_deg_from_xy(x: float, y: float) -> float:
    az = math.degrees(math.atan2(y, x))
    return az + 360.0 if az < 0.0 else az


def calibrated_doa_deg(raw_az_deg: float, zero_offset_deg: float) -> float:
    return (raw_az_deg - zero_offset_deg) % 360.0


def rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))


def compute_zcr(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def compute_hi_lo_ratio_db(psd: np.ndarray, freqs: np.ndarray) -> float:
    lo_idx = (freqs >= LOW_BAND[0]) & (freqs < LOW_BAND[1])
    hi_idx = (freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])
    e_lo = float(np.sum(psd[lo_idx])) + 1e-12
    e_hi = float(np.sum(psd[hi_idx])) + 1e-12
    return 10.0 * math.log10(e_hi / e_lo)


def compute_voice_ratio_db(psd: np.ndarray, freqs: np.ndarray) -> float:
    voice_idx = (freqs >= VOICE_BAND[0]) & (freqs < VOICE_BAND[1])
    nv_lo_idx = (freqs >= NON_VOICE_LOW[0]) & (freqs < NON_VOICE_LOW[1])
    nv_hi_idx = (freqs >= NON_VOICE_HIGH[0]) & (freqs <= NON_VOICE_HIGH[1])
    e_voice = float(np.sum(psd[voice_idx])) + 1e-12
    e_nonvoice = float(np.sum(psd[nv_lo_idx]) + np.sum(psd[nv_hi_idx])) + 1e-12
    return 10.0 * math.log10(e_voice / e_nonvoice)


def compute_upper_high_ratio_db(psd: np.ndarray, freqs: np.ndarray) -> float:
    mid_idx = (freqs >= MID_HIGH_BAND[0]) & (freqs < MID_HIGH_BAND[1])
    upper_idx = (freqs >= UPPER_HIGH_BAND[0]) & (freqs <= UPPER_HIGH_BAND[1])
    e_mid = float(np.sum(psd[mid_idx])) + 1e-12
    e_upper = float(np.sum(psd[upper_idx])) + 1e-12
    return 10.0 * math.log10(e_upper / e_mid)


def compute_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    psd = psd + 1e-12
    return float(np.sum(freqs * psd) / np.sum(psd))


def compute_flux(psd_curr: np.ndarray, psd_prev: Optional[np.ndarray]) -> float:
    if psd_prev is None:
        return 0.0
    p1 = psd_prev / (np.sum(psd_prev) + 1e-12)
    p2 = psd_curr / (np.sum(psd_curr) + 1e-12)
    return float(np.sqrt(np.sum((p2 - p1) ** 2)))


def compute_features(x_int16: np.ndarray, fs: int, psd_prev: Optional[np.ndarray]) -> Dict[str, float]:
    x = x_int16.astype(np.float32)
    x -= float(np.mean(x))
    zcr = compute_zcr(x)

    x[1:] = x[1:] - 0.97 * x[:-1]
    x *= np.hanning(x.size).astype(np.float32)

    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2).astype(np.float32) + 1e-12
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)

    return {
        "hi_lo_ratio_db": compute_hi_lo_ratio_db(psd, freqs),
        "voice_ratio_db": compute_voice_ratio_db(psd, freqs),
        "upper_high_ratio_db": compute_upper_high_ratio_db(psd, freqs),
        "centroid_hz": compute_centroid(psd, freqs),
        "zcr": zcr,
        "flux": compute_flux(psd, psd_prev),
        "psd": psd,
    }


class FeatureHistory:
    KEYS = ("hi_lo_ratio_db", "voice_ratio_db", "upper_high_ratio_db", "centroid_hz", "zcr", "flux")

    def __init__(self, window_sec: float, update_hz: float):
        self.max_len = max(1, int(window_sec * update_hz))
        self.history: Dict[str, List[float]] = {k: [] for k in self.KEYS}
        self.last_track_id: Optional[int] = None

    def update(self, feats: Dict[str, float], track_id: int) -> Dict[str, float]:
        if self.last_track_id != track_id:
            self.clear()
            self.last_track_id = track_id

        smoothed = {}
        for key in self.KEYS:
            self.history[key].append(float(feats[key]))
            if len(self.history[key]) > self.max_len:
                self.history[key].pop(0)
            smoothed[key] = float(np.median(self.history[key]))
        return smoothed

    def clear(self):
        for key in self.KEYS:
            self.history[key] = []


class RingBuffer:
    def __init__(self, n_channels: int, max_samples: int):
        self.n_channels = int(n_channels)
        self.max_samples = int(max_samples)
        self.buf = np.zeros((self.max_samples, self.n_channels), dtype=np.int16)
        self.idx = 0
        self.full = False

    def write(self, samples: np.ndarray):
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            return

        n = samples.shape[0]
        if n >= self.max_samples:
            self.buf[:, :] = samples[-self.max_samples:, :]
            self.idx = 0
            self.full = True
            return

        end = self.idx + n
        if end <= self.max_samples:
            self.buf[self.idx:end, :] = samples
        else:
            first = self.max_samples - self.idx
            self.buf[self.idx:, :] = samples[:first, :]
            self.buf[: end - self.max_samples, :] = samples[first:, :]

        self.idx = end % self.max_samples
        if self.idx == 0:
            self.full = True

    def read_latest(self, n: int) -> np.ndarray:
        n = min(int(n), self.max_samples)
        if not self.full and self.idx < n:
            return np.zeros((0, self.n_channels), dtype=np.int16)

        start = (self.idx - n) % self.max_samples
        if start < self.idx:
            return self.buf[start:self.idx, :].copy()
        return np.vstack((self.buf[start:, :], self.buf[:self.idx, :]))


@dataclass
class LeakLatch:
    leak_on: bool = False
    on_count: int = 0
    off_count: int = 0
    latched_key: Optional[Tuple[int, int]] = None
    latched_raw_az: Optional[float] = None
    latched_doa: Optional[float] = None
    grace_mode: bool = False
    contested_hold_count: int = 0


class OdasLeakClassifierNode(Node):
    def __init__(self):
        super().__init__("odas_leak_classifier_node")

        self._sst_topic = self.declare_parameter("sst_topic", "sst_raw").get_parameter_value().string_value
        self._sss_topic = self.declare_parameter("sss_topic", "sss").get_parameter_value().string_value
        self._leak_topic = self.declare_parameter("leak_topic", "/leak_detected").get_parameter_value().string_value
        self._doa_topic = self.declare_parameter("doa_topic", "/doa_angle").get_parameter_value().string_value
        self._doa_zero_offset_deg = self.declare_parameter("doa_zero_offset_deg", 0.0).get_parameter_value().double_value
        self._print_hz = self.declare_parameter("print_hz", 5.0).get_parameter_value().double_value
        self._fs = self.declare_parameter("sampling_frequency", 16000).get_parameter_value().integer_value
        self._win_sec = self.declare_parameter("window_sec", 0.50).get_parameter_value().double_value
        self._median_sec = self.declare_parameter("median_sec", 1.0).get_parameter_value().double_value

        self._min_activity = self.declare_parameter("min_activity", 0.40).get_parameter_value().double_value
        self._min_rms = self.declare_parameter("min_rms", 70.0).get_parameter_value().double_value
        self._min_dir_eps = self.declare_parameter("min_dir_eps", 1e-6).get_parameter_value().double_value

        self._hi_lo_on = self.declare_parameter("hi_lo_on", 3.0).get_parameter_value().double_value
        self._voice_on = self.declare_parameter("voice_on", -2.0).get_parameter_value().double_value
        self._hi_lo_latch = self.declare_parameter("hi_lo_latch", 8.0).get_parameter_value().double_value
        self._voice_latch = self.declare_parameter("voice_latch", -5.0).get_parameter_value().double_value
        self._hi_lo_off = self.declare_parameter("hi_lo_off", 0.0).get_parameter_value().double_value
        self._voice_off = self.declare_parameter("voice_off", 0.0).get_parameter_value().double_value
        self._voice_relaxed_on = self.declare_parameter("voice_relaxed_on", 1.0).get_parameter_value().double_value
        self._strict_mode = self.declare_parameter("strict_mode", True).get_parameter_value().bool_value

        self._hold_on = self.declare_parameter("hold_on", 3).get_parameter_value().integer_value
        self._hold_off = self.declare_parameter("hold_off", 6).get_parameter_value().integer_value
        self._contested_voice_max = self.declare_parameter("contested_voice_max_db", 1.0).get_parameter_value().double_value
        self._max_contested_hold_ticks = self.declare_parameter("max_contested_hold_ticks", 2).get_parameter_value().integer_value

        self._max_flux_for_latch = self.declare_parameter("max_flux_for_latch", 0.10).get_parameter_value().double_value
        self._hi_lo_very_strong = self.declare_parameter("hi_lo_very_strong", 15.0).get_parameter_value().double_value
        self._voice_skip_flux = self.declare_parameter("voice_skip_flux", -8.0).get_parameter_value().double_value

        self._publish_debug = self.declare_parameter("publish_debug_logs", True).get_parameter_value().bool_value

        self._latch = LeakLatch()
        self._feat_histories: Dict[int, FeatureHistory] = defaultdict(
            lambda: FeatureHistory(self._median_sec, self._print_hz)
        )
        self._prev_psd: Dict[int, np.ndarray] = {}
        self._prev_track_id: Dict[int, int] = {}
        self._acq_counts: Dict[Tuple[int, int], int] = {}

        self._last_sst_sources: List = []
        self._ring: Optional[RingBuffer] = None
        self._sss_channel_count: Optional[int] = None

        self._leak_pub = self.create_publisher(Bool, self._leak_topic, 10)
        self._doa_pub = self.create_publisher(Int32, self._doa_topic, 10)

        self.create_subscription(OdasSstArrayStamped, self._sst_topic, self._sst_cb, 10)
        self.create_subscription(AudioFrame, self._sss_topic, self._sss_cb, 20)

        dt = 1.0 / max(0.5, float(self._print_hz))
        self.create_timer(dt, self._on_timer)

        self.get_logger().info(
            f"Leak classifier started: sst_topic={self._sst_topic}, sss_topic={self._sss_topic}, "
            f"leak_topic={self._leak_topic}, doa_topic={self._doa_topic}, "
            f"doa_zero_offset_deg={self._doa_zero_offset_deg:.1f}, hz={self._print_hz:.2f}"
        )

    def _sst_cb(self, msg: OdasSstArrayStamped):
        self._last_sst_sources = list(msg.sources)

    def _sss_cb(self, msg: AudioFrame):
        if msg.format != "signed_16":
            self.get_logger().warning(f"Unsupported sss format '{msg.format}', expected 'signed_16'")
            return

        expected = int(msg.frame_sample_count) * int(msg.channel_count) * 2
        if len(msg.data) != expected:
            self.get_logger().warning(
                f"Dropped malformed sss frame: bytes={len(msg.data)}, expected={expected}, "
                f"samples={msg.frame_sample_count}, channels={msg.channel_count}"
            )
            return

        if self._sss_channel_count != int(msg.channel_count):
            self._sss_channel_count = int(msg.channel_count)
            max_samples = int(self._fs * max(2.0, self._win_sec * 3.0))
            self._ring = RingBuffer(self._sss_channel_count, max_samples)
            self._prev_psd.clear()
            self._prev_track_id.clear()
            self._feat_histories.clear()
            self._acq_counts.clear()
            self.get_logger().info(f"Configured audio ring buffer: channels={self._sss_channel_count}, max_samples={max_samples}")

        if msg.sampling_frequency > 0:
            self._fs = int(msg.sampling_frequency)

        frames = np.frombuffer(msg.data, dtype=np.int16).reshape((int(msg.frame_sample_count), int(msg.channel_count)))
        if self._ring is not None:
            self._ring.write(frames)

    def _publish_outputs(self):
        leak_msg = Bool()
        leak_msg.data = bool(self._latch.leak_on)
        self._leak_pub.publish(leak_msg)

        if self._latch.leak_on and self._latch.latched_doa is not None:
            doa_msg = Int32()
            doa_msg.data = int(round(self._latch.latched_doa)) % 360
            self._doa_pub.publish(doa_msg)

    def _log_debug(self, text: str):
        if self._publish_debug:
            self.get_logger().info(text)

    def _raw_and_calibrated_doa(self, x: float, y: float) -> Tuple[float, float]:
        raw_az = azimuth_deg_from_xy(x, y)
        doa = calibrated_doa_deg(raw_az, self._doa_zero_offset_deg)
        return raw_az, doa

    def _reset_latch(self, reason: str):
        old_key = self._latch.latched_key
        old_raw_az = self._latch.latched_raw_az
        old_doa = self._latch.latched_doa
        self._latch = LeakLatch()
        if old_key is None:
            self._log_debug(f"[LEAK=0] {reason}")
        else:
            self._log_debug(
                f"[LEAK=0] {reason} id={old_key[0]} ch={old_key[1]} "
                f"raw_az={old_raw_az} doa={old_doa}"
            )

    def _on_timer(self):
        if self._ring is None:
            return

        win_n = int(self._fs * self._win_sec)
        if win_n <= 0:
            return

        frames = self._ring.read_latest(win_n)
        if frames.shape[0] < win_n:
            return

        src_list = list(self._last_sst_sources)
        n_chan = frames.shape[1]
        if not src_list:
            self._publish_outputs()
            return

        track_map: Dict[Tuple[int, int], Tuple[object, float, float, Dict[str, float]]] = {}
        for ch in range(min(n_chan, len(src_list))):
            src = src_list[ch]
            try:
                track_id = int(src.id)
                act = float(src.activity)
                x = float(src.x)
                y = float(src.y)
            except Exception:
                continue

            if act < self._min_activity:
                continue
            if abs(x) < self._min_dir_eps and abs(y) < self._min_dir_eps:
                continue

            sig = frames[:, ch]
            r = rms_int16(sig)
            if r < self._min_rms:
                continue

            if ch in self._prev_track_id and self._prev_track_id[ch] != track_id:
                self._prev_psd.pop(ch, None)
            self._prev_track_id[ch] = track_id

            feats_raw = compute_features(sig, self._fs, self._prev_psd.get(ch))
            self._prev_psd[ch] = feats_raw["psd"]
            feats = self._feat_histories[ch].update(feats_raw, track_id)
            track_map[(track_id, ch)] = (src, act, r, feats)

        if self._latch.leak_on and self._latch.latched_key is not None:
            if self._latch.latched_key in track_map:
                src, act, r, feats = track_map[self._latch.latched_key]
                track_id, ch = self._latch.latched_key
                x = float(src.x)
                y = float(src.y)
                raw_az, doa = self._raw_and_calibrated_doa(x, y)
                self._latch.latched_raw_az = raw_az
                self._latch.latched_doa = doa
                self._latch.grace_mode = False

                if self._strict_mode:
                    on_cond = (feats["hi_lo_ratio_db"] >= self._hi_lo_on) and (feats["voice_ratio_db"] <= self._voice_on)
                else:
                    on_cond = (feats["hi_lo_ratio_db"] >= self._hi_lo_on) and (
                        feats["voice_ratio_db"] <= self._voice_relaxed_on
                    )
                off_cond = (feats["hi_lo_ratio_db"] <= self._hi_lo_off) or (feats["voice_ratio_db"] >= self._voice_off)
                contested = (
                    (feats["hi_lo_ratio_db"] > self._hi_lo_off)
                    and (feats["voice_ratio_db"] >= self._voice_off)
                    and (feats["voice_ratio_db"] < self._contested_voice_max)
                )

                if on_cond:
                    self._latch.on_count += 1
                    self._latch.off_count = 0
                    self._latch.contested_hold_count = 0
                elif off_cond:
                    self._latch.off_count += 1
                    self._latch.on_count = 0
                    self._latch.contested_hold_count = 0
                elif contested:
                    if self._latch.contested_hold_count < self._max_contested_hold_ticks:
                        self._latch.contested_hold_count += 1
                    else:
                        self._latch.off_count += 1
                    self._latch.on_count = max(0, self._latch.on_count - 1)
                else:
                    self._latch.on_count = max(0, self._latch.on_count - 1)
                    self._latch.off_count = max(0, self._latch.off_count - 1)
                    self._latch.contested_hold_count = 0

                if self._latch.off_count >= self._hold_off:
                    self._reset_latch("OFF")
                else:
                    self._log_debug(
                        f"[LEAK=1] TRACK ch={ch} id={track_id} raw_az={raw_az:.1f} doa={doa:.1f} "
                        f"act={act:.2f} rms={r:.0f} "
                        f"hi_lo={feats['hi_lo_ratio_db']:+.1f}dB voice={feats['voice_ratio_db']:+.1f}dB "
                        f"flux={feats['flux']:.3f}"
                    )
            else:
                latched_id, latched_ch = self._latch.latched_key
                alt_same_id = next((k for k in track_map.keys() if k[0] == latched_id), None)
                if alt_same_id is not None:
                    new_src = track_map[alt_same_id][0]
                    self._latch.latched_key = alt_same_id
                    raw_az, doa = self._raw_and_calibrated_doa(float(new_src.x), float(new_src.y))
                    self._latch.latched_raw_az = raw_az
                    self._latch.latched_doa = doa
                    self._latch.off_count = 0
                    self._latch.grace_mode = False
                    self._log_debug(
                        f"[LEAK=1] MIGRATE id={latched_id} ch={latched_ch}->{alt_same_id[1]} "
                        f"raw_az={raw_az:.1f} doa={doa:.1f}"
                    )
                else:
                    self._latch.grace_mode = True
                    self._latch.off_count += 1
                    self._latch.on_count = 0
                    if self._latch.off_count >= self._hold_off:
                        self._reset_latch("LOST")
                    else:
                        raw_az = self._latch.latched_raw_az if self._latch.latched_raw_az is not None else float("nan")
                        doa = self._latch.latched_doa if self._latch.latched_doa is not None else float("nan")
                        self._log_debug(
                            f"[LEAK=1] GRACE id={self._latch.latched_key[0]} ch={self._latch.latched_key[1]} "
                            f"raw_az={raw_az:.1f} doa={doa:.1f} off_cnt={self._latch.off_count}/{self._hold_off}"
                        )
        else:
            if not track_map:
                for key in list(self._acq_counts.keys()):
                    self._acq_counts[key] = max(0, self._acq_counts[key] - 1)
                    if self._acq_counts[key] == 0:
                        del self._acq_counts[key]
                self._publish_outputs()
                return

            best_key: Optional[Tuple[int, int]] = None
            best_data: Optional[Tuple[object, float, float, Dict[str, float]]] = None
            best_rank: Optional[Tuple[int, float, float]] = None

            for key, (src, act, r, feats) in track_map.items():
                if self._strict_mode:
                    latch_ok = (feats["hi_lo_ratio_db"] >= self._hi_lo_latch) and (feats["voice_ratio_db"] <= self._voice_latch)
                else:
                    latch_ok = (feats["hi_lo_ratio_db"] >= self._hi_lo_latch) and (
                        feats["voice_ratio_db"] <= self._voice_relaxed_on
                    )

                skip_flux = (feats["hi_lo_ratio_db"] >= self._hi_lo_very_strong) or (
                    feats["voice_ratio_db"] <= self._voice_skip_flux
                )
                if not skip_flux and feats["flux"] > self._max_flux_for_latch:
                    latch_ok = False

                contested = (
                    (feats["hi_lo_ratio_db"] > self._hi_lo_off)
                    and (feats["voice_ratio_db"] >= self._voice_off)
                    and (feats["voice_ratio_db"] < self._contested_voice_max)
                )

                if latch_ok:
                    self._acq_counts[key] = min(self._hold_on, self._acq_counts.get(key, 0) + 1)
                elif contested:
                    self._acq_counts[key] = self._acq_counts.get(key, 0)
                else:
                    self._acq_counts[key] = max(0, self._acq_counts.get(key, 0) - 1)

                rank = (self._acq_counts.get(key, 0), feats["hi_lo_ratio_db"], -feats["voice_ratio_db"])
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_key = key
                    best_data = (src, act, r, feats)

            for key in list(self._acq_counts.keys()):
                if key not in track_map:
                    self._acq_counts[key] = max(0, self._acq_counts[key] - 1)
                    if self._acq_counts[key] == 0:
                        del self._acq_counts[key]

            if best_key is not None and best_data is not None:
                track_id, ch = best_key
                src, act, r, feats = best_data
                raw_az, doa = self._raw_and_calibrated_doa(float(src.x), float(src.y))
                on_cnt = self._acq_counts.get(best_key, 0)
                if on_cnt >= self._hold_on:
                    self._latch.leak_on = True
                    self._latch.latched_key = best_key
                    self._latch.latched_raw_az = raw_az
                    self._latch.latched_doa = doa
                    self._latch.on_count = on_cnt
                    self._latch.off_count = 0
                    self._latch.grace_mode = False
                    self._latch.contested_hold_count = 0
                    self._acq_counts.clear()
                    self._log_debug(
                        f"[LEAK=1] LATCH ch={ch} id={track_id} raw_az={raw_az:.1f} doa={doa:.1f} "
                        f"act={act:.2f} rms={r:.0f} "
                        f"hi_lo={feats['hi_lo_ratio_db']:+.1f}dB voice={feats['voice_ratio_db']:+.1f}dB "
                        f"flux={feats['flux']:.3f}"
                    )
                else:
                    self._log_debug(
                        f"[LEAK=0] ACQUIRE ch={ch} id={track_id} raw_az={raw_az:.1f} doa={doa:.1f} "
                        f"on_cnt={on_cnt}/{self._hold_on} "
                        f"act={act:.2f} rms={r:.0f} hi_lo={feats['hi_lo_ratio_db']:+.1f}dB "
                        f"voice={feats['voice_ratio_db']:+.1f}dB flux={feats['flux']:.3f}"
                    )

        self._publish_outputs()


def main():
    rclpy.init()
    node = OdasLeakClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
