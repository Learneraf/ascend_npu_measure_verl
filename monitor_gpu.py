#!/usr/bin/env python3
"""
1s 间隔秒级监控：per-GPU util/mem/power/temp + CPU util + RAM
支持平台：NVIDIA CUDA、海光 DCU (ROCm)、昇腾 NPU、昆仑芯 XPU、沐曦 MACA
用法（由 run_qwen3_8b_a100.sh 自动启动）：
  python3 monitor_gpu.py --output monitor.csv --interval 1
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

# ─── 跨平台 GPU 采集层 ────────────────────────────────────────────────────────

class GPUSample:
    """单个 GPU 在某一时刻的采样值。缺失字段为 None。"""
    __slots__ = ("util_pct", "mem_used_gb", "mem_total_gb", "power_w", "temp_c")

    def __init__(self, util_pct=None, mem_used_gb=None, mem_total_gb=None,
                 power_w=None, temp_c=None):
        self.util_pct    = util_pct
        self.mem_used_gb = mem_used_gb
        self.mem_total_gb = mem_total_gb
        self.power_w     = power_w
        self.temp_c      = temp_c


class BaseMonitor:
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids

    def sample(self) -> dict[int, GPUSample]:
        """返回 {gpu_id: GPUSample}"""
        raise NotImplementedError

    def close(self):
        pass


# ── NVIDIA pynvml ─────────────────────────────────────────────────────────────

class NVMLMonitor(BaseMonitor):
    def __init__(self, gpu_ids):
        import pynvml
        super().__init__(gpu_ids)
        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids}

    def sample(self):
        results = {}
        for gid, hdl in self._handles.items():
            try:
                util  = self._pynvml.nvmlDeviceGetUtilizationRates(hdl)
                mem   = self._pynvml.nvmlDeviceGetMemoryInfo(hdl)
                power = self._pynvml.nvmlDeviceGetPowerUsage(hdl) / 1000.0
                temp  = self._pynvml.nvmlDeviceGetTemperature(hdl, self._pynvml.NVML_TEMPERATURE_GPU)
                results[gid] = GPUSample(
                    util_pct=util.gpu,
                    mem_used_gb=round(mem.used / 1024**3, 2),
                    mem_total_gb=round(mem.total / 1024**3, 2),
                    power_w=round(power, 1),
                    temp_c=temp,
                )
            except Exception:
                results[gid] = GPUSample()
        return results

    def close(self):
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


# ── 海光 DCU：rocm-smi ────────────────────────────────────────────────────────

class ROCMSMIMonitor(BaseMonitor):
    """
    海光 DCU 使用 ROCm 的 rocm-smi 工具。
    支持 rocm-smi (旧) 和 amd-smi (新) 两种接口。
    """
    # rocm-smi --showuse --showmeminfo vram --showpower --showtemp --json
    _RE_USE  = re.compile(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", re.I)
    _RE_MEM  = re.compile(r"GPU\[(\d+)\].*?VRAM Total Memory.*?:\s*(\d+).*?VRAM Total Used.*?:\s*(\d+)", re.I | re.S)
    _RE_POW  = re.compile(r"GPU\[(\d+)\].*?Average Graphics Package Power.*?:\s*([\d.]+)", re.I)
    _RE_TEMP = re.compile(r"GPU\[(\d+)\].*?Temperature.*?:\s*([\d.]+)", re.I)

    def __init__(self, gpu_ids):
        super().__init__(gpu_ids)
        # detect amd-smi or rocm-smi
        for cmd in ("amd-smi", "rocm-smi"):
            try:
                r = subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
                if r.returncode == 0:
                    self._smi = cmd
                    break
            except FileNotFoundError:
                continue
        else:
            self._smi = "rocm-smi"

    def _run_smi(self, *flags):
        try:
            r = subprocess.run([self._smi] + list(flags), capture_output=True,
                               text=True, timeout=5)
            return r.stdout
        except Exception:
            return ""

    def sample(self):
        # Use JSON output if available
        out = self._run_smi("--showuse", "--showmeminfo", "vram",
                            "--showpower", "--showtemp", "--json")
        results = {i: GPUSample() for i in self.gpu_ids}
        if out:
            try:
                data = json.loads(out)
                for card_key, card_val in data.items():
                    m = re.search(r"(\d+)", card_key)
                    if not m:
                        continue
                    gid = int(m.group(1))
                    if gid not in results:
                        continue
                    s = GPUSample()
                    # GPU use
                    for k, v in card_val.items():
                        k_low = k.lower()
                        if "gpu use" in k_low:
                            try:
                                s.util_pct = float(str(v).strip("%"))
                            except Exception:
                                pass
                        elif "vram total memory" in k_low:
                            try:
                                s.mem_total_gb = round(int(v) / 1024, 2)
                            except Exception:
                                pass
                        elif "vram total used" in k_low:
                            try:
                                s.mem_used_gb = round(int(v) / 1024, 2)
                            except Exception:
                                pass
                        elif "average graphics" in k_low or "power" in k_low:
                            try:
                                s.power_w = float(str(v).split()[0])
                            except Exception:
                                pass
                        elif "temperature" in k_low:
                            try:
                                s.temp_c = float(str(v).split()[0])
                            except Exception:
                                pass
                    results[gid] = s
                return results
            except json.JSONDecodeError:
                pass

        # Fallback: plain text
        out = self._run_smi("--showuse", "--showmeminfo", "vram",
                            "--showpower", "--showtemp")
        for m in self._RE_USE.finditer(out):
            gid = int(m.group(1))
            if gid in results:
                results[gid].util_pct = float(m.group(2))
        return results

    def close(self):
        pass


# ── 昇腾 NPU：npu-smi ─────────────────────────────────────────────────────────

class NPUSMIMonitor(BaseMonitor):
    """
    昇腾 NPU 使用 npu-smi 工具。
    npu-smi info -t usages 格式（每行一个 NPU/Chip）：
      NPU   Chip  Process  Memory-Usage(MB)      HBM-Usage(MB)   Util(%)
      0     0     pid      used/total            used/total      83
    """
    # 尝试使用 pynpusmi（如果安装了）；否则用 subprocess
    _pynpusmi = None

    def __init__(self, gpu_ids):
        super().__init__(gpu_ids)
        try:
            import pynpusmi  # noqa: F401
            self._pynpusmi = pynpusmi
        except ImportError:
            self._pynpusmi = None

    def _sample_subprocess(self):
        results = {i: GPUSample() for i in self.gpu_ids}
        try:
            r = subprocess.run(
                ["npu-smi", "info", "-t", "usages"],
                capture_output=True, text=True, timeout=5
            )
            lines = r.stdout.splitlines()
        except Exception:
            return results

        # Parse table lines: skip headers (lines containing "NPU" or "---")
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("NPU", "+", "-", "=")):
                continue
            parts = re.split(r"\s+", line)
            # Expected: npu_id chip_id [proc_id] mem_used/mem_total  util
            # Real format varies by CANN version, try to extract numbers
            try:
                npu_id = int(parts[0])
                if npu_id not in results:
                    continue
                s = GPUSample()

                # Look for memory pattern "used/total" (MB)
                for p in parts:
                    m = re.match(r"(\d+)/(\d+)$", p)
                    if m:
                        used_mb  = int(m.group(1))
                        total_mb = int(m.group(2))
                        s.mem_used_gb  = round(used_mb  / 1024, 2)
                        s.mem_total_gb = round(total_mb / 1024, 2)
                        break

                # Last numeric field is often util%
                nums = [x for x in parts[1:] if re.match(r"^\d+$", x)]
                if nums:
                    s.util_pct = float(nums[-1])

                results[npu_id] = s
            except (ValueError, IndexError):
                continue

        # Try npu-smi info -t power for power/temp
        try:
            r = subprocess.run(
                ["npu-smi", "info", "-t", "power"],
                capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith(("NPU", "+", "-", "=")):
                    continue
                parts = re.split(r"\s+", line)
                try:
                    npu_id = int(parts[0])
                    if npu_id not in results:
                        continue
                    nums = [x for x in parts[1:] if re.match(r"^[\d.]+$", x)]
                    if len(nums) >= 2:
                        results[npu_id].power_w = float(nums[0])
                        results[npu_id].temp_c  = float(nums[1])
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass

        return results

    def sample(self):
        if self._pynpusmi:
            try:
                results = {}
                for gid in self.gpu_ids:
                    info = self._pynpusmi.get_device_info(gid)
                    results[gid] = GPUSample(
                        util_pct=info.get("aicore_util"),
                        mem_used_gb=round(info.get("memory_used_mb", 0) / 1024, 2),
                        mem_total_gb=round(info.get("memory_total_mb", 0) / 1024, 2),
                        power_w=info.get("power"),
                        temp_c=info.get("temperature"),
                    )
                return results
            except Exception:
                pass
        return self._sample_subprocess()


# ── 昆仑芯 XPU：xpu-smi ──────────────────────────────────────────────────────

class XPUSMIMonitor(BaseMonitor):
    """
    昆仑芯 R300/R200 使用 xpu-smi 工具。
    xpu-smi stat 或 xpu-smi query
    """
    def __init__(self, gpu_ids):
        super().__init__(gpu_ids)

    def _run_smi(self, *args):
        try:
            r = subprocess.run(["xpu-smi"] + list(args),
                               capture_output=True, text=True, timeout=5)
            return r.stdout
        except Exception:
            return ""

    def sample(self):
        results = {i: GPUSample() for i in self.gpu_ids}
        out = self._run_smi("stat")
        if not out:
            out = self._run_smi("query", "--format=csv", "--query-device=utilization.xpu,memory.used,memory.total,power.draw,temperature.xpu")
        # Parse CSV-like output
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                # Try to detect device id in line
                m = re.search(r"XPU\s*(\d+)", line, re.I)
                if m:
                    gid = int(m.group(1))
                    if gid in results:
                        nums = re.findall(r"[\d.]+", line)
                        if len(nums) >= 1:
                            results[gid].util_pct = float(nums[0])
                        if len(nums) >= 3:
                            results[gid].mem_used_gb  = round(float(nums[1]) / 1024, 2)
                            results[gid].mem_total_gb = round(float(nums[2]) / 1024, 2)
        return results


# ── torch 内存 fallback ───────────────────────────────────────────────────────

class TorchMemMonitor(BaseMonitor):
    """
    当 SMI 工具不可用时，用 torch API 至少采集显存。
    不提供 util%, power, temp。
    """
    def __init__(self, gpu_ids):
        super().__init__(gpu_ids)
        import torch
        self._torch = torch
        self._device = self._detect_device()

    def _detect_device(self):
        try:
            import torch_npu  # noqa: F401
            if self._torch.npu.is_available():
                return "npu"
        except ImportError:
            pass
        try:
            import torch_xpu  # noqa: F401
            if self._torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def sample(self):
        results = {}
        for gid in self.gpu_ids:
            s = GPUSample()
            try:
                dev = f"{self._device}:{gid}"
                if self._device == "cuda":
                    s.mem_used_gb  = round(self._torch.cuda.memory_allocated(dev) / 1e9, 2)
                    s.mem_total_gb = round(self._torch.cuda.get_device_properties(dev).total_memory / 1e9, 2)
                elif self._device == "npu":
                    s.mem_used_gb  = round(self._torch.npu.memory_allocated(dev) / 1e9, 2)
                    s.mem_total_gb = round(self._torch.npu.get_device_properties(dev).total_memory / 1e9, 2)
            except Exception:
                pass
            results[gid] = s
        return results


# ── 自动检测 ──────────────────────────────────────────────────────────────────

def auto_detect_monitor(gpu_ids) -> BaseMonitor:
    """按优先级尝试各平台监控工具。"""
    # 1. NVIDIA pynvml (CUDA, MACA 如果有 NVML compat 层)
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        if n > 0:
            print(f"[monitor] backend=nvml  ({n} devices)", file=sys.stderr)
            return NVMLMonitor(gpu_ids)
    except Exception:
        pass

    # 2. 昇腾 NPU：npu-smi
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True, timeout=5)
        if r.returncode == 0:
            print("[monitor] backend=npu-smi", file=sys.stderr)
            return NPUSMIMonitor(gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. 海光 DCU：rocm-smi / amd-smi
    for cmd in ("amd-smi", "rocm-smi"):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                print(f"[monitor] backend={cmd}", file=sys.stderr)
                return ROCMSMIMonitor(gpu_ids)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # 4. 昆仑芯 XPU：xpu-smi
    try:
        r = subprocess.run(["xpu-smi", "--version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            print("[monitor] backend=xpu-smi", file=sys.stderr)
            return XPUSMIMonitor(gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 5. Torch fallback（只有显存）
    print("[monitor] backend=torch (memory only — SMI tool not found)", file=sys.stderr)
    return TorchMemMonitor(gpu_ids)


# ─── CSV 列头 ─────────────────────────────────────────────────────────────────

def build_header(n_gpus: int) -> list:
    cols = ["wall_time", "elapsed_s"]
    for i in range(n_gpus):
        cols += [
            f"gpu{i}_util_pct",
            f"gpu{i}_mem_used_gb",
            f"gpu{i}_mem_total_gb",
            f"gpu{i}_power_w",
            f"gpu{i}_temp_c",
        ]
    cols += ["cpu_util_pct", "ram_used_gb", "ram_total_gb"]
    return cols


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",   required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--gpu-ids",  default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    n_gpus  = len(gpu_ids)

    monitor = auto_detect_monitor(gpu_ids)
    psutil.cpu_percent(interval=None)  # 预热

    header = build_header(n_gpus)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    stop = [False]
    def _sig(s, f):
        stop[0] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    t0 = time.time()

    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()
        fout.flush()

        while not stop[0]:
            t_loop = time.time()
            row = {k: "" for k in header}
            row["wall_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            row["elapsed_s"] = round(t_loop - t0, 1)


            # ── GPU 采样 ─────────────────────────────────────────
            samples = monitor.sample()
            for gid in gpu_ids:
                s = samples.get(gid, GPUSample())
                row[f"gpu{gid}_util_pct"]    = "" if s.util_pct    is None else s.util_pct
                row[f"gpu{gid}_mem_used_gb"]  = "" if s.mem_used_gb  is None else s.mem_used_gb
                row[f"gpu{gid}_mem_total_gb"] = "" if s.mem_total_gb is None else s.mem_total_gb
                row[f"gpu{gid}_power_w"]      = "" if s.power_w      is None else s.power_w
                row[f"gpu{gid}_temp_c"]        = "" if s.temp_c       is None else s.temp_c

            # ── CPU / RAM ────────────────────────────────────────
            row["cpu_util_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            row["ram_used_gb"]  = round(vm.used  / 1024**3, 2)
            row["ram_total_gb"] = round(vm.total / 1024**3, 2)

            writer.writerow(row)
            fout.flush()

            elapsed = time.time() - t_loop
            time.sleep(max(0.0, args.interval - elapsed))

    monitor.close()
    print(f"[monitor] finished → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
