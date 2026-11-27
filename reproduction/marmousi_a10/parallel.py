"""把炮集按轮转方式分配到多个进程，在同一张 GPU 上并发正演和伴随。"""

import multiprocessing as mp
import os
import traceback
from types import SimpleNamespace

import numpy as np


def _pin_core(offset):
    """把当前进程绑定到父进程可用核列表中的第 offset 个核。"""
    try:
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[offset % len(allowed)]})
    except (AttributeError, OSError):
        pass


def _worker(conn, parameters, source_rows, receivers, shape, core_offset):
    _pin_core(core_offset)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from PyFWI import acquisition
    from PyFWI.wave_propagation import WavePropagator

    try:
        source = acquisition.Source(np.asarray(source_rows, dtype=np.float32),
                                    parameters["dh"], parameters["dt"])
        source.Ricker(parameters["fdom"])
        wave = WavePropagator(parameters, source, receivers, shape, components=0, chpr=10)
        device = wave.queue.device
        conn.send(("ready", {
            "name": device.name, "platform": device.platform.name,
            "driver": device.driver_version, "memory_bytes": device.global_mem_size,
            "nr": wave.nr, "nt_ext": wave.nt_ext, "nt": wave.nt, "dt": float(wave.dt),
        }))
        while True:
            task = conn.recv()
            if task is None:
                break
            kind, payload = task
            if kind == "forward":
                conn.send(("ok", wave.forward_modeling(payload)["taux"]))
            elif kind == "gradient":
                taux, tauz = payload
                conn.send(("ok", wave.gradient({
                    "taux": taux, "tauz": taux if tauz is None else tauz})))
            else:
                raise ValueError(f"未知任务 {kind}")
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class ParallelWave:
    """用 duck typing 模拟 WavePropagator 的正演和伴随接口。

    第 w 个工作进程负责炮号 w, w+W, w+2W, ...。父进程把各进程的结果
    按炮号拼回 seismic_section 的 Fortran 展平顺序，或把偏梯度相加。
    """

    parallel = True

    def __init__(self, parameters, sources, receivers, shape, workers, timeout=900,
                 start_method="spawn"):
        self.ns = len(sources)
        self.workers = max(1, min(int(workers), self.ns))
        self.timeout = timeout
        if self.workers < 2:
            raise ValueError("并行至少需要两个工作进程")

        self.shots = [list(range(index, self.ns, self.workers))
                      for index in range(self.workers)]
        context = mp.get_context(start_method)
        self.pipes, self.processes, metadata = [], [], []
        for index, shots in enumerate(self.shots):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, parameters, np.asarray(sources[shots], dtype=np.float32),
                      receivers, shape, index),
                daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

        for pipe in self.pipes:
            metadata.append(self._receive(pipe, "ready"))

        first = metadata[0]
        if any(item["name"] != first["name"] for item in metadata):
            self.close()
            raise RuntimeError("并行工作进程选择了不同设备")
        self.nr, self.nt_ext = first["nr"], first["nt_ext"]
        self.nt, self.dt = first["nt"], first["dt"]
        self.queue = SimpleNamespace(device=SimpleNamespace(
            name=first["name"], platform=SimpleNamespace(name=first["platform"]),
            driver_version=first["driver"], global_mem_size=first["memory_bytes"]))

    def _receive(self, pipe, expected="ok"):
        try:
            ready = pipe.poll(self.timeout)
        except (EOFError, OSError):
            ready = True
        if not ready:
            self.close()
            raise TimeoutError(f"工作进程 {self.timeout} 秒内无响应")
        try:
            status, payload = pipe.recv()
        except EOFError as error:
            raise RuntimeError("工作进程提前退出") from error
        if status != expected:
            self.close()
            raise RuntimeError(f"工作进程失败：{payload}")
        return payload

    @staticmethod
    def _columns(shots, nr):
        if len(shots) == 1:
            start = shots[0] * nr
            return slice(start, start + nr)
        return np.concatenate([np.arange(shot * nr, (shot + 1) * nr)
                               for shot in shots])

    def forward_modeling(self, model):
        typed = {key: np.ascontiguousarray(model[key], dtype=np.float32)
                 for key in ("vp", "vs", "rho")}
        for pipe in self.pipes:
            pipe.send(("forward", typed))
        parts = [self._receive(pipe) for pipe in self.pipes]
        data = np.empty((self.nt_ext, self.ns * self.nr), dtype=np.float32)
        for shots, part in zip(self.shots, parts):
            for offset, shot in enumerate(shots):
                data[:, shot * self.nr:(shot + 1) * self.nr] = \
                    part[:, offset * self.nr:(offset + 1) * self.nr]
        return {"taux": data, "tauz": data}

    def gradient(self, residual):
        taux = np.ascontiguousarray(residual["taux"], dtype=np.float32)
        tauz = residual.get("tauz")
        same = tauz is None or tauz is residual["taux"]
        if not same:
            tauz = np.ascontiguousarray(tauz, dtype=np.float32)
        for pipe, shots in zip(self.pipes, self.shots):
            columns = self._columns(shots, self.nr)
            pipe.send(("gradient", (np.ascontiguousarray(taux[:, columns]),
                                    None if same else np.ascontiguousarray(tauz[:, columns]))))
        parts = [self._receive(pipe) for pipe in self.pipes]
        return {key: sum(part[key].astype(np.float64) for part in parts)
                for key in parts[0]}

    def close(self):
        for pipe in getattr(self, "pipes", []):
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in getattr(self, "processes", []):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
