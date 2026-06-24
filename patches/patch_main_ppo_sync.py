#!/usr/bin/env python3
"""
patch_main_ppo_sync.py — add per-worker gen_imbalance measurement to veRL 0.8.0 main_ppo_sync.py

Changes:
  1. AgentLoopWorkerTQ.__init__: add _gen_timing_future field
  2. generate_sequences: record t_start, collect task list, launch fire-and-forget tracker
  3. Add get_last_gen_time() async method
  4. AgentLoopManagerTQ: add get_last_gen_times() method
  5. _compute_metrics: compute rollout/gen_imbalance from per-worker times
"""
import sys, re, py_compile
from pathlib import Path

def find_file(venv_root):
    p = Path(venv_root) / "lib"
    candidates = list(p.rglob("verl/trainer/main_ppo_sync.py"))
    if not candidates:
        print("[patch] ERROR: main_ppo_sync.py not found under", venv_root); sys.exit(1)
    return candidates[0]

def apply(path):
    src = path.read_text(encoding="utf-8")

    # Guard: already patched?
    if "_gen_timing_future" in src:
        print("[patch] main_ppo_sync.py: already patched, skipping"); return

    # Edit 1: __init__
    old1 = "        tq.init()\n        self.background_tasks = set()"
    new1 = "        tq.init()\n        self.background_tasks = set()\n        self._gen_timing_future = None  # [gen_imbalance]"
    assert old1 in src, "Edit 1 target not found (veRL version mismatch?)"
    src = src.replace(old1, new1, 1)

    # Edit 2: generate_sequences — timing tracker
    idx_start = src.find("        # create background tasks for each sample in the batch")
    assert idx_start != -1, "Edit 2 start anchor not found"
    anchor_end = "            task.add_done_callback(self.background_tasks.discard)"
    idx_end = src.find(anchor_end, idx_start) + len(anchor_end)
    src = src[:idx_start] + '''        # create background tasks for each sample in the batch
        t_gen_start = time.perf_counter()  # [gen_imbalance]
        _timing_tasks = []
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # fire-and-forget background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            _timing_tasks.append(task)

        # [gen_imbalance] fire-and-forget tracker
        loop = asyncio.get_event_loop()
        self._gen_timing_future = loop.create_future()
        _fut = self._gen_timing_future

        async def _track_gen_time(tasks=_timing_tasks, t0=t_gen_start, fut=_fut):
            await asyncio.gather(*tasks, return_exceptions=True)
            if not fut.done():
                fut.set_result(time.perf_counter() - t0)

        _track_task = asyncio.create_task(_track_gen_time())
        self.background_tasks.add(_track_task)
        _track_task.add_done_callback(self.background_tasks.discard)''' + src[idx_end:]

    # Edit 3: add get_last_gen_time() before _run_prompt
    idx3 = src.find("    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict")
    assert idx3 != -1, "Edit 3 anchor not found"
    src = src[:idx3] + '''    async def get_last_gen_time(self) -> float:
        """Return wall-clock gen time for last step (gen_imbalance)."""
        if self._gen_timing_future is None:
            return 0.0
        return await self._gen_timing_future

    ''' + src[idx3:]

    # Edit 4: add get_last_gen_times() to AgentLoopManagerTQ
    old4 = "# ======================================= USER SECTION END ======================================="
    assert old4 in src, "Edit 4 anchor not found"
    src = src.replace(old4,
        '''    def get_last_gen_times(self) -> list:
        """Collect per-worker gen wall-clock times for gen_imbalance."""
        return ray.get([w.get_last_gen_time.remote() for w in self.agent_loop_workers])

# ======================================= USER SECTION END =======================================================================''', 1)

    # Edit 5: add gen_imbalance in _compute_metrics
    old5 = "        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))\n\n        # 3. other auxiliary metrics"
    assert old5 in src, "Edit 5 anchor not found"
    src = src.replace(old5, '''        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # [gen_imbalance] per-worker rollout time max/min ratio
        try:
            worker_times = self.async_rollout_manager.get_last_gen_times()
            valid = [t for t in worker_times if t and t > 0]
            if len(valid) >= 2:
                metrics["rollout/gen_imbalance"] = max(valid) / min(valid)
        except Exception:
            pass

        # 3. other auxiliary metrics''', 1)

    path.write_text(src, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[patch] main_ppo_sync.py: all 5 edits applied OK -> {path}")

if __name__ == "__main__":
    from pathlib import Path
    import sys
    venv = Path(sys.prefix)
    apply(find_file(venv))
