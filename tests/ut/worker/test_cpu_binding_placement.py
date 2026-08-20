#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Structural guard for the cpu-binding call site (P1 / A2e).
#
# bind_cpus runs migratepages over the worker's resident pages. It must
# execute in _init_device (small, movable footprint) and NOT in
# compile_or_warm_up_model, where large host buffers registered for DMA
# (e.g. the LMCache shared CPU cache slab pinned via aclrtHostRegister)
# cannot be migrated and stall migratepages past its subprocess timeout.
#

import ast
import pathlib
import unittest

WORKER_PY = (
    pathlib.Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "worker.py"
)


def _method_call_names(class_name: str, method_name: str) -> set[str]:
    """Names called inside `<class_name>.<method_name>` in worker.py."""
    tree = ast.parse(WORKER_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == class_name
        ):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return {
                        n.func.id
                        for n in ast.walk(item)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                    }
    raise AssertionError(f"{class_name}.{method_name} not found in {WORKER_PY}")


class TestCpuBindingCallSite(unittest.TestCase):
    def test_bind_cpus_called_in_init_device(self):
        calls = _method_call_names("NPUWorker", "_init_device")
        self.assertIn("bind_cpus", calls)
        self.assertIn("get_ascend_config", calls)  # guarded by enable_cpu_binding

    def test_bind_cpus_not_called_after_warmup_allocations(self):
        calls = _method_call_names("NPUWorker", "compile_or_warm_up_model")
        self.assertNotIn("bind_cpus", calls)


if __name__ == "__main__":
    unittest.main()
