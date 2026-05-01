#!/usr/bin/env python3
"""
NanoCompile Demo

Supported operators in this Demmo:
  - MatMul
  - Gemm
  - Add
  - Relu
  - Identity
  - Flatten

Compiler passes demonstrated:
  1. ONNX parsing into a compact IR (although this has not been tested by us yet, the demo command given bypasses this)
  2. Shape inference 
  3. Constant folding
  4. Linear/Add/Relu fusion (This is pattern matched currently just to show what we mean , but we aim to detect this)
  5. Static scratch-memory planning with liveness reuse
  6. C++ code generation with no runtime graph interpreter

The generated C++ depends only on the C++ standard library.
Compiler-time Python dependencies:
  - numpy
  - onnx, optional; only needed when compiling a real .onnx file

Usage:
  python nanocompile.py --demo --out generated_demo
  python nanocompile.py --onnx model.onnx --out generated_model

Then:
  cd generated_demo
  g++ -std=c++17 -O3 -march=native model.cpp main.cpp -o demo
  ./demo
"""

from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

def c_ident(name: str) -> str:
    """Convert tensor/node names into safe C++ identifiers."""
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not name or name[0].isdigit():
        name = "_" + name
    return name

def prod(shape: Sequence[int]) -> int:
    out = 1
    for x in shape:
        out *= int(x)
    return int(out)

def numpy_dtype_to_cpp(dtype: str) -> str:
    if dtype != "float32":
        raise NotImplementedError(f"NanoCompile MVP only emits float32, got {dtype}")
    return "float"

@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    is_const: bool = False
    producer: Optional[int] = None
    consumers: List[int] = field(default_factory=list)
    scratch_offset: Optional[int] = None
    scratch_size: int = 0

    @property
    def elements(self) -> int:
        return prod(self.shape)

@dataclass
class Node:
    name: str
    op: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphIR:
    name: str
    inputs: List[str]
    outputs: List[str]
    tensors: Dict[str, TensorInfo]
    nodes: List[Node]
    weights: Dict[str, np.ndarray]

    def rebuild_uses(self) -> None:
        for t in self.tensors.values():
            t.producer = None
            t.consumers = []
        for i, node in enumerate(self.nodes):
            for out in node.outputs:
                self.tensors[out].producer = i
            for inp in node.inputs:
                if inp in self.tensors:
                    self.tensors[inp].consumers.append(i)

    def validate(self) -> None:
        missing = []
        for node in self.nodes:
            for inp in node.inputs:
                if inp not in self.tensors:
                    missing.append((node.name, inp))
            for out in node.outputs:
                if out not in self.tensors:
                    missing.append((node.name, out))
        if missing:
            raise ValueError(f"Graph references missing tensors: {missing[:5]}")

class OnnxFrontend:
    """Converts an ONNX model into NanoCompile's compact IR."""

    @staticmethod
    def _shape_from_value_info(value_info: Any) -> Tuple[int, ...]:
        dims = []
        tensor_type = value_info.type.tensor_type
        for d in tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                raise ValueError(
                    f"Dynamic dimension in tensor {value_info.name!r}. "
                    "This MVP expects static shapes."
                )
        return tuple(dims)

    @staticmethod
    def _onnx_dtype_to_str(elem_type: int) -> str:

        if elem_type == 1:
            return "float32"
        raise NotImplementedError("NanoCompile MVP currently supports float32 ONNX tensors only")

    @staticmethod
    def _attr_value(attr: Any) -> Any:
        import onnx
        from onnx import helper

        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, tuple):
            return list(value)
        return value

    @classmethod
    def from_onnx(cls, path: str | os.PathLike[str]) -> GraphIR:
        try:
            import onnx
            from onnx import numpy_helper, shape_inference
        except Exception as exc:
            raise RuntimeError(
                "Install ONNX to compile real ONNX files: pip install onnx"
            ) from exc

        model = onnx.load(str(path))
        model = shape_inference.infer_shapes(model)
        graph = model.graph

        weights: Dict[str, np.ndarray] = {}
        tensors: Dict[str, TensorInfo] = {}

        initializer_names = {init.name for init in graph.initializer}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init).astype(np.float32, copy=False)
            weights[init.name] = np.ascontiguousarray(arr)
            tensors[init.name] = TensorInfo(
                name=init.name,
                shape=tuple(int(x) for x in arr.shape),
                dtype="float32",
                is_const=True,
            )

        value_infos = list(graph.input) + list(graph.value_info) + list(graph.output)
        for vi in value_infos:
            if vi.name in tensors:
                continue
            shape = cls._shape_from_value_info(vi)
            dtype = cls._onnx_dtype_to_str(vi.type.tensor_type.elem_type)
            tensors[vi.name] = TensorInfo(name=vi.name, shape=shape, dtype=dtype)

        inputs = [vi.name for vi in graph.input if vi.name not in initializer_names]
        outputs = [vi.name for vi in graph.output]

        nodes: List[Node] = []
        for idx, n in enumerate(graph.node):
            name = n.name or f"{n.op_type}_{idx}"
            attrs = {a.name: cls._attr_value(a) for a in n.attribute}
            for out in n.output:
                if out and out not in tensors:

                    tensors[out] = TensorInfo(name=out, shape=(), dtype="float32")
            nodes.append(Node(name=name, op=n.op_type, inputs=list(n.input), outputs=list(n.output), attrs=attrs))

        ir = GraphIR(
            name=Path(path).stem,
            inputs=inputs,
            outputs=outputs,
            tensors=tensors,
            nodes=nodes,
            weights=weights,
        )
        ShapeInferPass().run(ir)
        ir.rebuild_uses()
        ir.validate()
        return ir

class DemoFrontend:
    """Creates a deterministic mini-MLP graph without requiring ONNX."""

    @staticmethod
    def make_graph() -> GraphIR:
        rng = np.random.default_rng(7)
        in_dim, hidden_dim, out_dim = 8, 16, 4

        weights = {
            "W1": rng.normal(0.0, 0.25, size=(in_dim, hidden_dim)).astype(np.float32),
            "B1": rng.normal(0.0, 0.05, size=(hidden_dim,)).astype(np.float32),
            "W2": rng.normal(0.0, 0.20, size=(hidden_dim, out_dim)).astype(np.float32),
            "B2": rng.normal(0.0, 0.05, size=(out_dim,)).astype(np.float32),
        }

        tensors: Dict[str, TensorInfo] = {
            "input": TensorInfo("input", (1, in_dim)),
            "h_mm": TensorInfo("h_mm", (1, hidden_dim)),
            "h_bias": TensorInfo("h_bias", (1, hidden_dim)),
            "h_relu": TensorInfo("h_relu", (1, hidden_dim)),
            "y_mm": TensorInfo("y_mm", (1, out_dim)),
            "output": TensorInfo("output", (1, out_dim)),
        }
        for k, v in weights.items():
            tensors[k] = TensorInfo(k, tuple(v.shape), is_const=True)

        nodes = [
            Node("dense1_matmul", "MatMul", ["input", "W1"], ["h_mm"]),
            Node("dense1_bias", "Add", ["h_mm", "B1"], ["h_bias"]),
            Node("dense1_relu", "Relu", ["h_bias"], ["h_relu"]),
            Node("dense2_matmul", "MatMul", ["h_relu", "W2"], ["y_mm"]),
            Node("dense2_bias", "Add", ["y_mm", "B2"], ["output"]),
        ]
        ir = GraphIR(
            name="demo_mlp",
            inputs=["input"],
            outputs=["output"],
            tensors=tensors,
            nodes=nodes,
            weights=weights,
        )
        ir.rebuild_uses()
        return ir

class ShapeInferPass:
    """Infers static output shapes for supported operators."""

    def run(self, ir: GraphIR) -> GraphIR:
        for i, node in enumerate(ir.nodes):
            if node.op in {"MatMul", "FusedMatMulAdd", "FusedMatMulAddRelu"}:
                a = ir.tensors[node.inputs[0]].shape
                b = ir.tensors[node.inputs[1]].shape
                if len(a) != 2 or len(b) != 2:
                    raise NotImplementedError(f"{node.op} MVP supports 2D tensors only")
                if a[1] != b[0]:
                    raise ValueError(f"MatMul mismatch: {a} x {b}")
                self._set_shape(ir, node.outputs[0], (a[0], b[1]))

            elif node.op == "Gemm":
                a = ir.tensors[node.inputs[0]].shape
                b = ir.tensors[node.inputs[1]].shape
                trans_a = int(node.attrs.get("transA", 0))
                trans_b = int(node.attrs.get("transB", 0))
                a2 = (a[1], a[0]) if trans_a else a
                b2 = (b[1], b[0]) if trans_b else b
                if len(a2) != 2 or len(b2) != 2 or a2[1] != b2[0]:
                    raise ValueError(f"Gemm mismatch: {a} x {b}")
                self._set_shape(ir, node.outputs[0], (a2[0], b2[1]))

            elif node.op in {"Add", "Sub", "Mul"}:
                lhs = ir.tensors[node.inputs[0]].shape
                rhs = ir.tensors[node.inputs[1]].shape
                self._set_shape(ir, node.outputs[0], self._broadcast_shape(lhs, rhs))

            elif node.op in {"Relu", "Identity"}:
                self._set_shape(ir, node.outputs[0], ir.tensors[node.inputs[0]].shape)

            elif node.op == "Flatten":
                shape = ir.tensors[node.inputs[0]].shape
                axis = int(node.attrs.get("axis", 1))
                left = prod(shape[:axis])
                right = prod(shape[axis:])
                self._set_shape(ir, node.outputs[0], (left, right))

            else:
                raise NotImplementedError(f"Unsupported op in NanoCompile MVP: {node.op}")

        ir.rebuild_uses()
        return ir

    @staticmethod
    def _set_shape(ir: GraphIR, tensor_name: str, shape: Tuple[int, ...]) -> None:
        if tensor_name not in ir.tensors:
            ir.tensors[tensor_name] = TensorInfo(tensor_name, shape)
        elif not ir.tensors[tensor_name].shape:
            ir.tensors[tensor_name].shape = shape
        elif tuple(ir.tensors[tensor_name].shape) != tuple(shape):
            raise ValueError(
                f"Shape mismatch for {tensor_name}: existing {ir.tensors[tensor_name].shape}, inferred {shape}"
            )

    @staticmethod
    def _broadcast_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        try:
            return tuple(np.broadcast_shapes(a, b))
        except Exception as exc:
            raise ValueError(f"Cannot broadcast shapes {a} and {b}") from exc

class ConstantFoldPass:
    """Evaluates supported pure constant subgraphs at compile time."""

    def run(self, ir: GraphIR) -> GraphIR:
        new_nodes: List[Node] = []
        for node in ir.nodes:
            if node.op in {"Add", "Sub", "Mul", "Relu", "MatMul", "Identity"} and all(
                inp in ir.weights for inp in node.inputs
            ):
                result = self._eval(node, [ir.weights[i] for i in node.inputs])
                out = node.outputs[0]
                ir.weights[out] = np.ascontiguousarray(result.astype(np.float32, copy=False))
                ir.tensors[out].is_const = True
                ir.tensors[out].shape = tuple(ir.weights[out].shape)
            else:
                new_nodes.append(node)
        ir.nodes = new_nodes
        ir.rebuild_uses()
        return ir

    @staticmethod
    def _eval(node: Node, xs: List[np.ndarray]) -> np.ndarray:
        if node.op == "Add":
            return xs[0] + xs[1]
        if node.op == "Sub":
            return xs[0] - xs[1]
        if node.op == "Mul":
            return xs[0] * xs[1]
        if node.op == "Relu":
            return np.maximum(xs[0], 0)
        if node.op == "MatMul":
            return xs[0] @ xs[1]
        if node.op == "Identity":
            return xs[0]
        raise NotImplementedError(node.op)

class FusionPass:
    """Fuses common inference patterns into single codegen kernels."""

    def run(self, ir: GraphIR) -> GraphIR:
        ir.rebuild_uses()
        consumed: set[int] = set()
        fused: List[Node] = []

        for i, node in enumerate(ir.nodes):
            if i in consumed:
                continue

            if node.op == "MatMul" and len(node.outputs) == 1:
                mm_out = node.outputs[0]
                add_idx = self._single_consumer_of(ir, mm_out, "Add")
                if add_idx is not None:
                    add = ir.nodes[add_idx]
                    bias_name = add.inputs[1] if add.inputs[0] == mm_out else add.inputs[0]
                    if bias_name in ir.weights:
                        add_out = add.outputs[0]
                        relu_idx = self._single_consumer_of(ir, add_out, "Relu")
                        if relu_idx is not None:
                            relu = ir.nodes[relu_idx]
                            fused.append(
                                Node(
                                    name=node.name + "_add_relu_fused",
                                    op="FusedMatMulAddRelu",
                                    inputs=[node.inputs[0], node.inputs[1], bias_name],
                                    outputs=[relu.outputs[0]],
                                    attrs={"source": [node.name, add.name, relu.name]},
                                )
                            )
                            consumed.update({i, add_idx, relu_idx})
                            continue
                        else:
                            fused.append(
                                Node(
                                    name=node.name + "_add_fused",
                                    op="FusedMatMulAdd",
                                    inputs=[node.inputs[0], node.inputs[1], bias_name],
                                    outputs=[add.outputs[0]],
                                    attrs={"source": [node.name, add.name]},
                                )
                            )
                            consumed.update({i, add_idx})
                            continue

            fused.append(node)

        ir.nodes = fused
        ShapeInferPass().run(ir)
        ir.rebuild_uses()
        return ir

    @staticmethod
    def _single_consumer_of(ir: GraphIR, tensor_name: str, op: str) -> Optional[int]:
        consumers = ir.tensors[tensor_name].consumers
        if len(consumers) == 1 and ir.nodes[consumers[0]].op == op:
            return consumers[0]
        return None

class MemoryPlanPass:
    """Assigns intermediate tensors to a single static scratch pool with liveness reuse."""

    def run(self, ir: GraphIR) -> GraphIR:
        ir.rebuild_uses()
        intervals: List[Tuple[int, int, str, int]] = []
        graph_external = set(ir.inputs) | set(ir.outputs) | set(ir.weights)

        for name, t in ir.tensors.items():
            if name in graph_external or t.is_const:
                continue
            if t.producer is None:
                continue
            start = t.producer
            end = max(t.consumers) if t.consumers else start
            intervals.append((start, end, name, t.elements))

        intervals.sort(key=lambda x: (x[0], x[1]))
        active: List[Tuple[int, int, int]] = []
        free: List[Tuple[int, int]] = []
        scratch_size = 0

        for start, end, name, size in intervals:
            still_active = []
            for a_end, a_off, a_size in active:
                if a_end < start:
                    free.append((a_off, a_size))
                else:
                    still_active.append((a_end, a_off, a_size))
            active = still_active

            chosen_idx = None
            chosen = None
            for idx, (off, slot_size) in enumerate(free):
                if slot_size >= size:
                    if chosen is None or slot_size < chosen[1]:
                        chosen_idx = idx
                        chosen = (off, slot_size)
            if chosen is None:
                offset = scratch_size
                scratch_size += size
            else:
                offset, slot_size = chosen
                free.pop(chosen_idx)

            ir.tensors[name].scratch_offset = offset
            ir.tensors[name].scratch_size = size
            active.append((end, offset, size))

        ir.tensors["__scratch__"] = TensorInfo("__scratch__", (scratch_size,), scratch_size=scratch_size)
        return ir

class CppCodegen:
    def __init__(self, ir: GraphIR, out_dir: str | os.PathLike[str]):
        self.ir = ir
        self.out_dir = Path(out_dir)
        self.lines: List[str] = []

    def emit(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._emit_weights_hpp()
        self._emit_header()
        self._emit_model_cpp()
        self._emit_main_cpp()
        self._emit_cmake()
        self._emit_manifest()

    def _emit_weights_hpp(self) -> None:
        lines = [
            "#pragma once",
            "#include <cstddef>",
            "",
            "namespace nanocompile_weights {",
        ]
        for name, arr in self.ir.weights.items():
            if name in self.ir.inputs:
                continue
            flat = arr.astype(np.float32, copy=False).ravel()
            ident = "W_" + c_ident(name)
            values = ", ".join(self._float_literal(x) for x in flat)
            lines.append(f"alignas(64) static constexpr float {ident}[{len(flat)}] = {{ {values} }};")
        lines += ["}", ""]
        (self.out_dir / "weights.hpp").write_text("\n".join(lines), encoding="utf-8")

    def _emit_header(self) -> None:
        sig = self._signature(declaration=True)
        content = f"""#pragma once
#include <cstddef>

namespace nanocompile {{
{sig};
}}
"""
        (self.out_dir / "nanocompile_model.hpp").write_text(content, encoding="utf-8")

    def _emit_model_cpp(self) -> None:
        self.lines = [
            "#include \"nanocompile_model.hpp\"",
            "#include \"weights.hpp\"",
            "#include <algorithm>",
            "#include <cmath>",
            "#include <cstring>",
            "",
            "namespace nanocompile {",
            "namespace {",
            "inline float relu(float x) { return x > 0.0f ? x : 0.0f; }",
            "}",
            "",
        ]
        scratch = self.ir.tensors.get("__scratch__", TensorInfo("__scratch__", (0,)))
        scratch_size = max(1, scratch.shape[0] if scratch.shape else 0)
        self.lines.append(self._signature(declaration=False) + " {")
        self.lines.append(f"    alignas(64) static thread_local float scratch[{scratch_size}];")
        self.lines.append("    (void)scratch;")
        for node in self.ir.nodes:
            self._emit_node(node)
        self.lines.append("}")
        self.lines.append("}")
        self.lines.append("")
        (self.out_dir / "model.cpp").write_text("\n".join(self.lines), encoding="utf-8")

    def _emit_main_cpp(self) -> None:
        if len(self.ir.inputs) != 1 or len(self.ir.outputs) != 1:
            demo_body = "// Multi-input models: call nanocompile::inference(...) from your host code."
        else:
            inp = self.ir.tensors[self.ir.inputs[0]]
            out = self.ir.tensors[self.ir.outputs[0]]
            xs = [((i % 7) - 3) / 7.0 for i in range(inp.elements)]
            init = ", ".join(self._float_literal(x) for x in xs)
            demo_body = f"""
    alignas(64) float input[{inp.elements}] = {{ {init} }};
    alignas(64) float output[{out.elements}] = {{0}};
    nanocompile::inference(input, output);
    std::cout << "NanoCompile output:";
    for (float v : output) std::cout << " " << v;
    std::cout << "\\n";
""".rstrip()
        content = f"""#include \"nanocompile_model.hpp\"
#include <iostream>

int main() {{
{demo_body}
    return 0;
}}
"""
        (self.out_dir / "main.cpp").write_text(content, encoding="utf-8")

    def _emit_cmake(self) -> None:
        content = """cmake_minimum_required(VERSION 3.16)
project(nanocompile_generated LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(nanocompile_demo model.cpp main.cpp)
target_compile_options(nanocompile_demo PRIVATE -O3)
"""
        (self.out_dir / "CMakeLists.txt").write_text(content, encoding="utf-8")

    def _emit_manifest(self) -> None:
        lines = [
            f"Model: {self.ir.name}",
            "",
            "Inputs:",
        ]
        for name in self.ir.inputs:
            t = self.ir.tensors[name]
            lines.append(f"  - {name}: shape={t.shape}, dtype={t.dtype}")
        lines.append("Outputs:")
        for name in self.ir.outputs:
            t = self.ir.tensors[name]
            lines.append(f"  - {name}: shape={t.shape}, dtype={t.dtype}")
        lines.append("")
        lines.append("Compiled graph:")
        for node in self.ir.nodes:
            lines.append(f"  - {node.name}: {node.op}({', '.join(node.inputs)}) -> {', '.join(node.outputs)}")
        scratch = self.ir.tensors.get("__scratch__")
        if scratch:
            lines.append("")
            lines.append(f"Static scratch floats: {scratch.shape[0] if scratch.shape else 0}")
        (self.out_dir / "compile_manifest.txt").write_text("\n".join(lines), encoding="utf-8")

    def _emit_node(self, node: Node) -> None:
        self.lines.append(f"    // {node.name}: {node.op}")
        if node.op == "FusedMatMulAddRelu":
            self._emit_fused_matmul_add(node, with_relu=True)
        elif node.op == "FusedMatMulAdd":
            self._emit_fused_matmul_add(node, with_relu=False)
        elif node.op == "MatMul":
            self._emit_matmul(node)
        elif node.op == "Gemm":
            self._emit_gemm(node)
        elif node.op == "Add":
            self._emit_add(node)
        elif node.op == "Relu":
            self._emit_relu(node)
        elif node.op == "Identity":
            src = self._ptr(node.inputs[0])
            dst = self._ptr(node.outputs[0])
            n = self.ir.tensors[node.outputs[0]].elements
            self.lines.append(f"    std::memcpy({dst}, {src}, sizeof(float) * {n});")
        elif node.op == "Flatten":
            src = self._ptr(node.inputs[0])
            dst = self._ptr(node.outputs[0])
            n = self.ir.tensors[node.outputs[0]].elements
            self.lines.append(f"    std::memcpy({dst}, {src}, sizeof(float) * {n});")
        else:
            raise NotImplementedError(node.op)
        self.lines.append("")

    def _emit_fused_matmul_add(self, node: Node, with_relu: bool) -> None:
        x, w, b = node.inputs
        y = node.outputs[0]
        m, k = self.ir.tensors[x].shape
        wk, n = self.ir.tensors[w].shape
        assert wk == k
        px, pw, pb, py = self._ptr(x), self._ptr(w), self._ptr(b), self._ptr(y)
        for_i, for_j, for_k = self._loop_vars(node.name)
        self.lines.append(f"    for (int {for_i} = 0; {for_i} < {m}; ++{for_i}) {{")
        self.lines.append(f"        for (int {for_j} = 0; {for_j} < {n}; ++{for_j}) {{")
        self.lines.append(f"            float acc = {pb}[{for_j}];")
        self.lines.append(f"            for (int {for_k} = 0; {for_k} < {k}; ++{for_k}) {{")
        self.lines.append(
            f"                acc += {px}[{for_i} * {k} + {for_k}] * {pw}[{for_k} * {n} + {for_j}];"
        )
        self.lines.append("            }")
        if with_relu:
            self.lines.append(f"            {py}[{for_i} * {n} + {for_j}] = relu(acc);")
        else:
            self.lines.append(f"            {py}[{for_i} * {n} + {for_j}] = acc;")
        self.lines.append("        }")
        self.lines.append("    }")

    def _emit_matmul(self, node: Node) -> None:
        a, b = node.inputs
        y = node.outputs[0]
        m, k = self.ir.tensors[a].shape
        wk, n = self.ir.tensors[b].shape
        assert wk == k
        pa, pb, py = self._ptr(a), self._ptr(b), self._ptr(y)
        vi, vj, vk, vkk = self._loop_vars(node.name, count=4)
        block = 32
        self.lines.append(f"    std::fill({py}, {py} + {m * n}, 0.0f);")
        self.lines.append(f"    for (int {vi} = 0; {vi} < {m}; ++{vi}) {{")
        self.lines.append(f"        for (int {vkk} = 0; {vkk} < {k}; {vkk} += {block}) {{")
        self.lines.append(f"            const int kend = std::min({vkk} + {block}, {k});")
        self.lines.append(f"            for (int {vj} = 0; {vj} < {n}; ++{vj}) {{")
        self.lines.append(f"                float acc = {py}[{vi} * {n} + {vj}];")
        self.lines.append(f"                for (int {vk} = {vkk}; {vk} < kend; ++{vk}) {{")
        self.lines.append(f"                    acc += {pa}[{vi} * {k} + {vk}] * {pb}[{vk} * {n} + {vj}];")
        self.lines.append("                }")
        self.lines.append(f"                {py}[{vi} * {n} + {vj}] = acc;")
        self.lines.append("            }")
        self.lines.append("        }")
        self.lines.append("    }")

    def _emit_gemm(self, node: Node) -> None:
        if int(node.attrs.get("transA", 0)) or int(node.attrs.get("transB", 0)):
            raise NotImplementedError("Gemm transpose support can be added; disabled in MVP codegen for clarity")

        a, b = node.inputs[:2]
        c = node.inputs[2] if len(node.inputs) >= 3 else None
        y = node.outputs[0]
        m, k = self.ir.tensors[a].shape
        wk, n = self.ir.tensors[b].shape
        assert wk == k
        pa, pb, py = self._ptr(a), self._ptr(b), self._ptr(y)
        pc = self._ptr(c) if c else None
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        vi, vj, vk = self._loop_vars(node.name)
        self.lines.append(f"    for (int {vi} = 0; {vi} < {m}; ++{vi}) {{")
        self.lines.append(f"        for (int {vj} = 0; {vj} < {n}; ++{vj}) {{")
        if pc:
            self.lines.append(f"            float acc = {self._float_literal(beta)} * {pc}[{vj}];")
        else:
            self.lines.append("            float acc = 0.0f;")
        self.lines.append(f"            for (int {vk} = 0; {vk} < {k}; ++{vk}) {{")
        self.lines.append(
            f"                acc += {self._float_literal(alpha)} * {pa}[{vi} * {k} + {vk}] * {pb}[{vk} * {n} + {vj}];"
        )
        self.lines.append("            }")
        self.lines.append(f"            {py}[{vi} * {n} + {vj}] = acc;")
        self.lines.append("        }")
        self.lines.append("    }")

    def _emit_add(self, node: Node) -> None:
        a, b = node.inputs
        y = node.outputs[0]
        ashape = self.ir.tensors[a].shape
        bshape = self.ir.tensors[b].shape
        yshape = self.ir.tensors[y].shape
        pa, pb, py = self._ptr(a), self._ptr(b), self._ptr(y)
        n = prod(yshape)
        idx = "idx_" + c_ident(node.name)

        if len(yshape) == 2 and len(bshape) == 1 and bshape[0] == yshape[1]:
            cols = yshape[1]
            self.lines.append(f"    for (int {idx} = 0; {idx} < {n}; ++{idx}) {py}[{idx}] = {pa}[{idx}] + {pb}[{idx} % {cols}];")
        elif len(yshape) == 2 and len(ashape) == 1 and ashape[0] == yshape[1]:
            cols = yshape[1]
            self.lines.append(f"    for (int {idx} = 0; {idx} < {n}; ++{idx}) {py}[{idx}] = {pa}[{idx} % {cols}] + {pb}[{idx}];")
        elif ashape == bshape == yshape:
            self.lines.append(f"    for (int {idx} = 0; {idx} < {n}; ++{idx}) {py}[{idx}] = {pa}[{idx}] + {pb}[{idx}];")
        else:
            raise NotImplementedError(f"Add broadcasting case not implemented: {ashape} + {bshape} -> {yshape}")

    def _emit_relu(self, node: Node) -> None:
        x = node.inputs[0]
        y = node.outputs[0]
        px, py = self._ptr(x), self._ptr(y)
        n = self.ir.tensors[y].elements
        idx = "idx_" + c_ident(node.name)
        self.lines.append(f"    for (int {idx} = 0; {idx} < {n}; ++{idx}) {py}[{idx}] = relu({px}[{idx}]);")

    def _ptr(self, tensor_name: Optional[str]) -> str:
        if tensor_name is None:
            raise ValueError("None tensor")
        if tensor_name in self.ir.weights:
            return "nanocompile_weights::W_" + c_ident(tensor_name)
        if tensor_name in self.ir.inputs:
            return c_ident(tensor_name)
        if tensor_name in self.ir.outputs:
            return c_ident(tensor_name)
        t = self.ir.tensors[tensor_name]
        if t.scratch_offset is None:

            return c_ident(tensor_name)
        return f"(scratch + {t.scratch_offset})"

    def _signature(self, declaration: bool) -> str:
        parts = []
        for name in self.ir.inputs:
            parts.append(f"const float* {c_ident(name)}")
        for name in self.ir.outputs:
            parts.append(f"float* {c_ident(name)}")
        return f"void inference({', '.join(parts)})"

    @staticmethod
    def _float_literal(x: Any) -> str:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            raise ValueError("Cannot emit NaN/Inf weights in MVP")
        if xf == 0.0:
            return "0.0f"
        return f"{xf:.9g}f"

    @staticmethod
    def _loop_vars(name: str, count: int = 3) -> Tuple[str, ...]:
        stem = c_ident(name)
        names = [f"i_{stem}", f"j_{stem}", f"k_{stem}", f"kk_{stem}", f"t_{stem}"]
        return tuple(names[:count])

class NanoCompiler:
    def compile(self, ir: GraphIR, out_dir: str | os.PathLike[str]) -> GraphIR:
        ir.validate()
        ShapeInferPass().run(ir)
        ConstantFoldPass().run(ir)
        FusionPass().run(ir)
        MemoryPlanPass().run(ir)
        CppCodegen(ir, out_dir).emit()
        return ir

def main() -> None:
    parser = argparse.ArgumentParser(description="NanoCompile MVP: ONNX/static graph to standalone C++")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="Compile the built-in demo MLP graph")
    src.add_argument("--onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--out", type=str, default="generated_model", help="Output directory for generated C++")
    args = parser.parse_args()

    if args.demo:
        ir = DemoFrontend.make_graph()
    else:
        ir = OnnxFrontend.from_onnx(args.onnx)

    compiled = NanoCompiler().compile(ir, args.out)

    print(f"NanoCompile generated C++ in: {args.out}")
    print("Compiled nodes:")
    for node in compiled.nodes:
        print(f"  {node.name}: {node.op} -> {node.outputs}")
    scratch = compiled.tensors.get("__scratch__")
    print(f"Static scratch floats: {scratch.shape[0] if scratch and scratch.shape else 0}")

if __name__ == "__main__":
    main()
