# llama-fermi-clblast

Run a small LLM on **NVIDIA Fermi-era GPUs** through the old llama.cpp **OpenCL / CLBlast** backend.

This repository is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) pinned to upstream commit [`2e6cd4b`](https://github.com/ggml-org/llama.cpp/commit/2e6cd4b02549e343bef3768e6b946f999c82e823) from the May 2023 CLBlast era, with a few compatibility patches for OpenLLaMA-family models and legacy NVIDIA hardware.

> **Proof of life:** a GeForce **GT 430 (Fermi, sm_21, 964 MiB VRAM)** can generate text with OpenLLaMA-3B Q4_0 at about **1.35–2.40 tokens/s** when 4–10 layers are offloaded.
>
> This is mainly a retrocomputing / compatibility project. On the tested PCIe 2.0 x1 setup, CPU-only inference was faster.

日本語: 2023年当時の llama.cpp CLBlast バックエンドを使い、**GeForce GT 430 のような Fermi 世代GPUでもLLM推論を動かす**ためのフォークです。

## Why this exists

Modern llama.cpp has moved far beyond the model formats and GPU backends supported by hardware such as Fermi. This repository deliberately stays on an old llama.cpp revision where the CLBlast backend can still target OpenCL devices exposed by NVIDIA's legacy driver.

The goal is not modern performance or feature parity. The goal is simply:

- make a Fermi GPU enumerate through OpenCL,
- offload actual LLM layers to it,
- and demonstrate that useful inference still runs on hardware from the early 2010s.

## Tested setup

| Component | Tested configuration |
|---|---|
| GPU | NVIDIA GeForce GT 430 |
| Architecture | Fermi, sm_21 |
| VRAM | 964 MiB |
| Driver | NVIDIA 390.157 legacy driver |
| OpenCL | NVIDIA OpenCL via `libnvidia-opencl` |
| Host link | PCIe 2.0 x1 riser |
| Model | OpenLLaMA-3B Q4_0 |
| Model format | GGJT v3 / legacy GGML `.bin` |
| Context | 512 |

Other OpenCL/CLBlast-capable GPUs may work, but **GT 430 is the primary verified target**.

## What changed from upstream

The fork is based on upstream llama.cpp commit `2e6cd4b` and adds the following compatibility patches:

| Change | File | Purpose |
|---|---|---|
| BF16 safetensors decode using uint16 -> F32 conversion | `convert.py` | Read modern BF16 Hugging Face checkpoints without requiring PyTorch |
| `n_head` override for `n_embd == 3200` | `convert.py` | OpenLLaMA-3B uses 32 heads / head_dim 100 |
| `n_mult = 8640` for `n_embd == 3200` | `convert.py` | Make the legacy runtime derive the correct FFN size |
| `MODEL_1B` / `MODEL_3B` registration | `llama.cpp` | Allow non-original-LLaMA layer counts to load |
| Memory requirement table entries | `llama.cpp` | Support the added model sizes in the old loader |

## Quick start

### 1. Install OpenCL + CLBlast

You need:

- a working OpenCL runtime for the target GPU,
- OpenCL development headers,
- CLBlast.

On Debian-family systems the development packages are typically `ocl-icd-opencl-dev` and `libclblast-dev`; the actual NVIDIA OpenCL runtime comes from the installed NVIDIA driver.

Check that the GPU is visible before building:

```bash
clinfo -l
```

### 2. Build

```bash
git clone https://github.com/eightman999/llama-fermi-clblast.git
cd llama-fermi-clblast

make -j LLAMA_CLBLAST=1
```

### 3. Get a compatible model

This tree uses the old **GGJT v3 / pre-GGUF** format.

A ready-to-run OpenLLaMA-3B legacy model is available here:

https://huggingface.co/eightman999/openllama-3b-ggml-legacy

Modern GGUF files will **not** load directly in this tree.

### 4. Run on the OpenCL GPU

```bash
GGML_OPENCL_PLATFORM=0 \
GGML_OPENCL_DEVICE=1 \
./main \
  -m openllama-3b-q4_0-ggml.bin \
  -ngl 8 \
  -t 2 \
  -p "Hello" \
  -n 32
```

`GGML_OPENCL_PLATFORM` selects the OpenCL platform and `GGML_OPENCL_DEVICE` selects a device inside that platform.

Device numbering follows `clinfo -l`. On the original test machine, the NVIDIA platform contained:

```text
GT 730 -> device 0
GT 430 -> device 1
GT 710 -> device 2
```

Your numbering may be different.

## Converting a Hugging Face checkpoint

The included `convert.py` understands safetensors, including the BF16 compatibility patch used for OpenLLaMA.

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Convert to legacy GGML/GGJT F16:

```bash
python3 convert.py /path/to/model --outtype f16
```

Then quantize:

```bash
./quantize \
  /path/to/model/ggml-model-f16.bin \
  ./openllama-3b-q4_0-ggml.bin \
  q4_0
```

For OpenLLaMA-style Hugging Face directories, keep `tokenizer.model` in the model directory or provide it with `--vocab-dir`.

## Model compatibility

### Works / intended

- legacy LLaMA-style **MHA** architectures,
- OpenLLaMA-1B / OpenLLaMA-3B with the patches in this fork,
- F32 / F16 / Q4_0 / Q4_1 formats supported by this llama.cpp revision,
- GGJT v3 legacy `.bin` models.

### Not supported

- **GGUF**,
- modern llama.cpp model metadata,
- GQA models that require grouped-query attention,
- architectures introduced long after this 2023 llama.cpp revision.

Examples such as TinyLlama and Llama-2-70B use GQA and are therefore not compatible with this backend as-is.

## GT 430 benchmark

OpenLLaMA-3B Q4_0, approximately 1.93 GB, context length 512:

| Configuration | Decode speed | GPU memory |
|---|---:|---:|
| CPU only, `-ngl 0` | **4.91 t/s** | — |
| GT 430, `-ngl 4` | **2.40 t/s** | ~270 MB |
| GT 430, `-ngl 8` | **1.58 t/s** | 531 MB |
| GT 430, `-ngl 10` | **1.35 t/s** | 664 MB |
| GT 430, `-ngl 12` | OOM | 797 MB |

On this machine, every offloaded layer added roughly **55 ms/token** across the PCIe 2.0 x1 link.

So yes, the GT 430 really is doing LLM inference — but this configuration is a **compatibility demonstration, not a speedup**.

## NVIDIA 390.xx driver notes

Fermi support on Linux generally means using NVIDIA's legacy 390.xx driver branch.

The tested machine used **390.157**. A few things can make OpenCL fail even when the kernel module appears to be loaded:

- Verify that `nvidia-smi` reports the expected 390.xx driver before testing.
- If `clinfo` shows no NVIDIA platform, confirm that the NVIDIA OpenCL userspace library is installed.
- If OpenCL enumeration crashes or silently fails, compare the `nvidia-uvm` major number in `/proc/devices` with the existing `/dev/nvidia-uvm` node.
- A stale `/dev/nvidia-uvm` created by a newer driver can break the legacy driver. On the test machine, an old 470-era node and the 390 module used different major numbers.
- NVIDIA 390.157 can require distro-maintained DKMS patches on newer kernels. The tested setup used Debian's legacy 390.xx packaging on Linux 6.12.

These are driver/runtime issues rather than llama.cpp bugs, so always make sure `clinfo -l` works first.

## Scope

This repository intentionally preserves an old llama.cpp codebase. It is useful for:

- Fermi / legacy-GPU experiments,
- OpenCL and CLBlast archaeology,
- retrocomputing,
- verifying how far old consumer GPUs can be pushed.

If you want current model support, GGUF, newer quantization schemes, or current llama.cpp features, use upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) instead.

## Related model

Ready-made legacy OpenLLaMA-3B files:

https://huggingface.co/eightman999/openllama-3b-ggml-legacy

## License and attribution

MIT, following the upstream llama.cpp license.

This repository contains the upstream llama.cpp source as of commit `2e6cd4b`, plus the Fermi/OpenLLaMA compatibility changes described above. See the Git history for the exact diff.
