# LLM Finetuning Memory Calculator

Calculate the single-GPU memory needed to finetune dense and mixture-of-experts
large language models with full finetuning, LoRA, and QLoRA.

The calculator is intentionally conservative about what it calls exact. Model
state memory can be calculated directly from parameter counts and dtype choices.
Activation memory is a cited approximation because it depends on the concrete
Transformer implementation, attention kernel, checkpointing strategy, allocator,
temporary buffers, and framework runtime.

For plain-language explanations of every website input, result value, symbol,
and technical term, see [GLOSSARY.md](GLOSSARY.md).

## Abstract

Finetuning memory is dominated by two classes of tensors: model states and
residual states. Model states include parameters, gradients, and optimizer
states. Residual states include activations, temporary buffers, allocator
fragmentation, and implementation-specific workspaces. For a single GPU, this
calculator estimates the peak resident training memory as

```text
total_memory = base_weight_storage
             + trainable_weight_storage
             + gradient_storage
             + master_weight_storage
             + optimizer_state_storage
             + activation_storage
             + temporary/framework_overhead
```

The exact set of terms depends on the finetuning method:

- Full finetuning trains all model parameters.
- LoRA stores the frozen base model and trains low-rank adapter matrices.
- QLoRA stores the frozen base model in 4-bit quantized form and trains LoRA
  adapters through the quantized base.

## Scope

- Dense decoder-only Transformers such as LLaMA, Mistral, Gemma, Qwen dense, and
  similar architectures.
- Sparse mixture-of-experts models where total parameters and active parameters
  are treated separately.
- Full finetuning, LoRA, and QLoRA memory breakdowns.
- Hugging Face metadata fetch with manual fallback.
- Transparent activation estimates with checkpointing modes.

Not included in the current implementation:

- ZeRO, FSDP, tensor parallelism, pipeline parallelism, expert parallelism, or CPU
  offload partition formulas.
- Exact CUDA allocator fragmentation or framework temporary-buffer prediction.
- Exact MoE routing kernel memory. MoE storage is modeled; MoE activation memory
  remains an approximation unless manually overridden.

## Notation

| Symbol | Meaning |
| --- | --- |
| `N` | Total stored model parameters. For MoE this is sparse/total parameters, not active parameters. |
| `N_train` | Parameters that receive gradients and optimizer states. |
| `N_lora` | Trainable LoRA adapter parameters. |
| `L` | Number of Transformer layers. |
| `h` | Hidden size. |
| `a` | Attention heads. |
| `a_kv` | Key/value heads for grouped-query attention. |
| `s` | Sequence length in tokens. |
| `b` | Per-device microbatch size. Gradient accumulation does not multiply activation memory per microbatch. |
| `r` | LoRA rank. |
| `d_in`, `d_out` | Input and output dimensions of a target linear layer. |
| `I` | Intermediate/feed-forward width. |
| `E` | Number of MoE experts per layer. |
| `K` | Number of active experts per token. |

Memory units in the calculator are reported as decimal GB first, with GiB shown
for hardware comparison. The literature often mixes both, so examples below use
decimal GB unless stated otherwise.

## Model-State Memory

Rajbhandari et al. divide training memory into model states and residual states.
Model states are parameters, gradients, and optimizer states; residual states are
activations, temporary buffers, and fragmented memory [1].

For mixed-precision Adam with fp16 parameters and gradients plus fp32 master
weights and fp32 Adam moments, ZeRO gives the canonical per-parameter accounting
[1]:

```text
fp16 parameter copy      = 2 bytes * N_train
fp16 gradient            = 2 bytes * N_train
fp32 master parameter    = 4 bytes * N_train
fp32 Adam first moment   = 4 bytes * N_train
fp32 Adam second moment  = 4 bytes * N_train
------------------------------------------------
total                    = 16 bytes * N_train
```

For BF16 training without a separate fp32 master parameter copy, a common
single-GPU estimate is:

```text
bf16 parameter copy      = 2 bytes * N_train
bf16 gradient            = 2 bytes * N_train
fp32 Adam first moment   = 4 bytes * N_train
fp32 Adam second moment  = 4 bytes * N_train
------------------------------------------------
total                    = 12 bytes * N_train
```

The second formula is a practical preset rather than the exact ZeRO example;
frameworks differ on whether master weights are retained.

## Full Finetuning Algorithm

Full finetuning updates every parameter in the model. In the simplest single-GPU
case:

```text
N_train = N

full_memory = parameter_storage(N)
            + gradient_storage(N)
            + master_weight_storage(N)
            + optimizer_state_storage(N)
            + activation_storage
            + overhead
```

Examples before activations and temporary buffers:

```text
65B parameters, BF16 AdamW without fp32 master:
65e9 * 12 bytes = 780 GB

65B parameters, fp16 mixed AdamW with fp32 master:
65e9 * 16 bytes = 1040 GB = 1.04 TB
```

These examples explain why ordinary full finetuning of large LLMs is usually a
distributed-training problem.

## LoRA Algorithm

LoRA freezes the pretrained weight matrix `W0` and trains a low-rank update
`Delta W = B A`. For a target linear layer with `W0` in `R^(d_out x d_in)`, LoRA
uses `A` in `R^(r x d_in)` and `B` in `R^(d_out x r)`, so the trainable parameter
count for that target matrix is [2]:

```text
N_lora_layer = r * (d_in + d_out)
```

For many target layers:

```text
N_lora = sum_over_targets(r * (d_in + d_out))
```

The single-GPU LoRA memory estimate is:

```text
LoRA_memory = frozen_base_weight_storage(N)
            + trainable_adapter_weight_storage(N_lora)
            + gradient_storage(N_lora)
            + master_weight_storage(N_lora)
            + optimizer_state_storage(N_lora)
            + activation_storage
            + overhead
```

Important consequence: LoRA greatly reduces gradient and optimizer-state memory,
but the base model still has to be resident unless it is quantized, sharded, or
offloaded.

### Decoder-Only Target Shapes

For a LLaMA/Mistral-style dense decoder layer:

```text
head_dim = h / a
kv_dim   = a_kv * head_dim

q_proj:    d_in = h, d_out = h
k_proj:    d_in = h, d_out = kv_dim
v_proj:    d_in = h, d_out = kv_dim
o_proj:    d_in = h, d_out = h
gate_proj: d_in = h, d_out = I
up_proj:   d_in = h, d_out = I
down_proj: d_in = I, d_out = h
```

The calculator uses these shapes to estimate LoRA parameters for selected target
modules. QLoRA-style training commonly targets all linear layers in the
Transformer block, which Hugging Face PEFT exposes through
`target_modules="all-linear"` [6].

## QLoRA Algorithm

QLoRA backpropagates through a frozen 4-bit quantized base model into LoRA
adapters [3]. It separates storage dtype from compute dtype: the base weights are
stored in a low-precision format such as NF4 and dequantized to BF16 for matrix
multiplication, while gradients are computed only for the LoRA parameters [3].

For QLoRA with NF4 and double quantization, Dettmers et al. report that double
quantization reduces quantization constant overhead from `32/64 = 0.5` bits per
parameter to `8/64 + 32/(64*256) = 0.127` bits per parameter [3]. Therefore:

```text
NF4 + double quantization bits/parameter
  = 4 + 8/64 + 32/(64*256)
  = 4.126953125 bits/parameter

NF4 + double quantization bytes/parameter
  = 4.126953125 / 8
  = 0.515869140625 bytes/parameter
```

Without double quantization, using 32-bit constants and block size 64:

```text
NF4 without double quantization bits/parameter
  = 4 + 32/64
  = 4.5 bits/parameter

NF4 without double quantization bytes/parameter
  = 0.5625 bytes/parameter
```

The single-GPU QLoRA estimate is:

```text
QLoRA_memory = quantized_base_weight_storage(N)
             + trainable_adapter_weight_storage(N_lora)
             + gradient_storage(N_lora)
             + master_weight_storage(N_lora)
             + optimizer_state_storage(N_lora)
             + activation_storage
             + overhead
```

Sanity check:

```text
65B parameters, NF4 + double quantization base storage:
65e9 * 0.515869140625 bytes = 33.53 GB
```

The QLoRA paper reports that its method reduces a 65B model finetuning setup to
fit on a single 48 GB GPU, but the full peak depends on LoRA target modules,
sequence length, microbatch, activation checkpointing, paged optimizer behavior,
and implementation overhead [3].

## Activation Memory

Korthikanti et al. derive an approximate activation-storage formula for a single
stack Transformer with fp16 activations and dropout masks included [4]. For one
layer without tensor/pipeline parallelism:

```text
activation_bytes_per_layer = s * b * h * (34 + 5 * a * s / h)
```

For `L` layers:

```text
activation_bytes = L * s * b * h * (34 + 5 * a * s / h)
```

The same paper gives simplified recomputation variants [4]:

```text
selective_recompute_activation_bytes = 34 * s * b * h * L

full_activation_recompute_checkpoint_bytes ~= 2 * s * b * h * L
```

These formulas are most useful as transparent baselines. Modern training often
uses SDPA, FlashAttention, fused cross entropy, fused RMSNorm, Liger kernels, or
other memory-saving kernels. Those can reduce activation memory compared with the
classic formula, but exact memory depends on implementation details [5]. The
calculator therefore offers manual override and kernel multiplier fields.

## MoE Handling

Sparse mixture-of-experts models require two different parameter counts. Mixtral
8x7B, for example, has 47B total/sparse parameters, while each token uses about
13B active parameters because the router selects two experts per token [7].

For this single-GPU memory calculator:

```text
storage memory uses total/sparse parameters
optimizer memory uses all trainable parameters
active parameters are displayed for compute context only
```

This is important. Active parameters do not reduce single-GPU weight storage if
all experts are resident. They also do not reduce optimizer memory if LoRA or full
finetuning trains every expert. Active parameters matter for per-token compute,
routing, and expert-parallel/offload systems, which are out of scope for v1.

For MoE LoRA target estimates, the calculator multiplies expert FFN LoRA targets
by `E`, the number of experts. It also shows an active-per-token LoRA estimate by
multiplying expert FFN targets by `K`, the number of selected experts per token.
Only the total trainable LoRA parameter count is used for memory.

## Optimizer Presets

The browser calculator includes these presets:

| Preset | Parameter bytes | Gradient bytes | Master bytes | Optimizer bytes | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16/FP16 AdamW, no fp32 master | 2 | 2 | 0 | 8 | Practical BF16-style preset. |
| FP16 mixed AdamW, fp32 master | 2 | 2 | 4 | 8 | ZeRO's 16 bytes/parameter accounting [1]. |
| 8-bit AdamW moments | 2 | 2 | 0 | about 2.125 | Approximation for blockwise 8-bit optimizer states; metadata is implementation-dependent [8]. |
| Paged AdamW | 2 | 2 | 0 | 8 | Same logical states as AdamW, but may page to CPU under pressure [3, 8]. |

Paged optimizers are not modeled as a fixed memory subtraction because their GPU
residency depends on runtime pressure and unified-memory paging behavior. The UI
keeps the state in the total and explains the assumption.

## Hugging Face Metadata

The calculator accepts a Hugging Face repo id or URL. It tries to fetch:

- `https://huggingface.co/api/models/{repo}` for safetensors parameter metadata.
- `https://huggingface.co/{repo}/resolve/main/config.json` for architecture
  fields such as hidden size, layer count, attention heads, MoE expert counts,
  and vocabulary size.

Fetched values are editable. If exact total parameter metadata is unavailable,
the calculator can derive a rough decoder-only estimate from `config.json`, but it
labels this as an estimate.

## Limitations

The memory result is an estimate, not a replacement for measuring
`torch.cuda.max_memory_allocated()` in the exact training script.

Known sources of difference:

- PyTorch allocator caching and fragmentation.
- Temporary buffers for reductions, fused kernels, logits, and loss computation.
- Framework choices such as `Trainer`, Accelerate, DeepSpeed, FSDP, or custom
  loops.
- Attention implementation: classic attention, SDPA, FlashAttention, xFormers,
  Liger, or vendor kernels.
- Whether the optimizer keeps fp32 master weights.
- Whether gradients use fp16, bf16, or fp32 buffers.
- Whether embeddings, LM head, norms, biases, or newly added tokens are also
  trainable.
- MoE routing implementation and expert parallelism.

## References

[1] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. "ZeRO:
Memory Optimizations Toward Training Trillion Parameter Models." arXiv:1910.02054,
2019. [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)

[2] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
Shean Wang, Lu Wang, and Weizhu Chen. "LoRA: Low-Rank Adaptation of Large
Language Models." arXiv:2106.09685, 2021.
[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

[3] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. "QLoRA:
Efficient Finetuning of Quantized LLMs." arXiv:2305.14314, 2023.
[https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

[4] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael
Andersch, Mohammad Shoeybi, and Bryan Catanzaro. "Reducing Activation
Recomputation in Large Transformer Models." arXiv:2205.05198, 2022.
[https://arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)

[5] Hugging Face Transformers documentation. "GPU training." Accessed 2026-05-08.
[https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)

[6] Hugging Face PEFT documentation. "LoRA" and "Quantization." Accessed
2026-05-08.
[https://huggingface.co/docs/peft/developer_guides/lora](https://huggingface.co/docs/peft/developer_guides/lora)
and
[https://huggingface.co/docs/peft/developer_guides/quantization](https://huggingface.co/docs/peft/developer_guides/quantization)

[7] Albert Q. Jiang et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
[https://arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)

[8] Hugging Face bitsandbytes documentation. "8-bit optimizers" and "4-bit
quantization." Accessed 2026-05-08.
[https://huggingface.co/docs/bitsandbytes/explanations/optimizers](https://huggingface.co/docs/bitsandbytes/explanations/optimizers)
and
[https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit](https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit)
