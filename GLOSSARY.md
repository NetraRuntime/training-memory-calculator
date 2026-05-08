<!-- markdownlint-disable MD036 -->

# Glossary and Calculator Field Guide

This guide explains the meaning of the website fields, result values, formulas,
and terminology used in [README.md](README.md) and [index.html](index.html).

Use [README.md](README.md) for the research citations and derivations. Use this
file when you want to know what each term means in practical language.

## Big Picture

The calculator estimates how much GPU memory is needed to finetune a language
model on one GPU. It separates memory into two broad groups:

- Model-state memory: weights, trainable parameters, gradients, master weights,
  and optimizer states. This part is mostly direct arithmetic from parameter
  counts and byte sizes.
- Runtime memory: activations, temporary buffers, framework overhead, and memory
  fragmentation. This part is an estimate because it depends on the exact
  training code and kernels.

The main result is not a guarantee that a model will fit. It is a transparent
estimate that shows where the memory is going.

The GPU Reference section turns that estimate into a hardware starting point by
matching it against known GPU VRAM sizes from the local `gpu.json` database. That
hardware list is also a reference, not a guarantee.

## Count Formats

Some input fields accept shorthand counts:

| Input | Meaning |
| --- | --- |
| `7000000000` | Seven billion. |
| `7B` | Seven billion, same as `7,000,000,000`. |
| `65B` | Sixty-five billion. |
| `1.5M` | One and a half million. |
| `2T` | Two trillion. |

The suffixes are decimal:

| Suffix | Multiplier |
| --- | ---: |
| `K` | 1,000 |
| `M` | 1,000,000 |
| `B` | 1,000,000,000 |
| `T` | 1,000,000,000,000 |

Memory output uses decimal units first, then binary units in parentheses. For
example, `33.53 GB (31.23 GiB)` means 33.53 decimal gigabytes, or 31.23 binary
gibibytes.

## Website Inputs

### Model Metadata

**Hugging Face repo or URL**

The model id on Hugging Face, such as `mistralai/Mistral-7B-v0.1`, or a full
Hugging Face model URL. The calculator uses this value only when you press
`Fetch`.

**Fetch**

Loads public metadata from Hugging Face. It tries to read exact safetensors
parameter counts and architecture values from `config.json`. Any fetched value
can still be edited manually.

**Fetch status**

Shows whether metadata was fetched successfully or whether the calculator fell
back to manual inputs.

### Training Method

**Full**

Full finetuning trains the model weights themselves. Gradients and optimizer
states are allocated for every trainable model parameter. This is the most
memory-heavy option.

**LoRA**

LoRA keeps the base model frozen and trains small low-rank adapter matrices. The
base model still has to be stored in GPU memory, but gradients and optimizer
states apply only to the adapter parameters.

**QLoRA**

QLoRA keeps the base model frozen in quantized storage, usually 4-bit NF4, and
trains LoRA adapters. This reduces base model storage compared with normal LoRA.

### Architecture

**Total parameters**

The total number of model parameters stored by the model. For dense models, this
is the normal parameter count. For MoE models, use total or sparse parameters,
not active parameters.

This value controls base model storage. In full finetuning, it also controls
trainable parameters unless `Trainable override` is set.

**Layers**

The number of Transformer blocks. This affects activation memory and LoRA target
parameter estimates.

Common Hugging Face config names include `num_hidden_layers`, `n_layer`, and
`num_layers`.

**Hidden size**

The width of the model's hidden states. In formulas this is usually `h`. Larger
hidden size increases attention weights, feed-forward weights, LoRA adapter
counts, and activation memory.

Common Hugging Face config names include `hidden_size`, `n_embd`, `d_model`, and
`dim`.

**Attention heads**

The number of query attention heads. In formulas this is usually `a`. It is used
with hidden size to compute `head_dim = hidden_size / attention_heads`.

**KV heads**

The number of key/value heads. If this equals `Attention heads`, the model uses
standard multi-head attention. If it is smaller, the model uses grouped-query or
multi-query attention.

The calculator uses this to estimate the shapes of `k_proj` and `v_proj`:

```text
kv_dim = kv_heads * hidden_size / attention_heads
```

**Intermediate size**

The feed-forward network width inside each Transformer block. In formulas this is
usually `I`. It controls the size of `gate_proj`, `up_proj`, and `down_proj` LoRA
targets.

Common Hugging Face config names include `intermediate_size`, `ffn_dim`,
`hidden_dim`, and `n_inner`.

**Vocabulary size**

The number of token ids the model knows. This is used when estimating embeddings,
when estimating from config, and when `lm_head` is selected as a LoRA target.

**Active params override**

Optional display-only value for active parameters, mainly useful for MoE models.
It does not reduce memory in this single-GPU calculator. Storage still uses total
parameters because all weights must be resident unless sharding, offload, or
expert parallelism is used.

**Model has MoE feed-forward layers**

Marks the model as mixture-of-experts. When enabled, expert feed-forward LoRA
targets are multiplied by the number of experts for storage estimates.

**Experts per layer**

The total number of experts in each MoE layer. In formulas this is usually `E`.
For single-GPU storage, all experts count toward stored parameters and trainable
LoRA parameters if they are targeted.

**Active experts per token**

The number of experts selected by the router for each token. In formulas this is
usually `K`. This affects active-per-token display values, but it does not reduce
single-GPU storage memory.

### Batch And Activations

**Sequence length**

The number of tokens processed per training example. In formulas this is usually
`s`. Longer sequences increase activation memory, and classic attention grows
strongly with sequence length.

**Microbatch size**

The number of examples processed by the GPU at once before gradient
accumulation. In formulas this is usually `b`. Gradient accumulation does not
multiply activation memory for a single microbatch.

**Activation mode**

Controls how activation memory is estimated.

| Option | Meaning |
| --- | --- |
| `Classic no recompute` | Uses the full cited Transformer activation estimate. Largest activation estimate. |
| `Selective recompute` | Assumes some expensive activations are recomputed during backward pass instead of stored. |
| `Full checkpointing estimate` | A smaller rough estimate for aggressive activation checkpointing. |
| `Manual override` | Ignores the formula and uses the value from `Manual activations GB`. |

**Kernel multiplier**

Scales the formula-based activation estimate to represent memory-efficient
kernels. It is not applied when `Manual override` is selected.

| Option | Meaning |
| --- | --- |
| `1.00 classic baseline` | Do not reduce the cited activation formula. |
| `0.75 memory-efficient attention` | Reduce the activation estimate by 25 percent. |
| `0.50 aggressive fused kernels` | Reduce the activation estimate by 50 percent. |
| `Custom` | Use the `Custom multiplier` value. |

**Custom multiplier**

The activation-memory multiplier used when `Kernel multiplier` is set to
`Custom`. Example: `0.6` means use 60 percent of the formula estimate.

**Manual activations GB**

The activation memory to use when `Activation mode` is `Manual override`. This is
useful when you measured activation memory from a real training script.

### Precision And Optimizer

**Base/model dtype**

The byte size used for model weight storage in full finetuning and normal LoRA.
It does not control QLoRA base storage, because QLoRA uses the `QLoRA base
storage` field instead.

| Option | Meaning |
| --- | --- |
| `BF16/FP16 (2 bytes)` | Store each base parameter in 2 bytes. |
| `FP32 (4 bytes)` | Store each base parameter in 4 bytes. |

**Optimizer preset**

Controls the bytes per trainable parameter for trainable weights, gradients,
master weights, and optimizer states.

| Preset | Meaning |
| --- | --- |
| `BF16 AdamW, no fp32 master` | Uses 2-byte parameters, 2-byte gradients, and two fp32 Adam moment tensors. Total model-state cost is 12 bytes per trainable parameter for full finetuning. |
| `FP16 mixed AdamW + fp32 master` | Adds a separate fp32 master weight copy. Total model-state cost is 16 bytes per trainable parameter for full finetuning. |
| `8-bit AdamW moments` | Estimates Adam moments as 8-bit states plus small metadata. Useful as a rough lower-memory optimizer estimate. |
| `Paged AdamW states` | Keeps the logical AdamW state cost but notes that runtime paging may move some pages to CPU under pressure. |

**Framework overhead %**

Adds a percentage on top of the subtotal. This represents temporary buffers,
allocator overhead, fragmentation, framework workspaces, and implementation
details not captured by the simple formulas.

**QLoRA base storage**

Visible for QLoRA. Controls how many bytes are used per frozen base model
parameter.

| Option | Meaning |
| --- | --- |
| `NF4 + double quantization` | 4-bit NF4 weights plus compressed quantization constants. Uses about `0.515869` bytes per parameter. |
| `NF4 without double quantization` | 4-bit NF4 weights plus uncompressed 32-bit constants per block. Uses `0.5625` bytes per parameter. |
| `8-bit base storage` | Simple 1 byte per parameter estimate. |

**Trainable override**

Optional direct override for trainable parameter count.

For full finetuning, this replaces `Total parameters` as the number of trainable
parameters. For LoRA and QLoRA, this replaces the calculated LoRA parameter count
before `Extra trainable params` are added.

Use this when you already know the exact number of trainable parameters from your
training library.

**Extra trainable params**

Additional trainable parameters for LoRA and QLoRA. Examples include trainable
biases, newly added token embeddings, or a trainable classification head. For
full finetuning, include custom extra parameters through `Trainable override`.

### GPU Reference Controls

**Hardware headroom %**

Extra memory margin added before the calculator searches for GPUs. For example,
if the estimate is `40 GiB` and headroom is `10%`, the GPU reference searches for
cards with at least `44 GiB` of VRAM.

This exists because the main estimate is not a guaranteed peak measurement. Real
training can need extra memory for temporary buffers, fragmentation, kernels,
drivers, framework choices, and dataloader behavior.

**Vendor filter**

Limits the GPU reference list to one vendor, such as NVIDIA or AMD, or shows all
vendors in the local GPU database.

### LoRA Targets

This section is visible for LoRA and QLoRA.

**LoRA rank**

The rank `r` of the low-rank adapter. Higher rank gives more trainable adapter
parameters. For one target matrix, LoRA parameters are:

```text
r * (d_in + d_out)
```

**LoRA alpha**

The scaling value used by LoRA during training. In this calculator it is display
only because alpha changes how adapter output is scaled, not how many parameters
the adapter has.

**q_proj**

The query projection in attention. Shape is usually `hidden_size -> hidden_size`.

**k_proj**

The key projection in attention. Shape is usually `hidden_size -> kv_dim`.

**v_proj**

The value projection in attention. Shape is usually `hidden_size -> kv_dim`.

**o_proj**

The output projection after attention. Shape is usually
`hidden_size -> hidden_size`.

**gate_proj**

One of the feed-forward projections in SwiGLU-style Transformer blocks. Shape is
usually `hidden_size -> intermediate_size`.

**up_proj**

Another feed-forward projection. Shape is usually
`hidden_size -> intermediate_size`.

**down_proj**

The feed-forward projection back down to hidden size. Shape is usually
`intermediate_size -> hidden_size`.

**lm_head**

The final projection from hidden states to vocabulary logits. Shape is usually
`hidden_size -> vocabulary_size`. It is often tied with input embeddings, but if
you train it with LoRA, it still adds adapter parameters.

**Q/V**

Selects `q_proj` and `v_proj`, a common small LoRA target set.

**Attention**

Selects `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

**All linear**

Selects attention and feed-forward linear layers. This matches the common QLoRA
style of targeting all linear modules inside Transformer blocks.

## Result Values

### Top Metrics

**Estimated total**

The final estimated GPU memory requirement:

```text
base storage
+ trainable weight storage
+ gradients
+ master weights
+ optimizer states
+ activations
+ framework overhead
```

**Trainable params**

The number of parameters that receive gradients and optimizer states. Full
finetuning usually uses all model parameters. LoRA and QLoRA usually use only
adapter parameters plus any extra trainable parameters.

**Base storage**

The memory used to store the base model weights. In QLoRA this is quantized base
storage. In full finetuning and normal LoRA this uses `Base/model dtype`.

**Activations**

The estimated memory needed to save forward-pass intermediate tensors for the
backward pass. This depends on sequence length, microbatch size, model shape,
checkpointing, and kernels.

**LoRA params**

The calculated LoRA adapter parameter count from the selected target modules,
rank, layer count, and architecture shape. This does not include `Extra trainable
params`.

**Active params display**

The displayed active parameter count. For dense models this is normally the same
as total parameters. For MoE models it can show active-per-token parameters. It
does not reduce memory by itself.

**LoRA active/token**

For dense models, this is the same as calculated LoRA parameters. For MoE models,
it estimates how many LoRA parameters are active for one token when only `K`
experts are selected. Storage and optimizer memory still use total trainable
LoRA parameters.

**Method**

The selected training method: `FULL`, `LORA`, or `QLORA`.

### Breakdown Rows

**Model weights**

The base model weights for full finetuning.

**Frozen base weights**

The base model weights for LoRA. They are frozen, meaning they do not receive
gradients or optimizer states, but they still occupy GPU memory.

**Quantized frozen base weights**

The quantized base model weights for QLoRA.

**Additional trainable weights**

Shown for full finetuning. In normal full finetuning this is `0` because the
trainable weights are already the model weights.

**LoRA adapter weights**

The trainable adapter weights for LoRA.

**QLoRA adapter weights**

The trainable adapter weights for QLoRA. QLoRA quantizes the base model, not the
adapter weights in this estimate.

**Gradients**

Memory used to store gradients for trainable parameters during backpropagation.
Frozen parameters do not get gradient storage.

**FP32 master weights**

A separate fp32 copy of trainable weights used by some mixed-precision training
setups. This row is nonzero for the `FP16 mixed AdamW + fp32 master` preset.

**Optimizer states**

Memory used by the optimizer. AdamW normally stores two moment tensors per
trainable parameter: first moment and second moment.

**Activations**

The activation memory estimate or manual activation value.

**Framework overhead**

The extra percentage added after the subtotal. This covers non-modeled memory.

**Estimated total**

The sum of all memory rows.

**Formula column**

Shows the arithmetic used for that row. This is there so the estimate can be
audited.

### Assumptions

The assumptions list explains what the calculator did not model, which optimizer
choice is active, which activation formula is active, which LoRA targets are
selected, and whether MoE storage rules are being applied.

### Sources

The sources list links to the papers and documentation behind the calculator.
The detailed reference list is in [README.md](README.md).

### GPU Reference Results

**Estimate**

The calculator's estimated total memory converted to GiB for comparison with GPU
VRAM.

**With headroom**

The estimate after applying `Hardware headroom %`. This is the number used when
deciding whether a GPU appears in the matching list.

**GPU table**

The table lists GPUs from `gpu.json` whose VRAM is at least the estimate plus
headroom. It sorts by the smallest matching VRAM first, so the first rows are
usually the closest practical hardware references.

If no GPU in the selected vendor filter has enough VRAM, the table shows the
largest entries instead and marks them as below the requirement.

**GPU database credit**

The GPU JSON database is credited to
[voidful/gpu-info-api](https://github.com/voidful/gpu-info-api).

## Symbols From The README

| Symbol | Meaning |
| --- | --- |
| `N` | Total stored model parameters. For MoE, this means sparse or total parameters. |
| `N_train` | Trainable parameters that receive gradients and optimizer states. |
| `N_lora` | Trainable LoRA adapter parameters. |
| `L` | Number of Transformer layers. |
| `h` | Hidden size. |
| `a` | Number of attention heads. |
| `a_kv` | Number of key/value heads. |
| `s` | Sequence length in tokens. |
| `b` | Per-device microbatch size. |
| `r` | LoRA rank. |
| `d_in` | Input dimension of a target linear layer. |
| `d_out` | Output dimension of a target linear layer. |
| `I` | Intermediate feed-forward width. |
| `E` | Total experts per MoE layer. |
| `K` | Active experts selected per token. |
| `W0` | The original pretrained weight matrix before LoRA is applied. |
| `A` | The LoRA down-projection matrix with shape `r x d_in`. |
| `B` | The LoRA up-projection matrix with shape `d_out x r`. |
| `Delta W` | The low-rank update added by LoRA, usually written as `B A`. |

## Terminology

**Activation checkpointing**

A technique that stores fewer activations during the forward pass and recomputes
some of them during the backward pass. It reduces memory at the cost of extra
compute.

**Activation memory**

Memory used by intermediate tensors saved from the forward pass so gradients can
be computed later.

**Active parameters**

The parameters used for one token or one forward pass. In MoE models this can be
much smaller than total parameters because only selected experts are active.

**AdamW**

An optimizer commonly used for training Transformers. It stores optimizer states,
usually two moment tensors, for each trainable parameter.

**Allocator fragmentation**

Memory lost because the GPU memory allocator has free memory split into chunks
that may not fit the next requested tensor.

**Attention heads**

Parallel attention subspaces inside a Transformer attention layer.

**Base model**

The pretrained model being finetuned.

**Base weight storage**

Memory used to store the base model weights.

**BF16**

Bfloat16, a 16-bit floating-point format. It uses 2 bytes per value and has a
wider exponent range than fp16.

**Block size**

The number of parameters grouped together for quantization metadata. The QLoRA
NF4 estimate uses block size 64.

**CPU offload**

Moving some tensors or optimizer states from GPU memory to CPU memory to reduce
GPU memory pressure.

**CUDA allocator**

The memory manager used by CUDA and PyTorch to reserve and reuse GPU memory.

**Decoder-only Transformer**

A Transformer architecture used by most autoregressive LLMs. It predicts the
next token from previous tokens.

**Dense model**

A model where every parameter in each layer is normally available to every token.
This contrasts with MoE models, where only selected experts are active per token.

**Double quantization**

In QLoRA, a method that also quantizes the quantization constants. This reduces
the metadata overhead for 4-bit weights.

**Expert**

One of several feed-forward subnetworks in an MoE layer.

**Expert parallelism**

A distributed-training strategy that spreads MoE experts across devices.

**Feed-forward network**

The MLP part of a Transformer block, usually made from `gate_proj`, `up_proj`,
and `down_proj` in LLaMA-style models.

**Finetuning**

Training a pretrained model further on a new dataset or task.

**FlashAttention**

A memory-efficient attention kernel family. It can reduce activation and
temporary memory compared with classic attention implementations.

**FP16**

Float16, a 16-bit floating-point format. It uses 2 bytes per value.

**FP32**

Float32, a 32-bit floating-point format. It uses 4 bytes per value.

**Framework overhead**

Extra memory used by PyTorch, CUDA, kernels, buffers, allocator behavior, and
other implementation details.

**Frozen weights**

Weights that are used during forward and backward computation but are not
updated by the optimizer.

**FSDP**

Fully Sharded Data Parallel. A distributed strategy that shards model weights,
gradients, and optimizer states across devices. It is not modeled by this
single-GPU calculator.

**Fused kernel**

A GPU kernel that combines multiple operations into one pass to reduce memory
movement and temporary tensors.

**GB**

Decimal gigabyte. `1 GB = 1,000,000,000 bytes`.

**GiB**

Binary gibibyte. `1 GiB = 1,073,741,824 bytes`.

**Gradient**

The derivative of the loss with respect to a trainable parameter. Optimizers use
gradients to update trainable weights.

**Gradient accumulation**

Running multiple microbatches before applying one optimizer update. It increases
effective batch size without multiplying activation memory for a single
microbatch.

**GPU database**

The local `gpu.json` file used by the GPU Reference section. It contains GPU
names, vendors, VRAM sizes, memory bandwidth, launch information, and other
hardware fields.

**Grouped-query attention**

An attention variant where multiple query heads share fewer key/value heads.
This reduces key/value projection size and KV cache size.

**Hugging Face config.json**

A model configuration file that stores architecture values such as layer count,
hidden size, attention heads, and vocabulary size.

**Hugging Face metadata**

Information fetched from the Hugging Face model API, including safetensors
parameter counts when available.

**Hardware headroom**

Extra VRAM margin added before selecting possible GPUs. It helps avoid treating a
near-equal estimate as a safe fit.

**Implementation-specific workspace**

Temporary memory required by a particular framework or kernel implementation.

**INT8**

An 8-bit integer storage format. In this calculator, the `8-bit base storage`
QLoRA option is modeled as 1 byte per parameter.

**Kernel**

A low-level GPU function that performs operations such as attention, matrix
multiplication, normalization, or loss computation.

**KV cache**

Stored key/value tensors used mainly during inference. Training memory is a
different problem, but `KV heads` still matters because it changes projection
shapes.

**LM head**

The final linear layer that maps hidden states to vocabulary logits.

**Liger**

A set of memory-efficient fused kernels used in some Transformer training setups.

**LLM**

Large language model.

**LoRA**

Low-Rank Adaptation. A parameter-efficient finetuning method that trains small
rank-`r` adapter matrices instead of updating the full base model.

**LoRA adapter**

The trainable low-rank matrices added to a frozen base layer.

**Master weights**

An fp32 copy of trainable weights used by some mixed-precision optimizers for
numerical stability.

**Microbatch**

The batch processed by the GPU in one forward/backward pass before gradient
accumulation.

**Mixed precision**

Training with a combination of lower-precision tensors, such as fp16 or bf16,
and fp32 tensors for selected states.

**MoE**

Mixture of Experts. A model architecture where each MoE layer has multiple expert
feed-forward networks and a router selects a subset per token.

**Model states**

Parameters, gradients, and optimizer states.

**Multi-query attention**

An attention variant with one or very few key/value heads shared by many query
heads.

**NF4**

NormalFloat4, the 4-bit data type used by QLoRA for normally distributed neural
network weights.

**Optimizer state storage**

Memory used by optimizer state tensors, such as AdamW first and second moments.

**Paged optimizer**

An optimizer that can use unified memory to page optimizer states between GPU and
CPU memory under pressure.

**Parameter**

A learned value in the model, such as a weight in a linear layer.

**Parameter-efficient finetuning**

Finetuning that updates only a small number of added or selected parameters
instead of the whole model.

**Pipeline parallelism**

A distributed strategy that splits model layers across devices. It is not modeled
by this single-GPU calculator.

**QLoRA**

Quantized LoRA. A finetuning method that stores the frozen base model in low-bit
quantized form and trains LoRA adapters.

**Quantization**

Representing weights with fewer bits than normal floating point. This reduces
storage memory but may require dequantization for computation.

**Quantization constants**

Scale or metadata values needed to reconstruct approximate floating-point values
from quantized weights.

**Residual states**

The non-model-state memory category from ZeRO: activations, temporary buffers,
and fragmented memory.

**RMSNorm**

A normalization layer used in many decoder-only LLMs. Fused RMSNorm kernels can
reduce temporary memory.

**Router**

The MoE component that chooses which experts process each token.

**Safetensors**

A tensor file format commonly used on Hugging Face. The Hugging Face API can
sometimes report exact parameter counts from safetensors metadata.

**SDPA**

Scaled Dot Product Attention. PyTorch includes optimized SDPA backends that may
use less memory than classic attention.

**Sequence length**

The number of tokens in each training example.

**Sharding**

Splitting model states across multiple devices so each device stores only part of
the whole state.

**Single-GPU estimate**

An estimate assuming all needed weights and states are resident on one GPU. It
does not apply distributed sharding or offload savings.

**SLM**

Small language model.

**Sparse parameters**

Total MoE parameters, including experts that are not active for a given token.

**Storage dtype**

The dtype used to store weights in memory.

**Tensor parallelism**

A distributed strategy that splits individual tensor operations across devices.
It is not modeled by this single-GPU calculator.

**Temporary buffers**

Short-lived tensors created by kernels, reductions, logits, loss computation, or
framework internals.

**Top-k experts**

The number of experts selected per token by an MoE router.

**Torch CUDA max memory allocated**

`torch.cuda.max_memory_allocated()` reports the peak GPU memory allocated by
PyTorch in a real run. It is the best way to validate the estimate for a specific
training script.

**Trainer**

The Hugging Face high-level training loop. Its memory behavior can differ from a
custom loop.

**Trainable parameters**

Parameters that receive gradients and optimizer updates.

**Trainable weight storage**

Memory used by trainable adapter or extra trainable weights outside the frozen
base model.

**Transformer block**

One repeated model layer containing attention, feed-forward layers, and
normalization.

**Vocabulary logits**

The raw output scores for each vocabulary token before softmax.

**VRAM**

Video RAM or GPU memory. This is the memory on the GPU that stores model weights,
gradients, optimizer states, activations, and temporary tensors during training.

**VRAM-only reference**

A hardware check based only on whether the GPU has enough listed memory. It does
not verify speed, kernel support, driver support, framework compatibility,
multi-GPU topology, or actual peak allocation in a real training script.

**ZeRO**

Zero Redundancy Optimizer. A family of memory optimizations that partitions model
states across devices. This calculator uses ZeRO's memory accounting idea but
does not model ZeRO partitioning.

## Exact Values Versus Estimates

More exact in this calculator:

- Base model storage from total parameters and dtype or quantization preset.
- Gradient storage from trainable parameters and optimizer preset.
- Optimizer state storage from trainable parameters and optimizer preset.
- LoRA parameter counts when the architecture fields and target modules match the
  real model.

Less exact in this calculator:

- Activation memory.
- Framework overhead.
- Temporary buffers.
- Memory effects of fused kernels and attention kernels.
- MoE routing runtime memory.
- Any distributed strategy such as ZeRO, FSDP, tensor parallelism, pipeline
  parallelism, expert parallelism, or CPU offload.

## Common Interpretation Tips

- If the estimate is only slightly below your GPU memory, it may still fail in a
  real run because temporary buffers and fragmentation can push peak memory over
  the limit.
- For MoE models, use total or sparse parameters for storage. Active parameters
  are useful for compute intuition, not for single-GPU resident weight storage.
- For QLoRA, base storage drops dramatically, but activations can still dominate
  at long sequence lengths.
- For LoRA and QLoRA, increasing rank increases adapter parameters linearly.
- For full finetuning, optimizer choice changes memory by hundreds of gigabytes
  on large models.
