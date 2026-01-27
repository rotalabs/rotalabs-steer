# Model Support

## Pre-configured Models

The following models have pre-configured settings including recommended layers for each behavior:

### Qwen3 Family

| Model | Layers | Hidden Size | Refusal Layers |
|-------|--------|-------------|----------------|
| `Qwen/Qwen3-4B` | 36 | 2560 | 14-18 |
| `Qwen/Qwen3-8B` | 36 | 4096 | 14-18 |
| `Qwen/Qwen3-14B` | 48 | 5120 | 20-24 |

### DeepSeek

| Model | Layers | Hidden Size | Refusal Layers |
|-------|--------|-------------|----------------|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 48 | 5120 | 20-24 |

### Llama 3.1

| Model | Layers | Hidden Size | Refusal Layers |
|-------|--------|-------------|----------------|
| `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 | 14-16 |
| `meta-llama/Llama-3.1-70B-Instruct` | 80 | 8192 | 35-40 |

### Mistral

| Model | Layers | Hidden Size | Refusal Layers |
|-------|--------|-------------|----------------|
| `mistralai/Mistral-7B-Instruct-v0.2` | 32 | 4096 | 12-20 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4096 | 12-20 |

### Gemma 2

| Model | Layers | Hidden Size | Refusal Layers |
|-------|--------|-------------|----------------|
| `google/gemma-2-9b-it` | 42 | 3584 | 14-22 |

## Using Pre-configured Models

```python
from rotalabs_steer import get_model_config, MODEL_CONFIGS

# Get config by exact name
config = get_model_config("Qwen/Qwen3-8B")

# Get recommended layers for a behavior
refusal_layers = config.get_recommended_layers("refusal")
uncertainty_layers = config.get_recommended_layers("uncertainty")

# List all pre-configured models
print(list(MODEL_CONFIGS.keys()))
```

## Recommended Layers by Behavior

Each model has empirically-determined recommended layers:

| Behavior | Typical Layer Range | Notes |
|----------|---------------------|-------|
| `refusal` | Middle-upper layers | Where safety behaviors are encoded |
| `uncertainty` | Middle layers | Confidence representations |
| `tool_restraint` | Upper-middle layers | Tool-use decision making |
| `instruction_hierarchy` | Middle layers | Instruction processing |

## Auto-inference for Other Models

For models not in the pre-configured list, the package can infer configuration:

```python
from rotalabs_steer import infer_model_config
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("some/other-model")
config = infer_model_config(model)

print(f"Layers: {config.num_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Default recommended layers: {config.get_recommended_layers('refusal')}")
```

Auto-inference:
1. Tries to find a matching config by model name
2. Falls back to inspecting model structure
3. Returns middle-third layers as default recommendation

## Supported Architectures

The hook system supports models with these layer structures:

### Llama/Qwen/Mistral Style
```
model.model.layers[i]          # Residual stream
model.model.layers[i].mlp      # MLP output
model.model.layers[i].self_attn # Attention output
```

### GPT-2 Style
```
model.transformer.h[i]         # Residual stream
model.transformer.h[i].mlp     # MLP output
model.transformer.h[i].attn    # Attention output
```

## Adding Support for New Models

To add a new model configuration:

```python
from rotalabs_steer import ModelConfig, MODEL_CONFIGS

MODEL_CONFIGS["my-org/my-model"] = ModelConfig(
    name="my-org/my-model",
    num_layers=32,
    hidden_size=4096,
    recommended_layers={
        "refusal": [12, 14, 16, 18],
        "uncertainty": [10, 12, 14],
        "tool_restraint": [14, 16, 18],
        "instruction_hierarchy": [12, 14, 16],
    },
)
```

## Finding Optimal Layers

To find the best layers for your specific model and behavior:

```python
from rotalabs_steer.extraction import extract_caa_vectors
from rotalabs_steer.evaluation import strength_sweep

# Extract from many layers
all_layers = list(range(10, 25))  # Test layers 10-24
vectors = extract_caa_vectors(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=dataset,
    layer_indices=all_layers,
)

# Evaluate each layer
results = {}
for layer_idx in all_layers:
    vector = vectors[layer_idx]
    sweep = strength_sweep(
        model=model,
        tokenizer=tokenizer,
        steering_vector=vector,
        test_prompts=test_prompts,
        is_target_behavior_fn=is_refusal,
        strengths=[1.0],
    )
    results[layer_idx] = sweep[0]["behavior_rate"]

# Find best layers
sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Best layers:")
for layer, rate in sorted_layers[:5]:
    print(f"  Layer {layer}: {rate:.2%}")
```

## Memory Requirements

Approximate VRAM requirements for inference:

| Model Size | float16 | float32 |
|------------|---------|---------|
| 4B | ~8 GB | ~16 GB |
| 7-8B | ~16 GB | ~32 GB |
| 13-14B | ~28 GB | ~56 GB |
| 70B | ~140 GB | ~280 GB |

Steering vectors add negligible memory overhead (~1-4 MB per vector).
