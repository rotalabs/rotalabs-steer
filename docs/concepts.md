# Core Concepts

## Steering Vectors

A steering vector is a direction in the model's activation space that corresponds to a specific behavior. When added to the model's activations during inference, it shifts the model's behavior in that direction.

### How Steering Vectors Work

Transformer models process text through a series of layers. At each layer, the input is transformed into a high-dimensional activation vector. These activations encode information about:

- The input tokens
- Contextual relationships
- The model's "intentions" for generating output

By adding a carefully computed vector to these activations, we can shift the model's behavior without changing its weights.

### Mathematical Foundation

For a behavior B, we extract activations from:
- **Positive examples**: Text exhibiting the target behavior
- **Negative examples**: Text NOT exhibiting the target behavior

The steering vector is computed as:

```
steering_vector = mean(positive_activations) - mean(negative_activations)
```

This is the core of Contrastive Activation Addition (CAA).

## Contrast Pairs

Contrast pairs are the training data for steering vector extraction. Each pair consists of:

- **Positive text**: An example exhibiting the target behavior
- **Negative text**: A matched example NOT exhibiting the behavior

### Example: Refusal Behavior

```python
from rotalabs_steer.datasets import ContrastPair

pair = ContrastPair(
    positive="I cannot help with that request as it could cause harm.",
    negative="Sure, I'd be happy to help you with that.",
)
```

### Properties of Good Contrast Pairs

1. **Matched context**: Both texts should address similar prompts
2. **Clear behavioral difference**: The behavior should be the primary difference
3. **Diverse examples**: Cover various scenarios where the behavior applies

## Activation Extraction

The `ActivationHook` class captures activations during forward passes:

```python
from rotalabs_steer import ActivationHook

hook = ActivationHook(
    model=model,
    layer_indices=[14, 15, 16],
    component="residual",      # "residual", "mlp", or "attn"
    token_position="last",     # "last", "first", or "all"
)

with hook:
    model(**inputs)

activations = hook.get_activations()  # {layer_idx: tensor}
```

### Layer Selection

Different behaviors are encoded in different layers:
- **Early layers** (0-10): Low-level features, syntax
- **Middle layers** (10-25): Semantics, behavior patterns
- **Late layers** (25+): Output formatting

Recommended layers are pre-configured per model in `MODEL_CONFIGS`.

## Activation Injection

Steering vectors are applied using forward hooks that modify activations during inference:

```python
from rotalabs_steer import ActivationInjector

injector = ActivationInjector(
    model=model,
    vectors=[steering_vector],
    strength=1.0,              # Multiplier for the vector
    injection_mode="all",      # "all", "last", or "first" token
)

with injector:
    outputs = model.generate(**inputs)
```

### Injection Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `all` | Add to all token positions | General behavior modification |
| `last` | Add only to last token | Generation-focused changes |
| `first` | Add only to first token | Context-setting changes |

### Strength Parameter

The strength parameter controls how strongly the behavior is modified:

- `0.0`: No effect (baseline)
- `0.5-1.0`: Subtle to moderate effect
- `1.0-2.0`: Strong effect
- `>2.0`: May cause incoherence

Finding optimal strength requires evaluation (see [Evaluation](api/evaluation.md)).

## Multi-Vector Injection

Apply multiple behaviors simultaneously with independent control:

```python
from rotalabs_steer import MultiVectorInjector

injector = MultiVectorInjector(
    model=model,
    vector_sets={
        "refusal": refusal_vectors,
        "uncertainty": uncertainty_vectors,
    },
    strengths={
        "refusal": 1.0,
        "uncertainty": 0.5,
    },
)

# Adjust at runtime
injector.set_strength("refusal", 0.8)
```

## Vector Persistence

Steering vectors can be saved and loaded:

```python
# Single vector
vector.save("./vectors/refusal_layer_15")
vector = SteeringVector.load("./vectors/refusal_layer_15")

# Vector set (multiple layers)
vector_set.save("./vectors/refusal/")
vector_set = SteeringVectorSet.load("./vectors/refusal/")
```

Storage format:
- `.json`: Metadata (behavior, layer, model, extraction method)
- `.pt`: PyTorch tensor (the actual vector)
