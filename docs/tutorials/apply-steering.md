# Tutorial: Apply Steering at Inference

This tutorial covers different ways to apply steering vectors during inference.

## Prerequisites

You need a pre-extracted steering vector. See [Extract Your First Vector](extract-vector.md) or load one:

```python
from rotalabs_steer import SteeringVector

vector = SteeringVector.load("./refusal_vectors/layer_15")
```

## Basic Injection

The simplest way to apply steering:

```python
from rotalabs_steer import ActivationInjector

injector = ActivationInjector(
    model=model,
    vectors=[vector],
    strength=1.0,
)

# Use as context manager
with injector:
    outputs = model.generate(**inputs, max_new_tokens=100)
```

## Adjusting Strength

The `strength` parameter controls the intensity:

```python
# Subtle effect
injector = ActivationInjector(model, [vector], strength=0.5)

# Strong effect
injector = ActivationInjector(model, [vector], strength=2.0)

# Dynamic adjustment
injector.strength = 1.5
```

### Strength Guidelines

| Strength | Effect |
|----------|--------|
| 0.0 | No effect (baseline) |
| 0.25-0.5 | Subtle nudge |
| 0.5-1.0 | Moderate effect |
| 1.0-1.5 | Strong effect |
| 1.5-2.0 | Very strong effect |
| >2.0 | May cause incoherence |

## Injection Modes

Control where the vector is added:

```python
# Add to all token positions (default)
injector = ActivationInjector(
    model, [vector],
    strength=1.0,
    injection_mode="all",
)

# Add only to last token
injector = ActivationInjector(
    model, [vector],
    strength=1.0,
    injection_mode="last",
)

# Add only to first token
injector = ActivationInjector(
    model, [vector],
    strength=1.0,
    injection_mode="first",
)
```

### When to Use Each Mode

| Mode | Use Case |
|------|----------|
| `"all"` | General behavior modification |
| `"last"` | Generation-focused; affects next token prediction |
| `"first"` | Context-setting; affects how prompt is interpreted |

## Multi-Layer Injection

Apply vectors from multiple layers simultaneously:

```python
from rotalabs_steer import SteeringVectorSet

# Load vector set
vectors = SteeringVectorSet.load("./refusal_vectors/")

# Get vectors for specific layers
layer_14 = vectors[14]
layer_15 = vectors[15]

# Apply both
injector = ActivationInjector(
    model,
    [layer_14, layer_15],
    strength=1.0,
)
```

## Multi-Behavior Injection

Apply multiple behaviors with independent control:

```python
from rotalabs_steer import MultiVectorInjector, SteeringVectorSet

# Load vector sets for different behaviors
refusal_vectors = SteeringVectorSet.load("./refusal_vectors/")
uncertainty_vectors = SteeringVectorSet.load("./uncertainty_vectors/")

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

# Use
with injector:
    outputs = model.generate(**inputs)

# Adjust individual behaviors
injector.set_strength("refusal", 0.8)
injector.set_strength("uncertainty", 1.0)

# Check current strengths
print(injector.get_strength("refusal"))
```

## Manual Hook Management

For advanced use cases, manage hooks manually:

```python
injector = ActivationInjector(model, [vector], strength=1.0)

# Attach hooks
injector.attach()

# Generate multiple times with steering
for prompt in prompts:
    outputs = model.generate(**tokenize(prompt))

# Detach when done
injector.detach()
```

## Integration with Generation Parameters

Steering works with any generation configuration:

```python
with injector:
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=3,
    )
```

## Streaming Generation

Steering also works with streaming:

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_special_tokens=True)

with injector:
    model.generate(
        **inputs,
        max_new_tokens=100,
        streamer=streamer,
    )
```

## Comparing Steered vs Unsteered

```python
def compare(prompt, vector, strength=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Without steering
    with torch.no_grad():
        baseline = model.generate(**inputs, max_new_tokens=100)

    # With steering
    injector = ActivationInjector(model, [vector], strength=strength)
    with injector:
        with torch.no_grad():
            steered = model.generate(**inputs, max_new_tokens=100)

    print("BASELINE:")
    print(tokenizer.decode(baseline[0], skip_special_tokens=True))
    print("\nSTEERED:")
    print(tokenizer.decode(steered[0], skip_special_tokens=True))

compare("How do I hack a computer?", refusal_vector)
```

## Performance Considerations

1. **Hook overhead**: Minimal (~1-2% generation time)
2. **Memory**: Vectors are small (model hidden size × 4 bytes)
3. **Multiple vectors**: Linear overhead per vector

## Next Steps

- [LangChain Integration](langchain-integration.md) - Use with LangChain agents
- [Evaluation](../api/evaluation.md) - Measure steering effectiveness
