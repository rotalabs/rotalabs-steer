# Extraction API Reference

## extract_caa_vector

Extract a steering vector for a single layer using Contrastive Activation Addition (CAA).

```python
from rotalabs_steer.extraction import extract_caa_vector
```

### Signature

```python
def extract_caa_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contrast_pairs: Union[ContrastPairDataset, List[dict]],
    layer_idx: int,
    token_position: Literal["last", "first", "mean"] = "last",
    batch_size: int = 1,
    show_progress: bool = True,
) -> SteeringVector
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `PreTrainedModel` | HuggingFace model |
| `tokenizer` | `PreTrainedTokenizer` | Corresponding tokenizer |
| `contrast_pairs` | `ContrastPairDataset` or `List[dict]` | Positive/negative pairs |
| `layer_idx` | `int` | Layer to extract from |
| `token_position` | `str` | Which token's activation to use |
| `batch_size` | `int` | Not currently used (kept for API compatibility) |
| `show_progress` | `bool` | Show progress bar |

### Returns

`SteeringVector` with the following metadata:
- `num_pairs`: Number of contrast pairs used
- `token_position`: Token position setting
- `pos_mean_norm`: L2 norm of mean positive activations
- `neg_mean_norm`: L2 norm of mean negative activations
- `vector_norm`: L2 norm of resulting steering vector

### Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rotalabs_steer.extraction import extract_caa_vector
from rotalabs_steer.datasets import load_refusal_pairs

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
refusal_pairs = load_refusal_pairs()

vector = extract_caa_vector(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=refusal_pairs,
    layer_idx=15,
    token_position="last",
)

print(f"Vector norm: {vector.norm}")
print(f"Vector dim: {vector.dim}")
```

---

## extract_caa_vectors

Extract steering vectors for multiple layers.

```python
from rotalabs_steer.extraction import extract_caa_vectors
```

### Signature

```python
def extract_caa_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contrast_pairs: Union[ContrastPairDataset, List[dict]],
    layer_indices: List[int],
    token_position: Literal["last", "first", "mean"] = "last",
    show_progress: bool = True,
) -> SteeringVectorSet
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `PreTrainedModel` | HuggingFace model |
| `tokenizer` | `PreTrainedTokenizer` | Corresponding tokenizer |
| `contrast_pairs` | `ContrastPairDataset` or `List[dict]` | Positive/negative pairs |
| `layer_indices` | `List[int]` | Layers to extract from |
| `token_position` | `str` | Which token's activation to use |
| `show_progress` | `bool` | Show progress bar |

### Returns

`SteeringVectorSet` containing vectors for all specified layers.

### Example

```python
from rotalabs_steer.extraction import extract_caa_vectors
from rotalabs_steer import get_model_config

# Get recommended layers for the model
config = get_model_config("Qwen/Qwen3-8B")
layers = config.get_recommended_layers("refusal")

# Extract vectors for all recommended layers
vectors = extract_caa_vectors(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=refusal_pairs,
    layer_indices=layers,
)

# Save entire set
vectors.save("./refusal_vectors/")

# Get best layer
best_vector = vectors.get_best()
print(f"Best layer: {best_vector.layer_index}")
```

---

## Token Position

The `token_position` parameter controls which token's activation is used:

| Value | Description | Use Case |
|-------|-------------|----------|
| `"last"` | Last token in sequence | Most common; captures full context |
| `"first"` | First token in sequence | For prefix-based behaviors |
| `"mean"` | Mean over all tokens | Aggregates entire sequence |

For most behaviors, `"last"` works best.

---

## Algorithm Details

The CAA extraction algorithm:

1. For each contrast pair:
   - Tokenize positive and negative texts
   - Run forward pass through model
   - Capture activations at specified layer
   - Select token position

2. Compute mean activations:
   - `pos_mean = mean(positive_activations)`
   - `neg_mean = mean(negative_activations)`

3. Compute steering vector:
   - `steering_vector = pos_mean - neg_mean`

This simple difference captures the direction in activation space that distinguishes the target behavior.
