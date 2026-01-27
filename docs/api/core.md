# Core API Reference

## SteeringVector

A steering vector for a specific behavior and layer.

```python
from rotalabs_steer import SteeringVector
```

### Constructor

```python
@dataclass
class SteeringVector:
    behavior: str              # Name of the behavior (e.g., "refusal")
    layer_index: int           # Layer this vector applies to
    vector: torch.Tensor       # The steering vector tensor
    model_name: str            # Model this was extracted from
    extraction_method: str     # Default: "caa"
    metadata: Dict[str, Any]   # Additional metadata
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `norm` | `float` | L2 norm of the vector |
| `dim` | `int` | Dimension of the vector |

### Methods

#### `normalize() -> SteeringVector`

Return an L2-normalized copy of the vector.

```python
normalized_vec = vector.normalize()
```

#### `scale(factor: float) -> SteeringVector`

Return a scaled copy of the vector.

```python
scaled_vec = vector.scale(0.5)
```

#### `to(device: str) -> SteeringVector`

Move vector to specified device.

```python
gpu_vec = vector.to("cuda")
```

#### `save(path: Path) -> None`

Save vector to disk. Creates two files:
- `{path}.json`: Metadata
- `{path}.pt`: Tensor

```python
vector.save("./vectors/refusal_layer_15")
```

#### `load(path: Path) -> SteeringVector` (classmethod)

Load vector from disk.

```python
vector = SteeringVector.load("./vectors/refusal_layer_15")
```

---

## SteeringVectorSet

Collection of steering vectors for a behavior across multiple layers.

```python
from rotalabs_steer import SteeringVectorSet
```

### Constructor

```python
class SteeringVectorSet:
    def __init__(
        self,
        behavior: str,
        vectors: Optional[List[SteeringVector]] = None
    )
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `layers` | `List[int]` | Sorted list of layer indices |
| `model_name` | `Optional[str]` | Model name from first vector |

### Methods

#### `add(vector: SteeringVector) -> None`

Add a vector to the set. Raises `ValueError` if behavior doesn't match.

#### `get(layer_index: int) -> Optional[SteeringVector]`

Get vector for a specific layer, or `None` if not found.

#### `get_best(metric: str = "norm") -> SteeringVector`

Get the "best" vector based on a metric. Currently supports `"norm"`.

#### `to(device: str) -> SteeringVectorSet`

Move all vectors to specified device.

#### `save(dir_path: Path) -> None`

Save all vectors to a directory.

#### `load(dir_path: Path) -> SteeringVectorSet` (classmethod)

Load all vectors from a directory.

---

## ActivationHook

Hook for capturing activations from transformer layers.

```python
from rotalabs_steer import ActivationHook
```

### Constructor

```python
class ActivationHook:
    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        component: Literal["residual", "mlp", "attn"] = "residual",
        token_position: Literal["last", "first", "all"] = "all",
    )
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The transformer model |
| `layer_indices` | `List[int]` | Layers to capture |
| `component` | `str` | Which part of layer to hook |
| `token_position` | `str` | Which tokens to capture |

### Methods

#### `attach() -> ActivationHook`

Attach hooks to model. Returns self for chaining.

#### `detach() -> None`

Remove all hooks from model.

#### `get_activations() -> Dict[int, torch.Tensor]`

Return dict mapping layer index to activation tensor.

### Context Manager

```python
with ActivationHook(model, [14, 15, 16]) as hook:
    model(**inputs)
activations = hook.get_activations()
```

---

## ActivationCache

Simple cache for storing captured activations.

```python
from rotalabs_steer import ActivationCache
```

### Methods

| Method | Description |
|--------|-------------|
| `store(name, tensor)` | Store tensor under name |
| `get(name)` | Retrieve tensor or None |
| `clear()` | Clear all stored activations |
| `keys()` | List stored names |

---

## ActivationInjector

Injects steering vectors into model activations during inference.

```python
from rotalabs_steer import ActivationInjector
```

### Constructor

```python
class ActivationInjector:
    def __init__(
        self,
        model: nn.Module,
        vectors: List[SteeringVector],
        strength: float = 1.0,
        injection_mode: Literal["all", "last", "first"] = "all",
    )
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The transformer model |
| `vectors` | `List[SteeringVector]` | Vectors to inject |
| `strength` | `float` | Multiplier for vectors |
| `injection_mode` | `str` | Where to inject |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `strength` | `float` | Get/set injection strength |

### Methods

#### `attach() -> ActivationInjector`

Attach injection hooks to model.

#### `detach() -> None`

Remove all injection hooks.

### Context Manager

```python
injector = ActivationInjector(model, [vector], strength=1.0)
with injector:
    outputs = model.generate(**inputs)
```

---

## MultiVectorInjector

Apply multiple steering vectors with independent strength control.

```python
from rotalabs_steer import MultiVectorInjector
```

### Constructor

```python
class MultiVectorInjector:
    def __init__(
        self,
        model: nn.Module,
        vector_sets: Dict[str, SteeringVectorSet],
        strengths: Optional[Dict[str, float]] = None,
        injection_mode: Literal["all", "last", "first"] = "all",
        default_layer: Optional[int] = None,
    )
```

### Methods

#### `set_strength(behavior: str, strength: float) -> None`

Set strength for a specific behavior.

#### `get_strength(behavior: str) -> float`

Get current strength for a behavior.

---

## extract_activations

Convenience function for extracting activations.

```python
from rotalabs_steer import extract_activations

activations = extract_activations(
    model=model,
    inputs=tokenized_inputs,
    layer_indices=[14, 15, 16],
    component="residual",
    token_position="last",
)
```

---

## ModelConfig

Configuration for model architectures.

```python
from rotalabs_steer import ModelConfig, MODEL_CONFIGS, get_model_config
```

### Fields

```python
@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    layer_template: str = "model.layers.{i}"
    residual_template: str = "model.layers.{i}"
    mlp_template: str = "model.layers.{i}.mlp"
    attn_template: str = "model.layers.{i}.self_attn"
    recommended_layers: Dict[str, List[int]]
```

### Functions

#### `get_model_config(model_name: str) -> ModelConfig`

Get config by model name. Tries exact match, then partial match.

#### `infer_model_config(model) -> ModelConfig`

Infer config from a loaded model.

### Pre-configured Models

```python
MODEL_CONFIGS = {
    "Qwen/Qwen3-8B": ModelConfig(...),
    "Qwen/Qwen3-4B": ModelConfig(...),
    "Qwen/Qwen3-14B": ModelConfig(...),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelConfig(...),
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(...),
    "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(...),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelConfig(...),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(...),
    "google/gemma-2-9b-it": ModelConfig(...),
}
```
