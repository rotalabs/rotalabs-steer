# Datasets API Reference

## ContrastPair

A single contrast pair for steering vector extraction.

```python
from rotalabs_steer.datasets import ContrastPair
```

### Constructor

```python
@dataclass
class ContrastPair:
    positive: str              # Text exhibiting target behavior
    negative: str              # Text NOT exhibiting target behavior
    metadata: dict = {}        # Optional metadata
```

Raises `ValueError` if either text is empty.

---

## ContrastPairDataset

Collection of contrast pairs for a specific behavior.

```python
from rotalabs_steer.datasets import ContrastPairDataset
```

### Constructor

```python
class ContrastPairDataset:
    def __init__(
        self,
        behavior: str,
        pairs: Optional[List[ContrastPair]] = None,
        description: str = "",
    )
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `positives` | `List[str]` | All positive texts |
| `negatives` | `List[str]` | All negative texts |

### Methods

#### `add(pair: ContrastPair) -> None`

Add a contrast pair to the dataset.

#### `add_pair(positive: str, negative: str, **metadata) -> None`

Convenience method to add a pair from strings.

```python
dataset.add_pair(
    positive="I cannot help with that.",
    negative="Sure, here's how to do it.",
    category="harmful_request",
)
```

#### `save(path: Path) -> None`

Save dataset to JSON file.

#### `load(path: Path) -> ContrastPairDataset` (classmethod)

Load dataset from JSON file.

### Iteration

```python
for pair in dataset:
    print(pair.positive, pair.negative)

print(len(dataset))
print(dataset[0])
```

---

## EvaluationExample

A single evaluation example.

```python
from rotalabs_steer.datasets import EvaluationExample
```

### Constructor

```python
@dataclass
class EvaluationExample:
    prompt: str                # The prompt to test
    expected_behavior: bool    # True if behavior should trigger
    category: str = ""         # Optional category
    metadata: dict = {}        # Optional metadata
```

---

## EvaluationDataset

Dataset for evaluating steering effectiveness.

```python
from rotalabs_steer.datasets import EvaluationDataset
```

### Constructor

```python
class EvaluationDataset:
    def __init__(
        self,
        behavior: str,
        examples: Optional[List[EvaluationExample]] = None,
        description: str = "",
    )
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `positive_examples` | `List[EvaluationExample]` | Examples where behavior should trigger |
| `negative_examples` | `List[EvaluationExample]` | Examples where behavior should NOT trigger |

### Methods

#### `add(example: EvaluationExample) -> None`

Add an evaluation example.

#### `add_example(prompt, expected_behavior, category="", **metadata) -> None`

Convenience method.

#### `save(path: Path) -> None`

Save to JSON.

#### `load(path: Path) -> EvaluationDataset` (classmethod)

Load from JSON.

---

## Pre-built Datasets

### Refusal Pairs

```python
from rotalabs_steer.datasets import load_refusal_pairs

# Returns ContrastPairDataset with ~50 pairs
refusal_pairs = load_refusal_pairs()
```

Categories:
- `harmful_instructions`: Requests for harmful activities
- `illegal_activities`: Requests for illegal actions
- `dangerous_info`: Requests for dangerous information

### Uncertainty Pairs

```python
from rotalabs_steer.datasets import load_uncertainty_pairs

# Returns ContrastPairDataset with ~26 pairs
uncertainty_pairs = load_uncertainty_pairs()
```

Contrasts overconfident vs. appropriately uncertain responses.

### Tool Restraint Pairs

```python
from rotalabs_steer.datasets import load_tool_restraint_pairs

# Returns ContrastPairDataset with ~41 pairs
tool_pairs = load_tool_restraint_pairs()
```

Contrasts unnecessary tool use vs. direct responses.

### Instruction Hierarchy Pairs

```python
from rotalabs_steer.datasets import load_hierarchy_pairs

# Returns ContrastPairDataset with ~26 pairs
hierarchy_pairs = load_hierarchy_pairs()
```

Contrasts following user instructions that conflict with system instructions vs. maintaining system instruction priority.

### Formality Pairs

```python
from rotalabs_steer.datasets import load_formality_pairs

# Returns ContrastPairDataset with ~29 pairs
formality_pairs = load_formality_pairs()
```

Contrasts formal, professional communication vs. casual, informal style.

### Conciseness Pairs

```python
from rotalabs_steer.datasets import load_conciseness_pairs

# Returns ContrastPairDataset with ~25 pairs
conciseness_pairs = load_conciseness_pairs()
```

Contrasts brief, direct responses vs. verbose, overly detailed responses.

### Creativity Pairs

```python
from rotalabs_steer.datasets import load_creativity_pairs

# Returns ContrastPairDataset with ~30 pairs
creativity_pairs = load_creativity_pairs()
```

Contrasts imaginative, novel responses vs. conventional, generic responses.

### Assertiveness Pairs

```python
from rotalabs_steer.datasets import load_assertiveness_pairs

# Returns ContrastPairDataset with ~27 pairs
assertiveness_pairs = load_assertiveness_pairs()
```

Contrasts direct, confident responses vs. hedging, passive responses.

### Humor Pairs

```python
from rotalabs_steer.datasets import load_humor_pairs

# Returns ContrastPairDataset with ~31 pairs
humor_pairs = load_humor_pairs()
```

Contrasts witty, playful responses vs. serious, straightforward responses.

### Empathy Pairs

```python
from rotalabs_steer.datasets import load_empathy_pairs

# Returns ContrastPairDataset with ~28 pairs
empathy_pairs = load_empathy_pairs()
```

Contrasts warm, emotionally supportive responses vs. detached, clinical responses.

### Technical Depth Pairs

```python
from rotalabs_steer.datasets import load_technical_depth_pairs

# Returns ContrastPairDataset with ~22 pairs
technical_depth_pairs = load_technical_depth_pairs()
```

Contrasts detailed, expert-level technical responses vs. simplified, beginner-friendly responses.

---

## Creating Custom Datasets

```python
from rotalabs_steer.datasets import ContrastPairDataset, ContrastPair

# Create empty dataset
dataset = ContrastPairDataset(
    behavior="custom_behavior",
    description="My custom behavior dataset",
)

# Add pairs
dataset.add_pair(
    positive="Response exhibiting the behavior",
    negative="Response NOT exhibiting the behavior",
)

# Or add ContrastPair objects
dataset.add(ContrastPair(
    positive="Another positive example",
    negative="Another negative example",
    metadata={"source": "manual"},
))

# Save for later
dataset.save("./my_dataset.json")

# Load
loaded = ContrastPairDataset.load("./my_dataset.json")
```
