# Tutorial: Create Custom Datasets

Build your own contrast pair datasets for custom behaviors.

!!! tip "Built-in Behaviors"
    The package includes 8 pre-built behaviors: `refusal`, `uncertainty`, `tool_restraint`, `instruction_hierarchy`, `formality`, `conciseness`, `creativity`, and `assertiveness`. Check [Datasets API](../api/datasets.md) before creating custom datasets.

## Understanding Contrast Pairs

A contrast pair consists of:
- **Positive**: Text exhibiting the target behavior
- **Negative**: Matched text NOT exhibiting the behavior

The key is that both texts should be similar except for the behavior difference.

## Creating a Dataset

### Basic Structure

```python
from rotalabs_steer.datasets import ContrastPairDataset, ContrastPair

# Create empty dataset
dataset = ContrastPairDataset(
    behavior="my_behavior",
    description="Description of what this behavior does",
)

# Add pairs
dataset.add_pair(
    positive="Response showing the target behavior",
    negative="Response NOT showing the target behavior",
)
```

### Example: Formality Behavior

Create a dataset for making responses more formal:

```python
formality_dataset = ContrastPairDataset(
    behavior="formality",
    description="Formal vs casual response style",
)

# Add contrast pairs
formality_dataset.add_pair(
    positive="I would be delighted to assist you with your inquiry regarding this matter.",
    negative="Sure, I can help you out with that!",
)

formality_dataset.add_pair(
    positive="Thank you for your patience. I shall investigate this issue promptly.",
    negative="Thanks for waiting! Let me look into that real quick.",
)

formality_dataset.add_pair(
    positive="I regret to inform you that this request cannot be accommodated at present.",
    negative="Sorry, can't do that right now.",
)

# Add more pairs...
```

### Example: Conciseness Behavior

```python
concise_dataset = ContrastPairDataset(
    behavior="conciseness",
    description="Brief vs verbose responses",
)

concise_dataset.add_pair(
    positive="Paris is the capital of France.",
    negative="The capital city of France, a country located in Western Europe, is Paris, which is also the largest city in France and serves as the country's political, economic, and cultural center.",
)

concise_dataset.add_pair(
    positive="Use list comprehensions for simple transformations.",
    negative="When you want to transform elements in a list in Python, one approach you might consider is using what's called a list comprehension, which is a concise way to create lists based on existing lists or other iterables.",
)
```

## Best Practices

### 1. Match Context

Both positive and negative should address the same underlying query:

```python
# GOOD: Same topic, different behavior
dataset.add_pair(
    positive="I cannot provide instructions for illegal activities.",
    negative="Here's how you could approach that...",
)

# BAD: Different topics
dataset.add_pair(
    positive="I cannot help with hacking.",
    negative="The weather today is sunny.",
)
```

### 2. Isolate the Behavior

The behavior should be the primary difference:

```python
# GOOD: Only formality differs
dataset.add_pair(
    positive="I appreciate your inquiry and shall respond forthwith.",
    negative="Thanks for asking, I'll get back to you soon.",
)

# BAD: Multiple differences (formality + length + content)
dataset.add_pair(
    positive="I appreciate your inquiry.",
    negative="Thanks! The answer is 42 and here's why...",
)
```

### 3. Cover Diverse Scenarios

Include various contexts where the behavior applies:

```python
# Different question types
dataset.add_pair(positive="...", negative="...")  # Factual questions
dataset.add_pair(positive="...", negative="...")  # Opinion questions
dataset.add_pair(positive="...", negative="...")  # How-to questions
dataset.add_pair(positive="...", negative="...")  # Creative requests
```

### 4. Balance the Dataset

Aim for 30-100 pairs. Too few may not capture the behavior; too many may overfit.

### 5. Use Metadata

Track sources and categories:

```python
dataset.add_pair(
    positive="...",
    negative="...",
    category="factual",
    source="manual",
    confidence="high",
)
```

## Loading from JSON

Create a JSON file:

```json
{
  "behavior": "formality",
  "description": "Formal vs casual response style",
  "pairs": [
    {
      "positive": "I would be delighted to assist you.",
      "negative": "Sure, happy to help!",
      "metadata": {"category": "greeting"}
    },
    {
      "positive": "I regret that this is not possible.",
      "negative": "Sorry, can't do that.",
      "metadata": {"category": "refusal"}
    }
  ]
}
```

Load it:

```python
dataset = ContrastPairDataset.load("./formality_pairs.json")
```

## Generating Pairs with LLMs

Use an LLM to help generate pairs:

```python
import openai

def generate_contrast_pair(behavior_description, example_context):
    prompt = f"""Generate a contrast pair for the behavior: {behavior_description}

Context: {example_context}

Return JSON with "positive" (exhibiting behavior) and "negative" (not exhibiting behavior).
Both should address the same underlying query/context."""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    return json.loads(response.choices[0].message.content)

# Generate pairs
pair = generate_contrast_pair(
    "Expressing uncertainty when unsure",
    "User asks about future stock prices"
)

dataset.add_pair(
    positive=pair["positive"],
    negative=pair["negative"],
    source="generated",
)
```

## Validating Your Dataset

Before extraction, validate your pairs:

```python
def validate_dataset(dataset):
    issues = []

    for i, pair in enumerate(dataset):
        # Check lengths are reasonable
        if len(pair.positive) < 10:
            issues.append(f"Pair {i}: Positive too short")
        if len(pair.negative) < 10:
            issues.append(f"Pair {i}: Negative too short")

        # Check they're different
        if pair.positive == pair.negative:
            issues.append(f"Pair {i}: Positive equals negative")

        # Check for obvious issues
        if pair.positive.strip() == "" or pair.negative.strip() == "":
            issues.append(f"Pair {i}: Empty content")

    return issues

issues = validate_dataset(dataset)
if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Dataset looks good!")
```

## Extract and Test

```python
from rotalabs_steer.extraction import extract_caa_vectors

# Extract vectors
vectors = extract_caa_vectors(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=dataset,
    layer_indices=[14, 15, 16],
)

# Save
vectors.save(f"./vectors/{dataset.behavior}/")

# Quick test
from rotalabs_steer import ActivationInjector

best = vectors.get_best()
injector = ActivationInjector(model, [best], strength=1.0)

test_prompt = "Can you help me with this?"

print("Without steering:")
# ... generate baseline

print("With steering:")
with injector:
    # ... generate steered
```

## Complete Example

```python
"""Create and test a custom 'enthusiasm' behavior."""

from rotalabs_steer.datasets import ContrastPairDataset
from rotalabs_steer.extraction import extract_caa_vectors
from rotalabs_steer import ActivationInjector

# Create dataset
dataset = ContrastPairDataset(
    behavior="enthusiasm",
    description="Enthusiastic vs neutral responses",
)

pairs = [
    ("I'd absolutely love to help you with that! This is such a great question!",
     "I can help you with that. Here's the information."),
    ("What a fantastic idea! I'm so excited to explore this with you!",
     "That's an interesting idea. Let me explain."),
    ("Oh wow, that's amazing! I can't wait to dive into this topic!",
     "That's a good topic. Here's what you should know."),
    ("This is incredibly interesting! Let me share some insights!",
     "This is interesting. Here are some insights."),
    # Add 20-30 more pairs...
]

for pos, neg in pairs:
    dataset.add_pair(positive=pos, negative=neg)

# Save dataset
dataset.save("./datasets/enthusiasm.json")

# Extract vectors (assuming model is loaded)
vectors = extract_caa_vectors(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=dataset,
    layer_indices=[14, 15, 16],
)

vectors.save("./vectors/enthusiasm/")

# Test
best = vectors.get_best()
injector = ActivationInjector(model, [best], strength=1.0)

test = "Tell me about machine learning"
inputs = tokenizer(test, return_tensors="pt").to(model.device)

print("Baseline:")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))

print("\nWith enthusiasm steering:")
with injector:
    out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
