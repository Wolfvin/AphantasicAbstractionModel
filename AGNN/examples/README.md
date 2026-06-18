# AGNN Examples

Examples will be added as the skeleton gets implemented.

## Quick start (once implemented)

```python
from AGNN.core import init_brain, learn, process, inspect_engrams, reinforce

brain = init_brain(model_path="Qwen3-0.6B")

result = learn(
    question="What causes diabetes?",
    wrong="Eating sugar causes diabetes",
    correction="Insulin resistance causes type 2 diabetes"
)

answer = process("What is the main cause of type 2 diabetes?")
audit = inspect_engrams()
```
