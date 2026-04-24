# CLAUDE.md

## Purpose

This file provides **strict guidance for Claude (and other coding agents)** when working on the Multi-Firefly (MultiFF) Project. It defines expected conventions, structure, and reasoning priorities.

Claude should treat this as the **source of truth** for:
- Code style
- Analysis structure
- Modeling assumptions
- Behavioral and neural interpretation

---

## Core Principle

> Always align code and analysis with the goal of linking **behavior → computation → neural activity**.

Avoid writing generic ML or neuroscience code that is not explicitly grounded in the task.

---

## Task Summary

- Continuous navigation task with **multiple flashing targets (fireflies)**
- Partial observability → requires **memory + inference**
- Agent must **stop within 25 cm** to capture
- Observations are **slot-based representations of visible fireflies**

Firefly flashing is generated via:
```python
make_ff_flash_from_random_sampling
```

---

## What Claude Should Optimize For

### 1. Behavioral Interpretability
All outputs should map onto meaningful behavioral quantities:
- Planning (trajectory bias, future targeting)
- Learning (capture rate, latency)
- Policy (retry vs switch)

### 2. Clean, Reproducible Analysis
- Deterministic outputs when possible
- Minimal hidden state or side effects
- Clear data flow

### 3. Alignment With Existing Codebase
Claude must:
- Preserve function inputs/outputs
- Avoid unnecessary refactors
- Insert code in clearly marked locations

---

## Coding Rules (STRICT)

### Style
- Use **single quotes only**
- Prefer **NumPy vectorization** over loops
- Avoid redundant variables

### Functions
- Keep functions **small and modular**
- Use helper functions instead of long monolithic blocks
- DO NOT change function signatures unless explicitly asked

### Naming
- Use literal, descriptive names
- Time bins must be named:
  ```python
  bins_2d
  ```

### Data Output
- Always return **tidy pandas DataFrames** for analysis
- Preserve metadata (trial index, block index, condition labels)

---

## Plotting Rules

Use Matplotlib only.

Every plot must include:
- Axis labels
- Event alignment markers (e.g., stop, capture)
- Clean formatting suitable for publication

No decorative styling.

---

## Reinforcement Learning Constraints

### Architectures
Allowed:
- LSTM
- GRU
- LSTM + Set Transformer
- Feedforward (baseline only)

### Requirements
- Maintain hidden state across timesteps
- Support burn-in period
- Handle partial observability explicitly

### Actions
- Continuous: `[linear_velocity, angular_velocity]`
- Stopping defined by small velocity threshold

---

## Neural Analysis Rules

Focus on **population-level structure**, not single-neuron anecdotes.

### Required Methods
- PSTH (event-aligned)
- GPFA / latent dynamics
- dPCA
- GLM
- Decoding axes / regression

### Key Events
- Stop
- Capture

### Outputs
- Low-dimensional trajectories
- Condition-separated activity
- Event-locked responses

---

## Behavioral Analysis Rules

Always connect metrics to hypotheses:

### Planning
- Angular deviation to future targets
- Same-side stopping

### Learning
- Capture rate over time
- Time-to-capture reduction

### Policy
- Retry vs switch after misses
- Outcome-conditioned behavior

---

## What Claude Should NOT Do

- Do NOT introduce unnecessary abstractions
- Do NOT rewrite large sections of working code
- Do NOT use non-vectorized loops unless unavoidable
- Do NOT produce vague analysis without quantitative outputs
- Do NOT ignore task structure (fireflies, stopping, partial observability)

---

## Expected Code Output Format

Claude should:
- Provide **complete, runnable code blocks**
- Show **exact insertion points** when modifying code
- Keep outputs concise and structured

---

## Design Philosophy

- Prefer **clarity over cleverness**
- Prefer **behavioral relevance over algorithmic novelty**
- Prefer **simple models that explain behavior** over complex black boxes

## What Claude Should Ignore

Claude must explicitly ignore the following unless the user requests otherwise:

Large Data Objects / Heavy Artifacts
all_monkey_data
RL_models
Any large preloaded datasets, model checkpoints, or serialized objects


## Final Reminder

If a piece of code or analysis does not help answer:

> "How does the agent plan, learn, or adapt, and how is that reflected in neural activity?"

then it should not be included.


