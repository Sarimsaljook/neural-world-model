
# Neural World Model (NWM)

Neural World Model (NWM) is a cognitive world modeling system that goes beyond latent only predictive models like Dreamer, JEPA, or RSSM by explicitly representing entities, relations, events, intuition, memory, goals, and plans in a unified probabilistic world graph.

NWM is designed to understand, reason about, and explain the world in real time beyond predicting pixels or maximizing reward.

Most modern world models optimize latent prediction accuracy or long-horizon reward. NWM instead models the world in semantic units that cognition operates on:

- Persistent entities like objects and agents
- Typed relations 
  - contact, containment, support, proximity
- Discrete events 
  - contact_begin, inside_end, support_lost
- Continuous intuition fields 
  - risk, stability, slip, collision
- Long-term memory 
  - episodic, semantic, spatial, rules
- Goals as constraints and not scalar rewards
- Interpretable plans, probes, and actions

This lets NWM anticipate, explain, and monitor invariants even without acting.

---

## Model Architecture

```
                           Camera / Evidence
                                    ↓
                  Encoder: vision, features, instances
                                    ↓
                  Compiler: Entity-Relation-Frame Graph
                                    ↓
                 Event Inference: discrete semantic changes
                                    ↓
                 Intuition: risk, stability, collision, slip
                                    ↓
                 Memory: episodic, semantic, spatial, rules
                                    ↓
               Planning: constraints, programs, MPC, probing
                                    ↓
               Language: goal parsing, explanations, thoughts
```
---

## Code Structure

- Encoder
  - Vision encoder with instance segmentation
  - Stable top-K instance selection
  - Per-entity embeddings, masks, boxes

- Compiler
  - Converts raw evidence into a structured ERFG
  - Maintains persistent entity identity
  - Infers relations and events deterministically + probabilistically

- ERFG (Entity-Relation-Frame Graph)
  - Entities with pose, velocity, affordances, physical properties
  - Typed relations with confidence
  - Temporal consistency across frames

- Events
  - Discrete semantic transitions
    - `contact_begin`, `inside_begin`, `contains_end`

- Mechanisms
  - Router selects active mechanisms
  - Executor applies mechanism logic
  - Library of interaction primitives like contact, hinge, or rigid

- Intuition
  - Continuous fields estimating:
    - Risk
    - Instability
    - Slip likelihood
    - Collision likelihood
  - Computed before failure

- Memory
  - Episodic: recent events
  - Semantic: persistent facts
  - Spatial: layout & containment
  - Rule memory: learned invariants
  - Consolidation across time

- Planning 
  - Constraint construction & scoring
  - Event-program synthesis
  - MPC-style action proposal
  - Probing policies
  - Policy distillation buffer (scaffold)

- Language
  - Natural-language goal parsing
  - Constraint mapping
  - Real time thoughts grounded in world state

- Live Dashboard
  - Real-time visualization
  - Intuition values
  - Events and belief summaries
  - Goals, plans, probes
  - Human readable explanations

---

### Future Extensions

The repository also includes the basis for:

- Training pipelines
- Robotics integration
- Simulation environments
- APIs like FastAPI or Uvicorn
- Evaluation and benchmarks
- Multi Agent extensions

These are not required to run or evaluate the current system.

---

## Installation

### Requirements

```txt
opencv-python~=4.12.0.88
numpy~=2.2.6
torch~=2.11.0.dev20251215+cu128
pyttsx3~=2.99
uvicorn~=0.40.0
PyYAML~=6.0.3
transformers~=4.44.2
fastapi~=0.128.0
pydantic~=2.12.5
````

> Note: CUDA-enabled PyTorch is recommended for better real-time performance.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Live Demo

The primary way to run, test, and evaluate NWM is via the live dashboard.

```bash
python scripts/demo.py
```

This will:

* Open webcam
* Run the full NWM pipeline in real time
* Display the live dashboard window
* Accept goals via the console

Press `q` to quit.

---

## Using Goals 

Goals are constraints not rewards.

You type goals into the console while the demo is running.

### Example Goals

```text
avoid contact with the phone
```

```text
place the cup on the table within 3 seconds risk <= 20%
```

```text
keep the object stable
```

### What Goals Do

A goal:

* Defines forbidden or required relations
* Sets risk budgets
* Activates constraint scoring
* Influences actions like planning and probing

The system continuously evaluates:

* Whether the goal is currently satisfied
* What risks may violate it next
* What entities are involved

---

## Understanding the Dashboard

### Key Sections

* Entities:
  Number of persistent objects currently tracked.

* Events:
  Discrete semantic changes detected like contact begin.

* CScore:
  Constraint satisfaction score.

* Thoughts:
  Natural language explanations grounded in the world state.

* Intuition:

  * RiskMax: highest estimated risk
  * SlipMax: slip likelihood
  * Unstable: instability likelihood
  * Collide: collision likelihood

* Belief Summary:

  * High-level symbolic beliefs derived from events.

* Plan / Action / Probe:

  * Synthesized programs
  * Proposed actions
  * Information gathering probes

---

## Conceptual Difference vs Other World Models

| Aspect               | Dreamer / JEPA | NWM                   |
| -------------------- | -------------- | --------------------- |
| World representation | Latent vector  | Entity-relation graph |
| Events               | Implicit       | Explicit              |
| Risk                 | Reward-encoded | First-class           |
| Goals                | Scalar reward  | Constraints           |
| Explainability       | Low            | High                  |
| Safety               | Emergent       | Structural            |

---

## Summary

This is a cognitive world modeling system that is event centric and designed for accurate anticipation and reasoning while being easily interpretable.

---

## License

TBD: research and internal use

---

## Citation

If you find this repository useful, please currently cite as:

```
Neural World Model (NWM), 2026.
```

---

## Contact

For discussion or collaboration please reach out directly.
