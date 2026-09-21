# NLP Teaching Standard

The goal is **predictive engineering intuition**, not notebook completion.

A learner should be able to state:

> If I change this control, I expect this representation or metric to move in this direction, because this mechanism changes.

## Required learning loop

1. **Mental model** — what enters, what transformation happens, what comes out.
2. **Control → effect** — change one thing and predict the consequence.
3. **Executed evidence** — run the example and inspect intermediate state.
4. **Interpretation** — explain why the observed result did or did not match the prediction.
5. **Boundary conditions** — when the technique helps, when it does not, and what fails first.
6. **Debugging order** — identify the failing layer before increasing complexity.

## Visual policy

A diagram is not automatically better than a table or executed plot.

- Use open-license external reference diagrams for established architectures when they are clearer and more polished.
- Use equations, tables and executed plots for weighting, preprocessing, metrics and data behavior.
- Do not create decorative diagrams merely to fill space.
- Keep attribution in [VISUAL_REFERENCES.md](VISUAL_REFERENCES.md).

## Success criterion

Students should leave a notebook able to **anticipate behavior before execution** and explain the observed result after execution.
