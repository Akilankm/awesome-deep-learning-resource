# NLP Teaching Standard — Causal, Visual, Engineering-First

The NLP track is designed to build **predictive mental models**, not only executable familiarity.

## Success criterion

A learner should be able to say **before running code**:

> “If I change this input, parameter, representation or threshold, I expect this downstream quantity to change in this direction, for this reason.”

## Required structure for concept notebooks

1. **Visual causal map** — input → mechanism → representation/state → output, plus control knob and failure lens.
2. **Change map** — explicit “change X → Y changes → consequence” relationships.
3. **Predict before running** — questions that force a hypothesis before seeing output.
4. **Mechanism, not API** — show the transformation the library call performs conceptually.
5. **When / when not** — conditions where the technique is useful and conditions where it misleads.
6. **Failure lens** — what breaks, how to detect it, and the next debugging question.
7. **Engineering takeaway** — a reusable rule of thumb.

## The mental-model loop

```text
observe
  ↓
build a causal hypothesis
  ↓
change one control
  ↓
predict downstream effect
  ↓
run / measure
  ↓
compare prediction with reality
  ↓
update mental model
```

This is the intended learning loop across the repository.
