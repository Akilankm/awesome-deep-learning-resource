# NLP Notebook Presentation Standard

Every notebook should read like a chapter, not a dump of cells.

## Order

1. Title and learning objective
2. Mental model
3. Reference visual only when it improves understanding
4. Mechanism / mathematics
5. Change-one-thing-at-a-time table
6. Predict-before-run prompt
7. Executed implementation
8. Interpretation of the result
9. Failure modes and when-not-to-use guidance
10. Engineering takeaway

## Formatting rules

- Keep headings short and consistent.
- Use one main idea per markdown cell.
- Avoid decorative emoji-heavy headings.
- Avoid repeating generic production advice in every section.
- Use tables for cause/effect relationships.
- Use blockquotes for predictions or warnings.
- Prefer rendered data/plots over screenshots of code.
- Use external architecture diagrams only when they are license-safe and genuinely clearer than a local sketch.
- Keep attribution in `VISUAL_REFERENCES.md`.

## Learning test

A notebook succeeds when a learner can answer:

> If I change this control, what representation changes next, and what behavior should I expect downstream?

before executing the next cell.
