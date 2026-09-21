# Data

The project notebook generates the full educational dataset deterministically when it runs.

After execution:

```text
data/
├── raw/
│   ├── support_tickets.csv
│   └── knowledge_base.csv
├── interim/
│   └── cleaned_tickets.csv
└── processed/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

This makes the project clone-and-run without a remote dataset download while still leaving every intermediate data stage inspectable on disk.
