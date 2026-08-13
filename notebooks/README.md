# Notebooks

**Analysis and figure exploration only.** No pipeline logic lives here — anything worth
keeping moves into `src/wfb/` where it is typed, tested and reusable. A notebook that
becomes load-bearing is a notebook that will silently break.

Start from:

```python
from wfb.serving.results_store import ResultsStore
from wfb.reporting.tables import full_report
from wfb.reporting.figures import generate_all

store = ResultsStore.load("experiments/results")
print(full_report(store))
```

Useful entry points:

| Task | Call |
|---|---|
| Load a dataset | `wfb.data.load_dataset(DataConfig(name="mosi"))` |
| Apply a corruption | `wfb.corruption.apply_plan(features, plan, stats, generator)` |
| Build the grid | `wfb.corruption.standard_grid()` |
| Sweep a checkpoint | `wfb.evaluation.run_sweep(model, datamodule, axes)` |
| Compare architectures | `wfb.evaluation.compare_all({"late": errors_a, "mult": errors_b})` |

Run one with `uv run jupyter lab` after `uv add --dev jupyterlab`.

Keep outputs cleared before committing — a diff full of base64 PNGs is unreviewable.
