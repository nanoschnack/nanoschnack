# Development Notes

## Profiling

Run training with cProfile:

```bash
python -m cProfile -o /tmp/profile.out model/training.py
```

Analyze the profile:

```bash
python - <<'PY'
import pstats
p = pstats.Stats('/tmp/profile.out')
p.sort_stats('tottime').print_stats(20)
PY
```
