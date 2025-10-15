# Datasets

## Meditation-miniset-v0.2

[link](https://huggingface.co/datasets/BuildaByte/Meditation-miniset-v0.2)


```{sql}
select
  distinct unnest(string_split(suggested_techniques, ','))
from train
```

- body scan
- grounding exercises
- gentle breathing
- mindfulness
- visualization exercises
- concentration drills
- deep breathing
- progressive muscle relaxation
- guided imagery
- energizing breathwork
- visualization
- affirmations
