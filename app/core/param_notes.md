## Params

Each id class (`idbook` or `smartid`) requires a different set of preprocessing parameters before being read by
Tesseract. Before an elegant solution is created, we are going to log the current best parameters for each class
here.

In the future, we hope not to require this complex input stream, but for the demo, this is fine.

---

### Smart IDs (`geom_params.idclass=0`)

```python
from idprocessing import pipeline

geom_params=pipeline.GeometryConfig(
    id_class=0,
    correction_angle=10,
    metadata_target_height=440
)

prep_params=pipeline.PreprocessConfig(
    thresh_c=4
)
```

### Green ID Books (`geom_params.idclass=1`)

```python
from idprocessing import pipeline

geom_params=pipeline.GeometryConfig(
    id_class=1,
    correction_angle=-100,
    metadata_target_height=350
)

prep_params=pipeline.PreprocessConfig(
    thresh_c=20
)
```