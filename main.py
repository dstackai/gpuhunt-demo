import dataclasses
from typing import List, Tuple

import gpuhunt
import pandas as pd
import streamlit as st


"""
# `gpuhunt`

Visit [dstackai/gpuhunt](https://github.com/dstackai/gpuhunt) at GitHub

```bash
pip install gpuhunt
```

```python
>>> import gpuhunt
>>> offers = gpuhunt.query(min_gpu_count=1, min_gpu_memory=40, provider=["aws", "gcp"])
```
"""


@st.cache_data
def get_catalog() -> gpuhunt.Catalog:
    catalog = gpuhunt.Catalog(fill_missing=False, auto_reload=True)
    catalog.load()
    return catalog


@st.cache_data
def get_offers(providers: List[str], gpu_count: Tuple[int, int], gpu_memory: List[float], gpu_name: List[str], spot: List[str]) -> pd.DataFrame:
    offers = get_catalog().query(
        provider=providers or None,
        min_gpu_count=gpu_count[0], max_gpu_count=gpu_count[1],
        min_gpu_memory=gpu_memory[0], max_gpu_memory=gpu_memory[-1],
        gpu_name=gpu_name or None,
        spot=True if spot == ["yes"] else (False if spot == ["no"] else None),
    )
    return pd.DataFrame([dataclasses.asdict(i) for i in offers])


PROVIDERS = ["aws", "azure", "gcp", "nebius", "lambdalabs"]
ALL_GPUS = sorted(set(i.gpu_name for i in get_catalog().query(provider=PROVIDERS, min_gpu_count=1)))
ALL_GPU_MEM = [0.0] + sorted(set(i.gpu_memory for i in get_catalog().query(provider=PROVIDERS, min_gpu_count=1)))
ALL_GPU_COUNT = sorted(set(i.gpu_count for i in get_catalog().query(provider=PROVIDERS)))


"""
## Find the cheapest GPU
"""

providers = st.multiselect("Provider", options=PROVIDERS, default=PROVIDERS)
gpu_count = st.select_slider("GPU count", options=ALL_GPU_COUNT, value=(ALL_GPU_COUNT[0], ALL_GPU_COUNT[-1]))
gpu_memory = st.select_slider("GPU memory", options=ALL_GPU_MEM, value=(ALL_GPU_MEM[0], ALL_GPU_MEM[-1]))
gpu_name = st.multiselect("GPU name", options=ALL_GPUS, default=ALL_GPUS)
spot = st.multiselect("Spot", options=["yes", "no"], default=["yes", "no"])

st.dataframe(get_offers(providers, gpu_count, gpu_memory, gpu_name, spot))
