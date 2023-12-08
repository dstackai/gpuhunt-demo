import dataclasses
import datetime
import time
from typing import List, Tuple

import gpuhunt
import gpuhunt.providers.vastai
import gpuhunt.providers.tensordock
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="gpuhunt | Find the cheapest GPU",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/dstackai/gpuhunt/issues",
        "Report a bug": "https://github.com/dstackai/gpuhunt-demo/issues",
        "About": "https://github.com/dstackai/gpuhunt",
    },
)


@st.cache_data
def get_catalog() -> gpuhunt.Catalog:
    catalog = gpuhunt.Catalog(balance_resources=True, auto_reload=True)
    catalog.load()
    catalog.add_provider(gpuhunt.providers.vastai.VastAIProvider())
    catalog.add_provider(gpuhunt.providers.tensordock.TensorDockProvider())
    return catalog


@st.cache_data
def get_all_offers() -> List[gpuhunt.CatalogItem]:
    return get_catalog().query(provider=PROVIDERS)


PROVIDERS = ["aws", "azure", "datacrunch", "gcp", "lambdalabs", "tensordock", "vastai"]
ALL_OFFERS = get_all_offers()
ALL_GPU_NAME = sorted(set(i.gpu_name for i in ALL_OFFERS if i.gpu_count > 0))
ALL_GPU_MEM = [0.0] + sorted(set(i.gpu_memory for i in ALL_OFFERS if i.gpu_count > 0))
ALL_GPU_COUNT = sorted(set(i.gpu_count for i in ALL_OFFERS))
ALL_CPU = sorted(set(i.cpu for i in ALL_OFFERS))
ALL_MEMORY = sorted(set(i.memory for i in ALL_OFFERS))
CACHE_TTL = 5 * 60

DEFAULT_PROVIDERS = [p for p in PROVIDERS if p not in ["vastai"]]
DEFAULT_GPUS = ["H100", "A100", "A6000", "A10", "A10G", "L40", "L4", "T4", "V100", "P100"]
DEFAULT_GPUS = [gpu for gpu in DEFAULT_GPUS if gpu in ALL_GPU_NAME]


def format_version(v: str) -> str:
    return f"{v[:4]}/{v[4:6]}/{v[6:]}"


f"""
# `gpuhunt`: find the cheapest GPU :dna:

Static catalog version: `{format_version(get_catalog().get_latest_version())}`
"""

with st.sidebar:
    """## Configuration"""
    providers = st.multiselect("Provider", options=PROVIDERS, default=DEFAULT_PROVIDERS)
    gpu_count = st.select_slider("GPU count", options=ALL_GPU_COUNT, value=(ALL_GPU_COUNT[0], ALL_GPU_COUNT[-1]))
    gpu_memory = st.select_slider("GPU memory", options=ALL_GPU_MEM, value=(ALL_GPU_MEM[0], ALL_GPU_MEM[-1]))
    gpu_name = st.multiselect("GPU name", options=ALL_GPU_NAME, default=DEFAULT_GPUS)
    spot = st.radio("Spot", options=["on-demand", "interruptable", "any"], index=0)
    cpu = st.select_slider("CPU", options=ALL_CPU, value=(ALL_CPU[0], ALL_CPU[-1]))
    memory = st.select_slider("RAM", options=ALL_MEMORY, value=(ALL_MEMORY[0], ALL_MEMORY[-1]))


@st.cache_data
def get_offers(
        providers: List[str],
        gpu_count: Tuple[int,
        int],
        gpu_memory: List[float],
        gpu_name: List[str],
        spot: str,
        cpu: Tuple[int],
        memory: Tuple[float],
        ttl: int,
) -> Tuple[datetime.datetime, pd.DataFrame]:
    offers = get_catalog().query(
        provider=providers or PROVIDERS,
        min_gpu_count=gpu_count[0], max_gpu_count=gpu_count[1],
        min_gpu_memory=gpu_memory[0], max_gpu_memory=gpu_memory[-1],
        min_cpu=cpu[0], max_cpu=cpu[-1],
        min_memory=memory[0], max_memory=memory[-1],
        gpu_name=gpu_name or None,
        spot={"interruptable": True, "on-demand": False, "any": None}[spot],
    )
    updated_at = datetime.datetime.utcnow()
    df = pd.DataFrame([dataclasses.asdict(i) for i in offers])
    if not df.empty:
        df["gpu"] = df.apply(lambda row: "" if row.gpu_count == 0 else f"{row.gpu_name} ({row.gpu_memory:g} GB)", axis=1)
        df = df[["provider", "price", "gpu_count", "gpu", "cpu", "memory", "spot", "location", "instance_name"]]
    return updated_at, df


updated_at, df = get_offers(providers, gpu_count, gpu_memory, gpu_name, spot, cpu, memory, round(time.time() / CACHE_TTL))
st.dataframe(
    df,
    column_config={
        "price": st.column_config.NumberColumn(format="$%.3f"),
    },
)
st.write(f"{len(df)} offers queried at `{updated_at} UTC`")


"""
Leave feedback at [dstackai/gpuhunt](https://github.com/dstackai/gpuhunt)

## How it works

`gpuhunt` aggregates offers from AWS, Azure, DataCrunch, GCP, and LambdaLabs every night.
TensorDock and VastAI offers are fetched in real-time.

You could also use `gpuhunt` as a Python library:

```bash
pip install gpuhunt
```

```python
>>> import gpuhunt
>>> offers = gpuhunt.query(
        min_gpu_count=1, min_gpu_memory=40,
        provider=["aws", "gcp"]
    )
```
"""

st.markdown("<style>footer::after { content: ' by 🧬 dstack' }</style>", unsafe_allow_html=True)
