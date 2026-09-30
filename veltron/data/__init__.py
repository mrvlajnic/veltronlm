"""Data pipeline package."""

from .acquire import (
    Document,
    DocumentWriter,
    fetch_all_wikipedia,
    fetch_gutenberg,
    fetch_permissive_code,
)
from .filters import Deduper, FilterStats, detect_language, detect_pii, normalize, quality_report
from .loader import PackedDataset, TokenShardLoader, build_dataloaders
from .pipeline import (
    DATASET_VERSION,
    DEFAULT_MIXTURE,
    MixtureComponent,
    build_dataset,
    collect_raw,
    mix_documents,
    normalize_mixture,
    split_documents,
)
from .sources import SOURCES, licence_report

__all__ = [
    "Document",
    "DocumentWriter",
    "fetch_all_wikipedia",
    "fetch_gutenberg",
    "fetch_permissive_code",
    "Deduper",
    "FilterStats",
    "detect_language",
    "detect_pii",
    "normalize",
    "quality_report",
    "DATASET_VERSION",
    "DEFAULT_MIXTURE",
    "MixtureComponent",
    "build_dataset",
    "collect_raw",
    "mix_documents",
    "normalize_mixture",
    "split_documents",
    "SOURCES",
    "licence_report",
    "PackedDataset",
    "TokenShardLoader",
    "build_dataloaders",
]
