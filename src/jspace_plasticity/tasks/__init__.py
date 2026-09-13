"""Task definitions. The reported experiment uses the closed-book geography task."""

from jspace_plasticity.tasks.closedbook_geo import (
    ENTITY_SPLITS,
    PROMPT_TEMPLATES,
    build_rows,
    load_split,
    normalize_text,
    select_entities,
    validate_manifest,
    validate_sources,
)

__all__ = [
    "ENTITY_SPLITS",
    "PROMPT_TEMPLATES",
    "build_rows",
    "load_split",
    "normalize_text",
    "select_entities",
    "validate_manifest",
    "validate_sources",
]
