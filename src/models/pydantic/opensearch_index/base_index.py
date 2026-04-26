from __future__ import annotations

from pydantic import BaseModel
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Type,
    get_args,
    get_origin,
    get_type_hints,
)

from .markers import Keyword, Text, Vector, Float, Boolean


class BaseIndex(BaseModel):
    class Meta:
        index_name: str = ""
        # Optional OpenSearch index settings override. If not provided,
        # IndexManager will fall back to its global defaults.
        settings: Optional[Dict[str, Any]] = None


def _iter_field_markers(model_class: Type[BaseIndex]) -> Dict[str, Any]:
    """
    Returns {field_name: marker_instance} for fields annotated with Annotated[..., marker].
    Uses get_type_hints(include_extras=True) to preserve Annotated metadata.
    """
    hints = get_type_hints(model_class, include_extras=True)
    out: Dict[str, Any] = {}
    for field_name, annotated_type in hints.items():
        if field_name.startswith("_"):
            continue
        if get_origin(annotated_type) is Annotated:
            args = get_args(annotated_type)
            # args: (base_type, *metadata)
            for meta in args[1:]:
                if isinstance(meta, (Text, Keyword, Vector, Float, Boolean)):
                    out[field_name] = meta
                    break
    return out


def get_vector_fields(model_class: Type[BaseIndex]) -> List[str]:
    markers = _iter_field_markers(model_class)
    vector_fields = [k for k, v in markers.items() if isinstance(v, Vector)]
    if vector_fields:
        return vector_fields
    # Backwards-compat fallback: heuristic by name
    return [
        field_name
        for field_name in model_class.model_fields.keys()
        if field_name.endswith("_vector") or "vector" in field_name.lower()
    ]


def get_text_fields(model_class: Type[BaseIndex]) -> List[str]:
    markers = _iter_field_markers(model_class)
    # Text fields here means BM25-capable (Text + Keyword).
    # If you need pure analyzed text only, filter by Text.
    weighted = [k for k, v in markers.items() if isinstance(v, (Text, Keyword))]
    if weighted:
        return weighted
    # Backwards-compat fallback: strings and list fields
    text_fields: List[str] = []
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        if field_type == str or (hasattr(field_type, "__origin__") and field_type.__origin__ == list):
            text_fields.append(field_name)
    return text_fields


def get_searchable_fields(model_class: Type[BaseIndex]) -> List[str]:
    markers = _iter_field_markers(model_class)
    searchable = [k for k, v in markers.items() if isinstance(v, (Text, Keyword))]
    if searchable:
        return searchable
    # Backwards-compat fallback
    searchable_fields: List[str] = []
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        if field_type in (str, Optional[str]) or (hasattr(field_type, "__origin__") and field_type.__origin__ == list):
            searchable_fields.append(field_name)
    return searchable_fields


def get_field_weights(model_class: Type[BaseIndex]) -> Dict[str, float]:
    """
    Returns text/keyword weight mapping from markers.
    """
    markers = _iter_field_markers(model_class)
    weights: Dict[str, float] = {}
    for k, v in markers.items():
        if isinstance(v, (Text, Keyword)):
            weights[k] = float(v.weight)
    return weights


def get_vector_weights(model_class: Type[BaseIndex]) -> Dict[str, float]:
    """
    Returns vector weight mapping from markers.
    """
    markers = _iter_field_markers(model_class)
    weights: Dict[str, float] = {}
    for k, v in markers.items():
        if isinstance(v, Vector):
            weights[k] = float(v.weight)
    return weights


def get_index_name(model_class: Type[BaseIndex]) -> str:
    return getattr(model_class.Meta, 'index_name', model_class.__name__.lower())


def build_field_types_from_markers(model_class: Type[BaseIndex]) -> Dict[str, Dict[str, Any]]:
    """
    Build OpenSearch mapping properties from markers.
    - Text -> text (with analyzer)
    - Keyword -> keyword
    - Vector -> knn_vector (with dimension + method)
    """
    markers = _iter_field_markers(model_class)
    out: Dict[str, Dict[str, Any]] = {}
    for field_name, marker in markers.items():
        if isinstance(marker, Text):
            out[field_name] = {"type": "text", "analyzer": marker.analyzer}
        elif isinstance(marker, Keyword):
            out[field_name] = {"type": "keyword"}
        elif isinstance(marker, Vector):
            out[field_name] = {
                "type": "knn_vector",
                "dimension": marker.dim,
                "method": {
                    "name": marker.method,
                    "space_type": marker.space_type,
                    "engine": marker.engine,
                },
            }
        elif isinstance(marker, Float):
            out[field_name] = {"type": "float"}
        elif isinstance(marker, Boolean):
            out[field_name] = {"type": "boolean"}
    return out
