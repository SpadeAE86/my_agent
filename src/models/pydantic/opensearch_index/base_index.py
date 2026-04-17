from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Type


class BaseIndex(BaseModel):
    class Meta:
        index_name: str = ""


def get_vector_fields(model_class: Type[BaseIndex]) -> List[str]:
    vector_fields = []
    for field_name, field_info in model_class.model_fields.items():
        if field_name.endswith('_vector') or 'vector' in field_name.lower():
            vector_fields.append(field_name)
    return vector_fields


def get_text_fields(model_class: Type[BaseIndex]) -> List[str]:
    text_fields = []
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        if field_type == str or (hasattr(field_type, '__origin__') and field_type.__origin__ == list):
            text_fields.append(field_name)
    return text_fields


def get_searchable_fields(model_class: Type[BaseIndex]) -> List[str]:
    searchable_fields = []
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        if field_type in (str, Optional[str]) or (hasattr(field_type, '__origin__') and field_type.__origin__ == list):
            searchable_fields.append(field_name)
    return searchable_fields


def get_index_name(model_class: Type[BaseIndex]) -> str:
    return getattr(model_class.Meta, 'index_name', model_class.__name__.lower())
