"""
OpenSearch field markers used with typing.Annotated.

Example:
  from typing import Annotated, List, Optional
  from .markers import Text, Keyword, Vector

  description: Annotated[str, Text(2.0)]
  tags: Annotated[List[str], Keyword(2.5)]
  embedding: Annotated[Optional[List[float]], Vector(384, weight=2.0)]
"""


class Text:
    def __init__(self, weight: float = 1.0, analyzer: str = "standard"):
        self.weight = float(weight)
        self.analyzer = analyzer


class Keyword:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)


class Vector:
    def __init__(
        self,
        dim: int,
        weight: float = 1.0,
        *,
        engine: str = "lucene",
        method: str = "hnsw",
        space_type: str = "cosinesimil",
    ):
        self.dim = int(dim)
        self.weight = float(weight)
        self.engine = engine
        self.method = method
        self.space_type = space_type


class Float:
    """Numeric field stored as float in OpenSearch."""
    def __init__(self):
        pass


class Boolean:
    """Boolean field stored as boolean in OpenSearch."""
    def __init__(self):
        pass

