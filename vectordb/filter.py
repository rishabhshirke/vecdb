from typing import Any


OPERATORS = {
    "eq":  lambda v, val: v == val,
    "ne":  lambda v, val: v != val,
    "gt":  lambda v, val: v > val,
    "gte": lambda v, val: v >= val,
    "lt":  lambda v, val: v < val,
    "lte": lambda v, val: v <= val,
    "in":  lambda v, val: v in val,
    "nin": lambda v, val: v not in val,
}


def matches(metadata: dict, filters: dict) -> bool:
    """
    filters format:
      {"field": value}                      → exact match
      {"field": {"op": "gt", "val": 5}}    → operator match
      {"$and": [...], "$or": [...]}         → logical combinators
    """
    for key, condition in filters.items():
        if key == "$and":
            if not all(matches(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(metadata, sub) for sub in condition):
                return False
        elif isinstance(condition, dict) and "op" in condition:
            op = condition["op"]
            val = condition["val"]
            field_val = metadata.get(key)
            if field_val is None:
                return False
            if op not in OPERATORS:
                raise ValueError(f"Unknown filter operator '{op}'")
            if not OPERATORS[op](field_val, val):
                return False
        else:
            if metadata.get(key) != condition:
                return False
    return True
