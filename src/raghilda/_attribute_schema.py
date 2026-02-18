from __future__ import annotations

from dataclasses import dataclass
import re
import types
from typing import (
    Annotated,
    Any,
    Iterable,
    Mapping,
    Optional,
    Union,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

AttributeScalar = str | int | float | bool
AttributeFilterValue = AttributeScalar | None
AttributeScalarType: TypeAlias = type[str] | type[int] | type[float] | type[bool]


@dataclass(frozen=True)
class AttributeFloatVectorType:
    dimension: int


@dataclass(frozen=True)
class AttributeStructType:
    fields: dict[str, "AttributeType"]


AttributeObjectValue: TypeAlias = dict[str, "AttributeValue"]
AttributeValue = AttributeScalar | list[float] | AttributeObjectValue | None
AttributeType: TypeAlias = (
    AttributeScalarType | AttributeFloatVectorType | AttributeStructType
)
AttributesSchemaSpec: TypeAlias = Mapping[str, Any] | type[Any]
AttributeFilter: TypeAlias = str | Mapping[str, Any]
_MISSING = object()


@dataclass(frozen=True)
class AttributeSpec:
    attribute_type: AttributeType
    nullable: bool
    required: bool
    default: AttributeValue = None


@dataclass(frozen=True)
class _SchemaItem:
    annotation: Any
    has_default: bool
    default: Any


@dataclass(frozen=True)
class _ParsedAttributeAnnotation:
    base: Any
    extras: tuple[Any, ...]
    nullable: bool
    is_annotated: bool


_ATTRIBUTE_SCALAR_TYPE_TO_NAME: dict[AttributeScalarType, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}
_ATTRIBUTE_NAME_TO_SCALAR_TYPE: dict[str, AttributeScalarType] = {
    value: key for key, value in _ATTRIBUTE_SCALAR_TYPE_TO_NAME.items()
}
_FLOAT_VECTOR_TYPE_PATTERN = re.compile(r"^float_vector\[(\d+)\]$")
_ATTRIBUTE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ATTRIBUTE_FILTER_RESERVED_KEYWORDS = {
    "AND",
    "OR",
    "IN",
    "IS",
    "NOT",
    "TRUE",
    "FALSE",
    "NULL",
}


def normalize_attributes_schema(
    attributes: Optional[AttributesSchemaSpec],
    *,
    reserved_columns: Iterable[str],
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeType]:
    attributes_spec = normalize_attributes_spec(
        attributes=attributes,
        reserved_columns=reserved_columns,
        allow_vector_types=allow_vector_types,
        allow_struct_types=allow_struct_types,
        allow_optional_values=allow_optional_values,
    )
    return {key: spec.attribute_type for key, spec in attributes_spec.items()}


def normalize_attributes_spec(
    attributes: Optional[AttributesSchemaSpec],
    *,
    reserved_columns: Iterable[str],
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeSpec]:
    schema_items = _attributes_schema_items(attributes)
    reserved = set(reserved_columns)
    spec: dict[str, AttributeSpec] = {}

    for key, item in schema_items.items():
        _validate_attribute_name(key, kind="Attribute column")
        if key in reserved:
            raise ValueError(f"Attribute column '{key}' is reserved")

        attribute_type, nullable = _parse_attribute_type(
            key=key,
            annotation=item.annotation,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )
        required = not item.has_default and not nullable
        default_value = item.default if item.has_default else None
        spec[key] = _build_attribute_spec(
            key=key,
            attribute_type=attribute_type,
            nullable=nullable,
            required=required,
            default_value=default_value,
            allow_optional_values=allow_optional_values,
            default_none_error_message=(
                f"Default None for attribute '{key}' requires an optional type annotation"
            ),
        )

    return spec


def _attributes_schema_items(
    attributes: Optional[AttributesSchemaSpec],
) -> dict[str, _SchemaItem]:
    if attributes is None:
        return {}
    if isinstance(attributes, Mapping):
        out: dict[str, _SchemaItem] = {}
        for key, value in attributes.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(
                        f"Attribute schema tuple for '{key}' must be (type, default)"
                    )
                out[key] = _SchemaItem(
                    annotation=value[0],
                    has_default=True,
                    default=value[1],
                )
            else:
                out[key] = _SchemaItem(
                    annotation=value,
                    has_default=False,
                    default=None,
                )
        return out
    if isinstance(attributes, type):
        try:
            annotations = get_type_hints(attributes, include_extras=True)
        except Exception as e:
            raise ValueError(
                f"Failed to parse attribute annotations from '{attributes.__name__}': {e}"
            )
        class_vars = vars(attributes)
        out: dict[str, _SchemaItem] = {}
        for key, annotation in annotations.items():
            has_default = key in class_vars
            out[key] = _SchemaItem(
                annotation=annotation,
                has_default=has_default,
                default=class_vars[key] if has_default else None,
            )
        return out
    raise ValueError("attributes must be a mapping or a class with type annotations")


def _build_attribute_spec(
    *,
    key: str,
    attribute_type: AttributeType,
    nullable: bool,
    required: bool,
    default_value: Any,
    allow_optional_values: bool,
    default_none_error_message: str,
) -> AttributeSpec:
    if not allow_optional_values and not required:
        raise ValueError(
            f"Optional attribute values are not supported for '{key}' in this backend"
        )
    normalized_default: AttributeValue = None
    if not required:
        if default_value is None and not nullable:
            raise ValueError(default_none_error_message)
        normalized_default = _normalize_attribute_value(
            key,
            default_value,
            attribute_type,
            context="default attribute",
            allow_none=nullable,
        )
    return AttributeSpec(
        attribute_type=attribute_type,
        nullable=nullable,
        required=required,
        default=normalized_default,
    )


def _parse_attribute_type(
    *,
    key: str,
    annotation: Any,
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> tuple[AttributeType, bool]:
    parsed = _parse_attribute_annotation(annotation)

    if isinstance(parsed.base, type) and parsed.base in _ATTRIBUTE_SCALAR_TYPE_TO_NAME:
        return cast(AttributeScalarType, parsed.base), parsed.nullable

    if isinstance(parsed.base, Mapping):
        if parsed.extras:
            raise ValueError(
                f"Unsupported attribute annotation for '{key}': {annotation}"
            )
        if not allow_struct_types:
            raise ValueError(
                f"Object attribute types are not supported for '{key}' in this backend"
            )
        return (
            _parse_struct_annotation(
                key=key,
                annotation=parsed.base,
                allow_vector_types=allow_vector_types,
                allow_struct_types=allow_struct_types,
            ),
            parsed.nullable,
        )

    if parsed.is_annotated:
        vector_type = _parse_vector_annotation(parsed.base, parsed.extras)
        if vector_type is None:
            raise ValueError(
                f"Unsupported attribute annotation for '{key}': {annotation}"
            )
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return vector_type, parsed.nullable

    if isinstance(parsed.base, AttributeFloatVectorType):
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return parsed.base, parsed.nullable

    raise ValueError(
        f"Attribute type for '{key}' must be one of: str, int, float, bool, optional scalar (T | None), object mapping, or Annotated[list[float], N]"
    )


def _parse_struct_annotation(
    *,
    key: str,
    annotation: Mapping[str, Any],
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> AttributeStructType:
    fields: dict[str, AttributeType] = {}
    for field_name, field_annotation in annotation.items():
        _validate_attribute_name(
            field_name,
            kind=f"Object attribute field for '{key}'",
        )
        field_type, field_nullable = _parse_attribute_type(
            key=f"{key}.{field_name}",
            annotation=field_annotation,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )
        if field_nullable:
            raise ValueError(
                f"Optional object attribute field '{key}.{field_name}' is not supported"
            )
        fields[field_name] = field_type
    return AttributeStructType(fields=fields)


def _parse_vector_annotation(
    base: Any, extras: tuple[Any, ...]
) -> Optional[AttributeType]:
    base_origin = get_origin(base)
    base_args = get_args(base)
    if base_origin is not list or len(base_args) != 1 or base_args[0] is not float:
        return None

    dimensions = [
        x for x in extras if isinstance(x, int) and not isinstance(x, bool) and x > 0
    ]
    if len(dimensions) != 1:
        raise ValueError(
            "Vector attribute annotations must include exactly one positive integer dimension"
        )

    return AttributeFloatVectorType(dimension=dimensions[0])


def _parse_attribute_annotation(annotation: Any) -> _ParsedAttributeAnnotation:
    base, nullable = _unwrap_optional_annotation(annotation)

    origin = get_origin(base)
    if origin is not Annotated:
        return _ParsedAttributeAnnotation(
            base=base,
            extras=(),
            nullable=nullable,
            is_annotated=False,
        )

    args = get_args(base)
    if len(args) == 0:
        raise ValueError(f"Unsupported annotation '{annotation}'")
    annotated_base, annotated_nullable = _unwrap_optional_annotation(args[0])
    return _ParsedAttributeAnnotation(
        base=annotated_base,
        extras=tuple(args[1:]),
        nullable=nullable or annotated_nullable,
        is_annotated=True,
    )


def _unwrap_optional_annotation(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin not in {Union, types.UnionType}:
        return annotation, False

    args = get_args(annotation)
    non_none_args = [arg for arg in args if arg is not type(None)]
    has_none = len(non_none_args) != len(args)
    if not has_none:
        return annotation, False
    if len(non_none_args) != 1:
        raise ValueError(
            f"Unsupported union annotation '{annotation}'. Only optional unions like T | None are supported."
        )
    return non_none_args[0], True


def attributes_schema_to_json_dict(
    attributes_schema: Mapping[str, AttributeType],
) -> dict[str, Any]:
    return {
        key: _attribute_type_to_json_value(value)
        for key, value in attributes_schema.items()
    }


def _attribute_type_to_name(attribute_type: AttributeType) -> str:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"float_vector[{attribute_type.dimension}]"
    if isinstance(attribute_type, AttributeStructType):
        raise ValueError("Structured object attributes are serialized as mappings")
    return _ATTRIBUTE_SCALAR_TYPE_TO_NAME[attribute_type]


def _attribute_type_to_json_value(attribute_type: AttributeType) -> Any:
    if isinstance(attribute_type, AttributeStructType):
        return {
            "type": "struct",
            "fields": {
                key: _attribute_type_to_json_value(value)
                for key, value in attribute_type.fields.items()
            },
        }
    return _attribute_type_to_name(attribute_type)


def attributes_schema_from_json_dict(
    attributes_schema_json: Mapping[str, Any],
    *,
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
) -> dict[str, AttributeType]:
    schema: dict[str, AttributeType] = {}
    for key, value in attributes_schema_json.items():
        _validate_attribute_name(key, kind="Attribute column")
        schema[key] = _attribute_type_from_json_value(
            key=key,
            value=value,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )

    return schema


def attributes_spec_to_json_dict(
    attributes_spec: Mapping[str, AttributeSpec],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in attributes_spec.items():
        payload: dict[str, Any] = {
            "type": _attribute_type_to_json_value(spec.attribute_type),
            "nullable": spec.nullable,
            "required": spec.required,
        }
        if not spec.required:
            payload["default"] = spec.default
        out[key] = payload
    return out


def attributes_spec_from_json_dict(
    attributes_spec_json: Mapping[str, Any],
    *,
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeSpec]:
    out: dict[str, AttributeSpec] = {}
    for key, payload in attributes_spec_json.items():
        _validate_attribute_name(key, kind="Attribute column")
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Attribute spec for '{key}' must be a mapping with keys: type, nullable, required, default"
            )
        type_value = payload.get("type")
        if not isinstance(type_value, (str, Mapping)):
            raise ValueError(
                f"Attribute spec for '{key}' must include 'type' as a string or mapping"
            )
        nullable = payload.get("nullable")
        if not isinstance(nullable, bool):
            raise ValueError(
                f"Attribute spec for '{key}' must include boolean 'nullable'"
            )
        required = payload.get("required")
        if not isinstance(required, bool):
            raise ValueError(
                f"Attribute spec for '{key}' must include boolean 'required'"
            )
        if not allow_optional_values and not required:
            raise ValueError(
                f"Optional attribute values are not supported for '{key}' in this backend"
            )

        attribute_type = _attribute_type_from_json_value(
            key=key,
            value=type_value,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )
        out[key] = _build_attribute_spec(
            key=key,
            attribute_type=attribute_type,
            nullable=nullable,
            required=required,
            default_value=payload.get("default"),
            allow_optional_values=allow_optional_values,
            default_none_error_message=(
                f"Default None for attribute '{key}' requires nullable=true in serialized spec"
            ),
        )
    return out


def _parse_attribute_type_name(type_name: str) -> Optional[AttributeType]:
    match = _FLOAT_VECTOR_TYPE_PATTERN.fullmatch(type_name)
    if match is not None:
        return AttributeFloatVectorType(dimension=int(match.group(1)))
    return None


def _attribute_type_from_json_value(
    *,
    key: str,
    value: Any,
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> AttributeType:
    if isinstance(value, str):
        if value in _ATTRIBUTE_NAME_TO_SCALAR_TYPE:
            return _ATTRIBUTE_NAME_TO_SCALAR_TYPE[value]

        vector_type = _parse_attribute_type_name(value)
        if vector_type is None:
            raise ValueError(
                f"Attribute type for '{key}' must be one of: str, int, float, bool, float_vector[N], or struct"
            )
        if isinstance(vector_type, AttributeFloatVectorType) and not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return vector_type

    if isinstance(value, Mapping):
        type_name = value.get("type")
        if type_name != "struct":
            raise ValueError(
                f"Attribute type mapping for '{key}' must include type='struct'"
            )
        if not allow_struct_types:
            raise ValueError(
                f"Object attribute types are not supported for '{key}' in this backend"
            )
        fields_value = value.get("fields")
        if not isinstance(fields_value, Mapping):
            raise ValueError(
                f"Attribute struct type for '{key}' must include mapping 'fields'"
            )

        fields: dict[str, AttributeType] = {}
        for field_name, field_type_json in fields_value.items():
            _validate_attribute_name(
                field_name,
                kind=f"Object attribute field for '{key}'",
            )
            fields[field_name] = _attribute_type_from_json_value(
                key=f"{key}.{field_name}",
                value=field_type_json,
                allow_vector_types=allow_vector_types,
                allow_struct_types=allow_struct_types,
            )
        return AttributeStructType(fields=fields)

    raise ValueError(
        f"Attribute type for '{key}' must be one of: str, int, float, bool, float_vector[N], or struct"
    )


def duckdb_sql_type_for_attribute_type(attribute_type: AttributeType) -> str:
    if attribute_type is str:
        return "VARCHAR"
    if attribute_type is int:
        return "INTEGER"
    if attribute_type is float:
        return "DOUBLE"
    if attribute_type is bool:
        return "BOOLEAN"
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"FLOAT[{attribute_type.dimension}]"
    if isinstance(attribute_type, AttributeStructType):
        fields_sql = ", ".join(
            f"{_quote_identifier(field_name)} {duckdb_sql_type_for_attribute_type(field_type)}"
            for field_name, field_type in attribute_type.fields.items()
        )
        return f"STRUCT({fields_sql})"
    raise ValueError(f"Unsupported attribute type: {attribute_type}")


def merge_attribute_values(
    *,
    attributes_spec: Mapping[str, AttributeSpec],
    sources: Iterable[Optional[Mapping[str, Any]]],
) -> dict[str, AttributeValue]:
    merged: dict[str, AttributeValue | object] = {
        key: _MISSING for key in attributes_spec
    }

    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            if key not in attributes_spec:
                raise ValueError(
                    f"Unknown attribute key '{key}'. Declare it in attributes when creating the store."
                )
            spec = attributes_spec[key]
            merged[key] = _normalize_attribute_value(
                key,
                value,
                spec.attribute_type,
                allow_none=spec.nullable,
            )

    result: dict[str, AttributeValue] = {}
    for key, spec in attributes_spec.items():
        value = merged[key]
        if value is _MISSING:
            if spec.required:
                raise ValueError(
                    f"Missing required attribute '{key}'. Provide a value in document or chunk attributes."
                )
            value = _copy_attribute_default(spec.default)
        result[key] = cast(AttributeValue, value)
    return result


def _copy_attribute_default(value: AttributeValue) -> AttributeValue:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: _copy_attribute_default(item) for key, item in value.items()}
    return value


def _normalize_attribute_value(
    key: str,
    value: Any,
    attribute_type: AttributeType,
    *,
    context: str = "attributes",
    allow_none: bool = True,
) -> AttributeValue:
    if value is None:
        if allow_none:
            return None
        raise ValueError(
            f"Invalid value for {context} '{key}': expected non-null value, got NoneType"
        )

    if isinstance(attribute_type, AttributeFloatVectorType):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {attribute_type.dimension}, got {type(value).__name__}"
            )
        if len(value) != attribute_type.dimension:
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {attribute_type.dimension}, got length {len(value)}"
            )
        normalized: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"Invalid value for {context} '{key}': expected list[float], got element of type {type(item).__name__}"
                )
            normalized.append(float(item))
        return normalized

    if isinstance(attribute_type, AttributeStructType):
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Invalid value for {context} '{key}': expected object mapping, got {type(value).__name__}"
            )
        extra_fields = set(value) - set(attribute_type.fields)
        if extra_fields:
            extra_display = ", ".join(sorted(extra_fields))
            raise ValueError(
                f"Invalid value for {context} '{key}': unknown object field(s): {extra_display}"
            )
        normalized_object: dict[str, AttributeValue] = {}
        for field_name, field_type in attribute_type.fields.items():
            if field_name not in value:
                raise ValueError(
                    f"Invalid value for {context} '{key}': missing object field '{field_name}'"
                )
            normalized_object[field_name] = _normalize_attribute_value(
                f"{key}.{field_name}",
                value[field_name],
                field_type,
                context=context,
                allow_none=False,
            )
        return normalized_object

    if attribute_type is str:
        ok = isinstance(value, str)
    elif attribute_type is bool:
        ok = isinstance(value, bool)
    elif attribute_type is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif attribute_type is float:
        ok = isinstance(value, float)
    else:
        ok = False

    if not ok:
        raise ValueError(
            f"Invalid value for {context} '{key}': expected {_attribute_type_label(attribute_type)}, got {type(value).__name__}"
        )
    return cast(AttributeValue, value)


def coerce_attribute_value_for_output(
    key: str,
    value: Any,
    attribute_type: AttributeType,
) -> AttributeValue:
    return _normalize_attribute_value(
        key,
        value,
        attribute_type,
        context="retrieved attributes",
        allow_none=True,
    )


def attribute_type_supports_filters(attribute_type: AttributeType) -> bool:
    return not isinstance(attribute_type, AttributeFloatVectorType)


def filterable_attribute_paths(
    attributes_schema: Mapping[str, AttributeType],
) -> set[str]:
    out: set[str] = set()
    for key, attribute_type in attributes_schema.items():
        out |= _filterable_attribute_paths_for_type(key, attribute_type)
    return out


def _filterable_attribute_paths_for_type(
    prefix: str,
    attribute_type: AttributeType,
) -> set[str]:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return set()
    if isinstance(attribute_type, AttributeStructType):
        out: set[str] = set()
        for key, nested in attribute_type.fields.items():
            out |= _filterable_attribute_paths_for_type(f"{prefix}.{key}", nested)
        return out
    return {prefix}


def _attribute_type_label(attribute_type: AttributeType) -> str:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"list[float] (length {attribute_type.dimension})"
    if isinstance(attribute_type, AttributeStructType):
        return "object"
    return attribute_type.__name__


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_attribute_name(name: Any, *, kind: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} names must be non-empty strings")
    if _ATTRIBUTE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"{kind} '{name}' must match [A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores only; no dots or dashes)"
        )
    if name.upper() in _ATTRIBUTE_FILTER_RESERVED_KEYWORDS:
        raise ValueError(
            f"{kind} '{name}' is reserved in string attributes_filter expressions"
        )
