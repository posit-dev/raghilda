from typing import Annotated

import pytest

from raghilda._attributes import (
    AttributeSpec,
    AttributeFloatVectorType,
    AttributeStructType,
    attributes_schema_from_json_dict,
    attributes_schema_to_json_dict,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    normalize_attributes_schema,
    normalize_attributes_spec,
)


def test_normalize_attributes_schema_accepts_class_annotations():
    class AttributesSpec:
        tenant: str
        priority: int
        is_public: bool

    schema = normalize_attributes_schema(
        AttributesSpec,
        reserved_columns=set(),
    )
    assert schema == {
        "tenant": str,
        "priority": int,
        "is_public": bool,
    }


def test_attributes_schema_roundtrip_with_vector_annotation():
    schema = normalize_attributes_schema(
        {"embedding25": Annotated[list[float], 25]},
        reserved_columns=set(),
        allow_vector_types=True,
    )
    assert schema["embedding25"] == AttributeFloatVectorType(dimension=25)

    encoded = attributes_schema_to_json_dict(schema)
    assert encoded == {"embedding25": "float_vector[25]"}

    decoded = attributes_schema_from_json_dict(encoded, allow_vector_types=True)
    assert decoded == schema


def test_normalize_attributes_spec_supports_inline_defaults_and_optional_union():
    spec = normalize_attributes_spec(
        {
            "tenant": str,
            "priority": (int, 0),
            "is_public": (bool, False),
            "topic": (str | None, None),
        },
        reserved_columns=set(),
    )
    assert spec == {
        "tenant": AttributeSpec(
            attribute_type=str,
            nullable=False,
            required=True,
        ),
        "priority": AttributeSpec(
            attribute_type=int,
            nullable=False,
            required=False,
            default=0,
        ),
        "is_public": AttributeSpec(
            attribute_type=bool,
            nullable=False,
            required=False,
            default=False,
        ),
        "topic": AttributeSpec(
            attribute_type=str,
            nullable=True,
            required=False,
            default=None,
        ),
    }


def test_attributes_spec_roundtrip_with_vector_annotation():
    spec = normalize_attributes_spec(
        {
            "tenant": str,
            "embedding25": Annotated[list[float], 25],
        },
        reserved_columns=set(),
        allow_vector_types=True,
    )
    encoded = attributes_spec_to_json_dict(spec)
    decoded = attributes_spec_from_json_dict(encoded, allow_vector_types=True)
    assert decoded == spec


def test_attributes_schema_roundtrip_with_nested_object_annotation():
    schema = normalize_attributes_schema(
        {
            "tenant": str,
            "details": {
                "source": str,
                "flags": {"is_public": bool},
            },
        },
        reserved_columns=set(),
        allow_struct_types=True,
    )
    assert schema["tenant"] is str
    assert schema["details"] == AttributeStructType(
        fields={
            "source": str,
            "flags": AttributeStructType(fields={"is_public": bool}),
        }
    )

    encoded = attributes_schema_to_json_dict(schema)
    decoded = attributes_schema_from_json_dict(
        encoded,
        allow_vector_types=True,
        allow_struct_types=True,
    )
    assert decoded == schema


def test_normalize_attributes_schema_rejects_vector_for_backends_without_support():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        normalize_attributes_schema(
            {"embedding25": Annotated[list[float], 25]},
            reserved_columns=set(),
            allow_vector_types=False,
        )


def test_normalize_attributes_spec_rejects_optional_values_when_not_supported():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'topic'"
    ):
        normalize_attributes_spec(
            {"tenant": str, "topic": str | None},
            reserved_columns=set(),
            allow_optional_values=False,
        )


def test_normalize_attributes_spec_rejects_dotted_top_level_attribute_names():
    with pytest.raises(ValueError, match="must match"):
        normalize_attributes_spec(
            {"a.b": str},
            reserved_columns=set(),
        )


def test_normalize_attributes_spec_rejects_hyphenated_top_level_attribute_names():
    with pytest.raises(ValueError, match="must match"):
        normalize_attributes_spec(
            {"a-b": str},
            reserved_columns=set(),
        )


def test_normalize_attributes_spec_rejects_dotted_nested_field_names():
    with pytest.raises(ValueError, match="must match"):
        normalize_attributes_spec(
            {"details": {"source.type": str}},
            reserved_columns=set(),
        )


def test_normalize_attributes_spec_rejects_hyphenated_nested_field_names():
    with pytest.raises(ValueError, match="must match"):
        normalize_attributes_spec(
            {"details": {"source-type": str}},
            reserved_columns=set(),
        )


def test_normalize_attributes_spec_rejects_reserved_keyword_top_level_name():
    with pytest.raises(ValueError, match="reserved"):
        normalize_attributes_spec(
            {"or": str},
            reserved_columns=set(),
        )


def test_normalize_attributes_spec_rejects_reserved_keyword_nested_field_name():
    with pytest.raises(ValueError, match="reserved"):
        normalize_attributes_spec(
            {"details": {"null": str}},
            reserved_columns=set(),
        )


def test_attributes_spec_from_json_rejects_optional_values_when_not_supported():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'topic'"
    ):
        attributes_spec_from_json_dict(
            {
                "topic": {
                    "type": "str",
                    "nullable": True,
                    "required": False,
                    "default": None,
                }
            },
            allow_optional_values=False,
        )


def test_attributes_schema_from_json_rejects_invalid_nested_field_name():
    with pytest.raises(ValueError, match="must match"):
        attributes_schema_from_json_dict(
            {
                "details": {
                    "type": "struct",
                    "fields": {"source-type": "str"},
                }
            }
        )
