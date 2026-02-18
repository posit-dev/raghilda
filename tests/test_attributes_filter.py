import pytest

from raghilda._attributes import (
    compile_filter_to_chroma_where,
    compile_filter_to_openai_filters,
    compile_filter_to_sql,
)


def test_compile_filter_to_sql_with_and_or():
    sql = compile_filter_to_sql(
        "tenant = 'docs' AND (priority >= 2 OR is_public = TRUE)",
        allowed_columns={"tenant", "priority", "is_public"},
    )
    assert sql == '("tenant" = \'docs\' AND ("priority" >= 2 OR "is_public" = TRUE))'


def test_compile_filter_to_sql_with_dot_path_identifier():
    sql = compile_filter_to_sql(
        "details.source = 'handbook' AND details.flags.is_public = TRUE",
        allowed_columns={"details.source", "details.flags.is_public"},
    )
    assert sql == (
        "(struct_extract(\"details\", 'source') = 'handbook' AND "
        "struct_extract(struct_extract(\"details\", 'flags'), 'is_public') = TRUE)"
    )


def test_compile_filter_to_sql_null_via_is_null():
    sql = compile_filter_to_sql(
        "tenant IS NULL",
        allowed_columns={"tenant"},
    )
    assert sql == '"tenant" IS NULL'


def test_compile_filter_to_sql_null_via_is_not_null():
    sql = compile_filter_to_sql(
        "tenant IS NOT NULL",
        allowed_columns={"tenant"},
    )
    assert sql == '"tenant" IS NOT NULL'


def test_compile_filter_to_sql_mapping_ast_null_uses_is_null():
    sql = compile_filter_to_sql(
        {"type": "eq", "key": "tenant", "value": None},
        allowed_columns={"tenant"},
    )
    assert sql == '"tenant" IS NULL'


def test_compile_filter_to_sql_rejects_null_with_inequality_operator():
    with pytest.raises(ValueError, match="NULL is only allowed with = and !="):
        compile_filter_to_sql(
            "priority > NULL",
            allowed_columns={"priority"},
        )


def test_compile_filter_to_sql_rejects_null_with_inequality_operator_mapping_ast():
    with pytest.raises(ValueError, match="NULL is only allowed with = and !="):
        compile_filter_to_sql(
            {"type": "gt", "key": "priority", "value": None},
            allowed_columns={"priority"},
        )


def test_compile_filter_to_sql_rejects_equals_null_syntax():
    with pytest.raises(ValueError, match="Use IS NULL or IS NOT NULL"):
        compile_filter_to_sql(
            "tenant = NULL",
            allowed_columns={"tenant"},
        )


def test_compile_filter_to_chroma_where():
    where = compile_filter_to_chroma_where(
        "tenant = 'docs' AND priority IN (1, 2, 3)",
        allowed_columns={"tenant", "priority"},
    )
    assert where == {
        "$and": [
            {"tenant": {"$eq": "docs"}},
            {"priority": {"$in": [1, 2, 3]}},
        ]
    }


def test_compile_filter_to_openai_filters():
    filters = compile_filter_to_openai_filters(
        "tenant = 'docs' AND score >= 0.75",
        allowed_columns={"tenant", "score"},
    )
    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "gte", "key": "score", "value": 0.75},
        ],
    }


def test_compile_filter_to_openai_filters_preserves_large_int_scalar():
    large_int = 9007199254740993
    filters = compile_filter_to_openai_filters(
        {"type": "eq", "key": "doc_id", "value": large_int},
        allowed_columns={"doc_id"},
    )
    assert filters is not None
    assert filters == {
        "type": "eq",
        "key": "doc_id",
        "value": large_int,
    }
    assert isinstance(filters["value"], int)


def test_compile_filter_to_openai_filters_preserves_large_int_list_items():
    large_int = 9007199254740993
    filters = compile_filter_to_openai_filters(
        {"type": "in", "key": "doc_id", "value": [9007199254740992, large_int]},
        allowed_columns={"doc_id"},
    )
    assert filters is not None
    assert filters == {
        "type": "in",
        "key": "doc_id",
        "value": [9007199254740992, large_int],
    }
    assert all(isinstance(value, int) for value in filters["value"])


def test_compile_filter_to_chroma_where_rejects_null():
    with pytest.raises(ValueError, match="NULL is not supported in Chroma"):
        compile_filter_to_chroma_where(
            "tenant IS NULL",
            allowed_columns={"tenant"},
        )


def test_compile_filter_to_openai_filters_rejects_null():
    with pytest.raises(ValueError, match="NULL is not supported in OpenAI"):
        compile_filter_to_openai_filters(
            "tenant IS NULL",
            allowed_columns={"tenant"},
        )


def test_compile_filter_rejects_unknown_column():
    with pytest.raises(ValueError, match="Unknown attribute column 'unknown'"):
        compile_filter_to_sql(
            "unknown = 'x'",
            allowed_columns={"tenant"},
        )


def test_compile_filter_to_sql_from_mapping_ast_with_in():
    sql = compile_filter_to_sql(
        {
            "type": "and",
            "filters": [
                {"type": "in", "key": "tenant", "value": ["docs", "blog"]},
                {"type": "gte", "key": "priority", "value": 2},
            ],
        },
        allowed_columns={"tenant", "priority"},
    )
    assert sql == "(\"tenant\" IN ('docs', 'blog') AND \"priority\" >= 2)"


def test_compile_filter_to_chroma_where_from_mapping_ast():
    where = compile_filter_to_chroma_where(
        {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "in", "key": "priority", "value": [1, 2, 3]},
            ],
        },
        allowed_columns={"tenant", "priority"},
    )
    assert where == {
        "$and": [
            {"tenant": {"$eq": "docs"}},
            {"priority": {"$in": [1, 2, 3]}},
        ]
    }


def test_compile_filter_to_openai_filters_from_mapping_ast():
    filters = compile_filter_to_openai_filters(
        {
            "type": "or",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "gt", "key": "score", "value": 0.9},
            ],
        },
        allowed_columns={"tenant", "score"},
    )
    assert filters == {
        "type": "or",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "gt", "key": "score", "value": 0.9},
        ],
    }
