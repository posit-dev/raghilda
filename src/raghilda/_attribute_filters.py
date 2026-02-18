from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional, cast

from ._attribute_schema import AttributeFilter, AttributeFilterValue, AttributeScalar


@dataclass(frozen=True)
class FilterComparison:
    column: str
    operator: str
    value: AttributeFilterValue | list[AttributeScalar]


@dataclass(frozen=True)
class FilterLogical:
    operator: str
    children: list["FilterNode"]


FilterNode = FilterComparison | FilterLogical


def compile_filter_to_sql(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[str]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_sql(node)


def compile_filter_to_chroma_where(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_chroma_where(node)


def compile_filter_to_openai_filters(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_openai_filters(node)


def _parse_filter_or_none(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]],
) -> Optional[FilterNode]:
    allowed_column_set = None if allowed_columns is None else set(allowed_columns)
    if attributes_filter is None:
        return None
    if isinstance(attributes_filter, str):
        text = attributes_filter.strip()
        if not text:
            return None
        parser = _FilterParser(text, allowed_columns=allowed_column_set)
        return parser.parse()
    if isinstance(attributes_filter, Mapping):
        return _parse_filter_mapping_node(
            attributes_filter,
            allowed_columns=allowed_column_set,
        )
    raise TypeError(
        f"attributes_filter must be a string or mapping, got {type(attributes_filter).__name__}"
    )


def _parse_filter_mapping_node(
    node: Mapping[str, Any], *, allowed_columns: Optional[set[str]]
) -> FilterNode:
    node_type = node.get("type")
    if not isinstance(node_type, str) or not node_type:
        raise ValueError("Filter mapping nodes must include a non-empty string 'type'")
    node_type = node_type.lower()

    if node_type in {"and", "or"}:
        filters = node.get("filters")
        if not isinstance(filters, list) or len(filters) == 0:
            raise ValueError(
                f"Logical filter '{node_type}' must include a non-empty list in 'filters'"
            )
        children: list[FilterNode] = []
        for child in filters:
            if not isinstance(child, Mapping):
                raise ValueError("Each item in 'filters' must be a mapping")
            children.append(
                _parse_filter_mapping_node(child, allowed_columns=allowed_columns)
            )
        return FilterLogical(operator=node_type, children=children)

    if node_type in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"}:
        column = node.get("key")
        if not isinstance(column, str) or not column:
            raise ValueError(
                f"Comparison filter '{node_type}' must include string 'key'"
            )
        _validate_allowed_filter_column(column, allowed_columns)

        if "value" not in node:
            raise ValueError(f"Comparison filter '{node_type}' must include 'value'")
        raw_value = node["value"]

        if node_type in {"in", "nin"}:
            value_list = _parse_filter_list_value(raw_value)
            return FilterComparison(column=column, operator=node_type, value=value_list)

        value = _parse_filter_scalar_value(raw_value, allow_null=True)
        _validate_null_comparison_operator(operator=node_type, value=value)
        return FilterComparison(column=column, operator=node_type, value=value)

    raise ValueError(f"Unknown filter node type '{node_type}'")


def _parse_filter_scalar_value(value: Any, *, allow_null: bool) -> AttributeFilterValue:
    if value is None:
        if allow_null:
            return None
        raise ValueError("NULL is not allowed in this filter value")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported filter value type: {type(value).__name__}")


def _parse_filter_list_value(value: Any) -> list[AttributeScalar]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("IN/NIN filter values must be a non-empty list")
    parsed: list[AttributeScalar] = []
    for item in value:
        item_value = _parse_filter_scalar_value(item, allow_null=False)
        assert item_value is not None
        parsed.append(item_value)
    return parsed


def _validate_null_comparison_operator(
    *, operator: str, value: AttributeFilterValue
) -> None:
    if value is None and operator not in {"eq", "ne"}:
        raise ValueError("NULL is only allowed with = and != operators")


def _validate_allowed_filter_column(
    column: str, allowed_columns: Optional[set[str]]
) -> None:
    if allowed_columns is None:
        return
    if column not in allowed_columns:
        allowed = ", ".join(sorted(allowed_columns))
        raise ValueError(
            f"Unknown attribute column '{column}' in filter. Allowed columns: {allowed}"
        )


class _FilterParser:
    def __init__(self, text: str, *, allowed_columns: Optional[set[str]]):
        self._tokens = _tokenize_filter(text)
        self._idx = 0
        self._allowed_columns = allowed_columns

    def parse(self) -> FilterNode:
        node = self._parse_or()
        self._expect("EOF")
        return node

    def _parse_or(self) -> FilterNode:
        left = self._parse_and()
        children = [left]
        while self._match("OR"):
            children.append(self._parse_and())
        if len(children) == 1:
            return children[0]
        return FilterLogical(operator="or", children=children)

    def _parse_and(self) -> FilterNode:
        left = self._parse_primary()
        children = [left]
        while self._match("AND"):
            children.append(self._parse_primary())
        if len(children) == 1:
            return children[0]
        return FilterLogical(operator="and", children=children)

    def _parse_primary(self) -> FilterNode:
        if self._match("LPAREN"):
            node = self._parse_or()
            self._expect("RPAREN")
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> FilterNode:
        column = self._parse_identifier_path()
        _validate_allowed_filter_column(column, self._allowed_columns)

        if self._match("IS"):
            if self._match("NOT"):
                self._expect("NULL")
                return FilterComparison(column=column, operator="ne", value=None)
            self._expect("NULL")
            return FilterComparison(column=column, operator="eq", value=None)

        if self._match("NOT"):
            self._expect("IN")
            operator = "nin"
            value = self._parse_literal_list()
            return FilterComparison(column=column, operator=operator, value=value)

        if self._match("IN"):
            operator = "in"
            value = self._parse_literal_list()
            return FilterComparison(column=column, operator=operator, value=value)

        token = self._expect("OP")
        op_map = {
            "=": "eq",
            "!=": "ne",
            ">": "gt",
            ">=": "gte",
            "<": "lt",
            "<=": "lte",
        }
        operator = op_map[cast(str, token.value)]
        value = self._parse_literal_value()
        _validate_null_comparison_operator(operator=operator, value=value)
        if value is None and operator in {"eq", "ne"}:
            raise ValueError("Use IS NULL or IS NOT NULL for NULL comparisons")
        return FilterComparison(column=column, operator=operator, value=value)

    def _parse_identifier_path(self) -> str:
        token = self._expect("IDENT")
        assert isinstance(token.value, str)
        parts = [token.value]
        while self._match("DOT"):
            next_token = self._expect("IDENT")
            assert isinstance(next_token.value, str)
            parts.append(next_token.value)
        return ".".join(parts)

    def _parse_literal_list(self) -> list[AttributeScalar]:
        self._expect("LPAREN")
        values: list[AttributeScalar] = []
        values.append(self._parse_list_literal_value())
        while self._match("COMMA"):
            values.append(self._parse_list_literal_value())
        self._expect("RPAREN")
        return values

    def _parse_list_literal_value(self) -> AttributeScalar:
        value = self._parse_literal_value()
        if value is None:
            raise ValueError("NULL is not allowed inside IN (...) lists")
        return value

    def _parse_literal_value(self) -> AttributeFilterValue:
        token = self._peek()
        if token.kind == "STRING":
            self._idx += 1
            assert isinstance(token.value, str)
            return token.value
        if token.kind == "NUMBER":
            self._idx += 1
            assert isinstance(token.value, (int, float))
            return token.value
        if token.kind == "TRUE":
            self._idx += 1
            return True
        if token.kind == "FALSE":
            self._idx += 1
            return False
        if token.kind == "NULL":
            self._idx += 1
            return None
        raise ValueError(f"Expected a literal value, got '{token.raw}'")

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self._idx += 1
            return True
        return False

    def _expect(self, kind: str) -> "_Token":
        token = self._peek()
        if token.kind != kind:
            raise ValueError(f"Expected {kind}, got '{token.raw}'")
        self._idx += 1
        return token

    def _peek(self) -> "_Token":
        return self._tokens[self._idx]


@dataclass(frozen=True)
class _Token:
    kind: str
    raw: str
    value: Any = None


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _tokenize_filter(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]

        if char.isspace():
            i += 1
            continue

        if (
            text.startswith(">=", i)
            or text.startswith("<=", i)
            or text.startswith("!=", i)
        ):
            op = text[i : i + 2]
            tokens.append(_Token(kind="OP", raw=op, value=op))
            i += 2
            continue

        if char in ("=", ">", "<"):
            tokens.append(_Token(kind="OP", raw=char, value=char))
            i += 1
            continue

        if char == "(":
            tokens.append(_Token(kind="LPAREN", raw=char))
            i += 1
            continue
        if char == ")":
            tokens.append(_Token(kind="RPAREN", raw=char))
            i += 1
            continue
        if char == ",":
            tokens.append(_Token(kind="COMMA", raw=char))
            i += 1
            continue
        if char == ".":
            tokens.append(_Token(kind="DOT", raw=char))
            i += 1
            continue

        if char == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n or text[j] != "'":
                raise ValueError("Unterminated string literal in attributes filter")
            raw = text[i : j + 1]
            value = raw[1:-1].replace("''", "'")
            tokens.append(_Token(kind="STRING", raw=raw, value=value))
            i = j + 1
            continue

        number_match = _NUMBER_RE.match(text, i)
        if number_match is not None:
            raw = number_match.group(0)
            value: int | float
            if "." in raw:
                value = float(raw)
            else:
                value = int(raw)
            tokens.append(_Token(kind="NUMBER", raw=raw, value=value))
            i = number_match.end()
            continue

        ident_match = _IDENT_RE.match(text, i)
        if ident_match is not None:
            raw = ident_match.group(0)
            keyword = raw.upper()
            if keyword in {"AND", "OR", "IN", "IS", "NOT", "TRUE", "FALSE", "NULL"}:
                tokens.append(_Token(kind=keyword, raw=raw))
            else:
                tokens.append(_Token(kind="IDENT", raw=raw, value=raw))
            i = ident_match.end()
            continue

        raise ValueError(f"Unexpected character '{char}' in attributes filter")

    tokens.append(_Token(kind="EOF", raw=""))
    return tokens


def _emit_sql(node: FilterNode) -> str:
    if isinstance(node, FilterLogical):
        joiner = " AND " if node.operator == "and" else " OR "
        return "(" + joiner.join(_emit_sql(child) for child in node.children) + ")"

    column = _sql_column_expression(node.column)
    if node.operator == "in":
        values = cast(list[AttributeScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} IN ({values_sql})"
    if node.operator == "nin":
        values = cast(list[AttributeScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} NOT IN ({values_sql})"

    value = cast(AttributeFilterValue, node.value)
    if value is None and node.operator == "eq":
        return f"{column} IS NULL"
    if value is None and node.operator == "ne":
        return f"{column} IS NOT NULL"

    op_map = {
        "eq": "=",
        "ne": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }
    assert value is not None
    return f"{column} {op_map[node.operator]} {_sql_literal_scalar(value)}"


def _emit_chroma_where(node: FilterNode) -> dict[str, Any]:
    if isinstance(node, FilterLogical):
        key = "$and" if node.operator == "and" else "$or"
        return {key: [_emit_chroma_where(child) for child in node.children]}

    if node.value is None:
        raise ValueError("NULL is not supported in Chroma attributes filters")

    op_map = {
        "eq": "$eq",
        "ne": "$ne",
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "in": "$in",
        "nin": "$nin",
    }
    return {node.column: {op_map[node.operator]: node.value}}


def _emit_openai_filters(node: FilterNode) -> dict[str, Any]:
    if isinstance(node, FilterLogical):
        return {
            "type": node.operator,
            "filters": [_emit_openai_filters(child) for child in node.children],
        }

    if node.value is None:
        raise ValueError("NULL is not supported in OpenAI attributes filters")

    value = node.value
    if isinstance(value, list):
        for item in value:
            # bool is a subclass of int, but OpenAI IN/NIN does not support booleans.
            if isinstance(item, bool):
                raise ValueError(
                    "Boolean values are not supported in IN/NIN for OpenAI filters"
                )
            if not isinstance(item, (str, int, float)):
                raise ValueError(
                    f"Unsupported filter value type: {type(item).__name__}"
                )
        value = cast(list[str | int | float], value)
    elif not isinstance(value, (bool, str, int, float)):
        raise ValueError(f"Unsupported filter value type: {type(value).__name__}")

    return {
        "type": node.operator,
        "key": node.column,
        "value": value,
    }


def _sql_literal_scalar(value: AttributeScalar) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_column_expression(column: str) -> str:
    parts = column.split(".")
    if len(parts) == 1:
        return _quote_identifier(parts[0])
    expr = _quote_identifier(parts[0])
    for field in parts[1:]:
        escaped = field.replace("'", "''")
        expr = f"struct_extract({expr}, '{escaped}')"
    return expr
