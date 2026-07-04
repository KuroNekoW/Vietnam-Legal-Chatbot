import re

ARTICLE_PATTERN = re.compile(
    r"(?=^\s*Điều\s+\d+[A-Za-z]?\.)",
    flags=re.MULTILINE | re.IGNORECASE,
)

CLAUSE_PATTERN = re.compile(
    r"(?=^\s*\d+\.)",
    flags=re.MULTILINE,
)

POINT_PATTERN = re.compile(
    r"(?=^\s*[a-zđ]\))",
    flags=re.MULTILINE | re.IGNORECASE,
)