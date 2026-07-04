import re

ARTICLE_PATTERN = re.compile(
    r"^Điều\s+\d+[\.:]?\s.*",
    flags=re.MULTILINE,
)

CLAUSE_PATTERN = re.compile(
    r"(?=^\s*\d+\.)",
    flags=re.MULTILINE,
)

POINT_PATTERN = re.compile(
    r"(?=^\s*[a-zđ]\))",
    flags=re.MULTILINE | re.IGNORECASE,
)