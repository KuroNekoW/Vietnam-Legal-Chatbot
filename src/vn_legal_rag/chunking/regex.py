import re

#
# Điều
#

ARTICLE_PATTERN = re.compile(
    r"(?m)^Điều\s+\d+[A-Za-z]?(?:\.\s*.*)?"
)

#
# Khoản
#

CLAUSE_PATTERN = re.compile(
    r"(?m)^\d+\.\s"
)

#
# Điểm
#

POINT_PATTERN = re.compile(
    r"(?m)^[a-zđ]\)\s"
)

#
# Bộ câu hỏi
#

QUESTION_PATTERN = re.compile(
    r"(?im)^Câu\s+\d+\."
)