"""
severity/utils/grade_schema.py

severity 标签的唯一权威定义。
所有模块必须从这里 import，禁止在其他地方重新定义。

顺序不可更改：
    Acceptable  = 0
    Marginal NG = 1
    NG          = 2
    Gross NG    = 3
"""

GRADE_NAMES: list = ["Acceptable", "Marginal NG", "NG", "Gross NG"]
GRADE_MAP: dict = {name: idx for idx, name in enumerate(GRADE_NAMES)}
NUM_GRADES: int = len(GRADE_NAMES)
