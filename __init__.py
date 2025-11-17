"""
Quality assessment modules for evidence-based medicine.

Provides three standardized quality assessment methodologies:
- GRADE: Overall quality of evidence assessment
- Cochrane RoB 2.0: Risk of bias for RCTs
- ROBINS-I: Risk of bias for non-randomized studies
"""

from quality_assessor.grade_assessor import GRADEAssessor
from quality_assessor.cochrane_rob_assessor import CochraneRoBAssessor
from quality_assessor.robins_i_assessor import ROBINSIAssessor

__version__ = "1.0.0"

__all__ = [
    "GRADEAssessor",
    "CochraneRoBAssessor",
    "ROBINSIAssessor",
]
