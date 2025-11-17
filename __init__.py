"""
Quality assessment modules.
"""

# Import from the installed shared package
from quality_assessor import (
    GRADEAssessor as UnifiedGRADEAssessor,
    CochraneRoBAssessor as UnifiedCochraneRoBAssessor,
    ROBINSIAssessor as UnifiedROBINSIAssessor
)

# Import eqas-specific dependencies
from eqas.models.study import StudyCharacteristics, StudyDesign
from eqas.models.assessments import (
    GRADEAssessment,
    GRADEDomainAssessment,
    GRADELevel,
    GRADEDomain,
    CochraneRoBAssessment,
    RoBDomainAssessment,
    RoBDomain,
    RoBJudgment,
    ROBINSIAssessment,
    ROBINSIDomainAssessment,
    ROBINSIDomain,
    ROBINSILevel
)
from eqas.llm.claude_provider import ClaudeProvider
from eqas.llm.prompts import MedicalPrompts
from eqas.exceptions import AssessmentError


class GRADEAssessor:
    """
    GRADE assessor configured for eqas package.

    Wraps the unified assessor with eqas-specific configuration.
    """

    def __init__(self, llm_provider: ClaudeProvider):
        """
        Initialize GRADE assessor.

        Args:
            llm_provider: ClaudeProvider instance
        """
        self._assessor = UnifiedGRADEAssessor(
            llm_provider=llm_provider,
            prompt_template=MedicalPrompts.GRADE_ASSESSMENT,
            models={
                'GRADELevel': GRADELevel,
                'GRADEDomain': GRADEDomain,
                'GRADEDomainAssessment': GRADEDomainAssessment,
                'GRADEAssessment': GRADEAssessment,
                'StudyDesign': StudyDesign
            },
            exception_class=AssessmentError
        )

    def assess_study(self, paper, characteristics, hypothesis=None):
        """
        Perform GRADE assessment.

        Args:
            paper: Parsed paper
            characteristics: Extracted study characteristics
            hypothesis: Optional hypothesis for indirectness assessment

        Returns:
            Complete GRADE assessment

        Raises:
            AssessmentError: If assessment fails
        """
        return self._assessor.assess_study(
            paper=paper,
            characteristics=characteristics,
            hypothesis=hypothesis
        )


class CochraneRoBAssessor:
    """
    Cochrane RoB 2.0 assessor configured for eqas package.

    Wraps the unified assessor with eqas-specific configuration.
    """

    def __init__(self, llm_provider: ClaudeProvider):
        """
        Initialize Cochrane RoB assessor.

        Args:
            llm_provider: ClaudeProvider instance
        """
        self._assessor = UnifiedCochraneRoBAssessor(
            llm_provider=llm_provider,
            prompt_template=MedicalPrompts.COCHRANE_ROB_RCT,
            models={
                'RoBDomain': RoBDomain,
                'RoBJudgment': RoBJudgment,
                'RoBDomainAssessment': RoBDomainAssessment,
                'CochraneRoBAssessment': CochraneRoBAssessment,
                'StudyDesign': StudyDesign
            },
            exception_class=AssessmentError
        )

    def assess_study(self, paper, characteristics):
        """
        Perform Cochrane RoB 2.0 assessment.

        Args:
            paper: Parsed paper
            characteristics: Extracted study characteristics

        Returns:
            Complete Cochrane RoB assessment

        Raises:
            AssessmentError: If assessment fails or study is not RCT
        """
        return self._assessor.assess_study(
            paper=paper,
            characteristics=characteristics
        )


class ROBINSIAssessor:
    """
    ROBINS-I assessor configured for eqas package.

    Wraps the unified assessor with eqas-specific configuration.
    """

    def __init__(self, llm_provider: ClaudeProvider):
        """
        Initialize ROBINS-I assessor.

        Args:
            llm_provider: ClaudeProvider instance
        """
        self._assessor = UnifiedROBINSIAssessor(
            llm_provider=llm_provider,
            prompt_template=MedicalPrompts.ROBINS_I_ASSESSMENT,
            models={
                'ROBINSIDomain': ROBINSIDomain,
                'ROBINSILevel': ROBINSILevel,
                'ROBINSIDomainAssessment': ROBINSIDomainAssessment,
                'ROBINSIAssessment': ROBINSIAssessment,
                'StudyDesign': StudyDesign
            },
            exception_class=AssessmentError,
            applicable_designs=[
                StudyDesign.COHORT,
                StudyDesign.CASE_CONTROL,
                StudyDesign.CASE_SERIES
            ]
        )

    def assess_study(self, paper, characteristics):
        """
        Perform ROBINS-I assessment.

        Args:
            paper: Parsed paper
            characteristics: Extracted study characteristics

        Returns:
            Complete ROBINS-I assessment

        Raises:
            AssessmentError: If assessment fails or study design not applicable
        """
        return self._assessor.assess_study(
            paper=paper,
            characteristics=characteristics
        )


__all__ = ["GRADEAssessor", "CochraneRoBAssessor", "ROBINSIAssessor"]
