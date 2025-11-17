"""
ROBINS-I (Risk Of Bias In Non-randomized Studies - of Interventions) assessor.

Implements ROBINS-I methodology for assessing risk of bias in non-randomized
intervention studies.
"""

import logging
from typing import Optional

from eqas.models.input import PaperInput
from eqas.models.study import StudyCharacteristics, StudyDesign
from eqas.models.assessments import (
    ROBINSIAssessment,
    ROBINSIDomainAssessment,
    ROBINSIDomain,
    ROBINSILevel
)
from eqas.llm.claude_provider import ClaudeProvider
from eqas.llm.prompts import MedicalPrompts
from eqas.exceptions import AssessmentError

logger = logging.getLogger(__name__)


class ROBINSIAssessor:
    """
    Implements ROBINS-I for non-randomized studies of interventions.

    ROBINS-I is applicable to:
    - Cohort studies (prospective and retrospective)
    - Case-control studies
    - Case series with comparison groups (e.g., outbreak investigations,
      studies comparing different exposure levels)
    - Controlled before-after studies

    NOT applicable to:
    - RCTs (use Cochrane RoB 2.0 instead)
    - Pure descriptive case series without comparisons
    - Cross-sectional studies (not intervention studies)

    Note: For case series, ROBINS-I works best when there are identifiable
    comparison groups (explicit or implicit) allowing evaluation of intervention
    or exposure effects.
    """

    # Study designs where ROBINS-I is applicable
    APPLICABLE_DESIGNS = [
        StudyDesign.COHORT,
        StudyDesign.CASE_CONTROL,
        StudyDesign.CASE_SERIES,  # When comparison groups exist
    ]

    def __init__(self, llm_provider: ClaudeProvider):
        """
        Initialize ROBINS-I assessor.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        logger.info("ROBINSIAssessor initialized")

    def assess_study(
        self,
        paper: PaperInput,
        characteristics: StudyCharacteristics
    ) -> ROBINSIAssessment:
        """
        Perform ROBINS-I assessment.

        Args:
            paper: Parsed paper
            characteristics: Extracted study characteristics

        Returns:
            Complete ROBINS-I assessment

        Raises:
            AssessmentError: If assessment fails or study design not applicable
            ValueError: If paper or characteristics are None
        """
        # Validate inputs
        if not paper:
            raise ValueError("paper cannot be None")
        if not characteristics:
            raise ValueError("characteristics cannot be None")

        # Check if study design is applicable
        if characteristics.study_design not in self.APPLICABLE_DESIGNS:
            raise AssessmentError(
                f"ROBINS-I is only applicable to non-randomized intervention studies "
                f"(cohort, case-control, case series with comparisons). "
                f"Study design: {characteristics.study_design}. "
                f"For RCTs, use Cochrane RoB 2.0."
            )

        try:
            logger.info(
                f"Starting ROBINS-I assessment for paper: {paper.paper_id}"
            )

            # Special note for case series
            if characteristics.study_design == StudyDesign.CASE_SERIES:
                logger.info(
                    "Note: ROBINS-I assessment of case series works best when "
                    "comparison groups exist (e.g., different exposure levels, "
                    "outbreak investigations). Ensure the study has identifiable "
                    "intervention/exposure comparisons for meaningful bias assessment."
                )

            # Extract relevant sections
            title = paper.title or paper.sections.get("title", "")
            methods = None
            results = None

            for key, value in paper.sections.items():
                if "method" in key.lower():
                    methods = value
                elif "result" in key.lower():
                    results = value

            if not methods:
                methods = ""
                logger.warning("No methods section found")

            if not results:
                results = ""
                logger.warning("No results section found")

            # Extract study characteristics for prompt
            population = characteristics.population or "not specified"
            intervention = characteristics.intervention_exposure or "not specified"
            comparator = characteristics.comparator or "not specified"
            outcome = characteristics.primary_outcome or "not specified"

            # Format prompt
            prompt = MedicalPrompts.ROBINS_I_ASSESSMENT.format(
                study_design=characteristics.study_design,
                population=population[:500],
                intervention=intervention[:500],
                comparator=comparator[:500],
                outcome=outcome[:500],
                title=title[:500],
                methods=methods[:5000],
                results=results[:3000]
            )

            # Get LLM assessment
            logger.debug("Calling LLM for ROBINS-I assessment")
            response = self.llm.complete_with_json(
                prompt=prompt,
                max_tokens=5000,
                temperature=0.1
            )

            data = response["json_data"]

            # Parse target trial description
            target_trial = data.get("target_trial", "")
            if not target_trial or len(target_trial) < 20:
                logger.warning("Target trial description missing or too short")
                target_trial = (
                    f"Hypothetical RCT comparing {intervention} vs {comparator} "
                    f"in {population} measuring {outcome}"
                )

            # Parse domain assessments
            domain_assessments = []
            for domain_data in data["domains"]:
                domain_enum = ROBINSIDomain(domain_data["domain"])
                level_enum = ROBINSILevel(domain_data["level"])

                domain_assess = ROBINSIDomainAssessment(
                    domain=domain_enum,
                    level=level_enum,
                    justification=domain_data["justification"],
                    confidence=float(domain_data["confidence"]),
                    key_evidence=domain_data.get("key_evidence", []),
                    signaling_questions=domain_data.get("signaling_questions", [])
                )
                domain_assessments.append(domain_assess)

            logger.debug(f"Assessed {len(domain_assessments)} ROBINS-I domains")

            # Parse overall bias
            overall_bias = ROBINSILevel(data["overall_bias"])

            # Verify overall bias follows ROBINS-I algorithm (worst domain)
            computed_overall = self._apply_robins_i_algorithm(domain_assessments)
            if computed_overall != overall_bias:
                logger.warning(
                    f"LLM overall bias ({overall_bias.value}) differs from "
                    f"computed ({computed_overall.value}). Using computed value."
                )
                overall_bias = computed_overall

            logger.debug(f"ROBINS-I complete: {overall_bias.value}")

            # Create assessment
            assessment = ROBINSIAssessment(
                paper_id=paper.paper_id,
                study_design=characteristics.study_design,
                target_trial_description=target_trial,
                domain_assessments=domain_assessments,
                overall_bias=overall_bias,
                summary=data["summary"],
                overall_confidence=float(data["overall_confidence"])
            )

            return assessment

        except KeyError as e:
            raise AssessmentError(
                f"Failed to parse ROBINS-I assessment: missing key {e}"
            ) from e

        except ValueError as e:
            raise AssessmentError(
                f"Invalid ROBINS-I assessment value: {e}"
            ) from e

        except Exception as e:
            logger.error(f"Error in ROBINS-I assessment: {e}", exc_info=True)
            raise AssessmentError(
                f"ROBINS-I assessment failed: {e}"
            ) from e

    def _apply_robins_i_algorithm(
        self,
        domain_assessments: list[ROBINSIDomainAssessment]
    ) -> ROBINSILevel:
        """
        Apply ROBINS-I algorithm to determine overall bias.

        ROBINS-I Rule: Overall risk = WORST domain risk
        - If any domain is CRITICAL → overall is CRITICAL
        - If any domain is SERIOUS (but none critical) → overall is SERIOUS
        - If any domain is MODERATE (but none serious/critical) → overall is MODERATE
        - If any domain is NO_INFORMATION → overall is NO_INFORMATION
        - Only if ALL domains are LOW → overall is LOW

        Args:
            domain_assessments: List of domain assessments

        Returns:
            Overall bias level
        """
        domain_levels = [d.level for d in domain_assessments]

        # Check for critical
        if ROBINSILevel.CRITICAL in domain_levels:
            logger.debug("Overall bias: CRITICAL (at least one critical domain)")
            return ROBINSILevel.CRITICAL

        # Check for serious
        if ROBINSILevel.SERIOUS in domain_levels:
            logger.debug("Overall bias: SERIOUS (at least one serious domain)")
            return ROBINSILevel.SERIOUS

        # Check for moderate
        if ROBINSILevel.MODERATE in domain_levels:
            logger.debug("Overall bias: MODERATE (at least one moderate domain)")
            return ROBINSILevel.MODERATE

        # Check for no information
        if ROBINSILevel.NO_INFORMATION in domain_levels:
            logger.debug(
                "Overall bias: NO_INFORMATION (at least one domain with no information)"
            )
            return ROBINSILevel.NO_INFORMATION

        # All low
        logger.debug("Overall bias: LOW (all domains low risk)")
        return ROBINSILevel.LOW
