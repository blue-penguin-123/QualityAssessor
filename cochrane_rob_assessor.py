"""
Cochrane Risk of Bias 2.0 assessor for RCTs.

Implements Cochrane RoB 2.0 methodology for assessing risk of bias.
"""

import logging

from eqas.models.input import PaperInput
from eqas.models.study import StudyCharacteristics, StudyDesign
from eqas.models.assessments import (
    CochraneRoBAssessment,
    RoBDomainAssessment,
    RoBDomain,
    RoBJudgment
)
from eqas.llm.claude_provider import ClaudeProvider
from eqas.llm.prompts import MedicalPrompts
from eqas.exceptions import AssessmentError

logger = logging.getLogger(__name__)


class CochraneRoBAssessor:
    """Implements Cochrane Risk of Bias 2.0 for RCTs."""

    def __init__(self, llm_provider: ClaudeProvider):
        """
        Initialize Cochrane RoB assessor.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        logger.info("CochraneRoBAssessor initialized")

    def assess_study(
        self,
        paper: PaperInput,
        characteristics: StudyCharacteristics
    ) -> CochraneRoBAssessment:
        """
        Perform Cochrane RoB 2.0 assessment.

        Currently supports RCTs only. Raises error for other study types.

        Args:
            paper: Parsed paper
            characteristics: Extracted study characteristics

        Returns:
            Complete Cochrane RoB assessment

        Raises:
            AssessmentError: If assessment fails or study is not RCT
        """
        # Issue #24 fix: Standardized input validation
        if not paper:
            raise ValueError("paper cannot be None")
        if not characteristics:
            raise ValueError("characteristics cannot be None")

        if characteristics.study_design != StudyDesign.RCT:
            raise AssessmentError(
                f"Cochrane RoB 2.0 is only applicable to RCTs. "
                f"Study design: {characteristics.study_design}"
            )

        try:
            logger.info(
                f"Starting Cochrane RoB assessment for paper: {paper.paper_id}"
            )

            # Get relevant sections
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

            # Format prompt
            prompt = MedicalPrompts.COCHRANE_ROB_RCT.format(
                title=title,
                methods=methods[:4000],
                results=results[:2000]
            )

            # Get LLM assessment
            logger.debug("Calling LLM for Cochrane RoB assessment")
            response = self.llm.complete_with_json(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1
            )

            data = response["json_data"]

            # Parse domain assessments
            domain_assessments = []
            for domain_data in data["domains"]:
                domain_enum = RoBDomain(domain_data["domain"])
                judgment_enum = RoBJudgment(domain_data["judgment"])

                domain_assess = RoBDomainAssessment(
                    domain=domain_enum,
                    judgment=judgment_enum,
                    justification=domain_data["justification"],
                    confidence=float(domain_data["confidence"]),
                    key_evidence=domain_data.get("key_evidence", [])
                )
                domain_assessments.append(domain_assess)

            logger.debug(f"Assessed {len(domain_assessments)} RoB domains")

            # Parse overall risk
            overall_risk = RoBJudgment(data["overall_risk"])

            # Verify overall risk follows algorithm
            computed_overall = self._apply_rob_algorithm(domain_assessments)
            if computed_overall != overall_risk:
                logger.warning(
                    f"LLM overall risk ({overall_risk.value}) differs from "
                    f"computed ({computed_overall.value}). Using computed value."
                )
                overall_risk = computed_overall

            logger.debug(f"RoB complete: {overall_risk.value}")

            # Create assessment
            assessment = CochraneRoBAssessment(
                overall_risk=overall_risk,
                domain_assessments=domain_assessments,
                summary=data["summary"],
                overall_confidence=float(data["overall_confidence"])
            )

            return assessment

        except KeyError as e:
            raise AssessmentError(
                f"Failed to parse RoB assessment: missing key {e}"
            ) from e

        except ValueError as e:
            raise AssessmentError(
                f"Invalid RoB assessment value: {e}"
            ) from e

        except Exception as e:
            logger.error(f"Error in RoB assessment: {e}", exc_info=True)
            raise AssessmentError(
                f"Cochrane RoB assessment failed: {e}"
            ) from e

    def _apply_rob_algorithm(
        self,
        domain_assessments: list[RoBDomainAssessment]
    ) -> RoBJudgment:
        """
        Apply Cochrane RoB 2.0 algorithm to determine overall risk.

        Improved rules per Cochrane RoB 2.0 Tool guidance:
        - Low risk: All domains low risk
        - Some concerns: At least one domain some concerns, no high risk
        - High risk:
            * At least one domain high risk, OR
            * Critical domains have concerns (randomization + deviations), OR
            * Multiple domains (≥3) with some concerns

        Note: Domains 1 (Randomization) and 2 (Deviations) are considered critical.
        Concerns in these domains carry more weight.

        Args:
            domain_assessments: List of domain assessments

        Returns:
            Overall risk judgment
        """
        domain_risks = [d.judgment for d in domain_assessments]

        # Create domain map for easier checking
        domain_map = {d.domain: d.judgment for d in domain_assessments}

        # Check for high risk in any domain
        if RoBJudgment.HIGH in domain_risks:
            logger.debug("Overall risk: HIGH (at least one high-risk domain)")
            return RoBJudgment.HIGH

        # Check for concerns in critical domains (randomization + deviations)
        critical_domains = ["randomization", "deviations_from_intended_interventions"]
        critical_concerns = sum(
            1 for domain in critical_domains
            if domain_map.get(domain) in [RoBJudgment.SOME_CONCERNS, RoBJudgment.HIGH]
        )

        if critical_concerns >= 2:
            logger.debug(
                "Overall risk: HIGH (concerns in multiple critical domains: "
                "randomization and deviations)"
            )
            return RoBJudgment.HIGH

        # Check for some concerns
        if RoBJudgment.SOME_CONCERNS in domain_risks:
            concern_count = domain_risks.count(RoBJudgment.SOME_CONCERNS)

            # Three or more "some concerns" escalates to high risk
            if concern_count >= 3:
                logger.debug(
                    f"Overall risk: HIGH ({concern_count} domains with some concerns)"
                )
                return RoBJudgment.HIGH

            logger.debug(
                f"Overall risk: SOME_CONCERNS ({concern_count} domain(s) with concerns)"
            )
            return RoBJudgment.SOME_CONCERNS

        # All low risk
        logger.debug("Overall risk: LOW (all domains low risk)")
        return RoBJudgment.LOW
