"""
GRADE (Grading of Recommendations Assessment, Development and Evaluation) assessor.

Implements GRADE methodology for assessing quality of evidence.
This is a standalone, framework-agnostic implementation.
"""

import logging
from typing import Any, Optional, Dict, Type

logger = logging.getLogger(__name__)


class GRADEAssessor:
    """
    Implements GRADE assessment methodology.

    This is a generic assessor that works with dependency injection.
    All dependencies (models, prompts, LLM provider, exceptions) are passed in.
    """

    def __init__(
        self,
        llm_provider: Any,
        prompt_template: str,
        models: Dict[str, Type],
        exception_class: Type[Exception] = Exception
    ):
        """
        Initialize GRADE assessor.

        Args:
            llm_provider: LLM provider instance with complete_with_json method
            prompt_template: Prompt template string for GRADE assessment
            models: Dictionary mapping model names to model classes:
                - 'GRADELevel': Enum for grade levels
                - 'GRADEDomain': Enum for GRADE domains
                - 'GRADEDomainAssessment': Domain assessment model
                - 'GRADEAssessment': Overall assessment model
                - 'StudyDesign': Enum for study designs
            exception_class: Exception class to raise on errors (default: Exception)
        """
        self.llm = llm_provider
        self.prompt_template = prompt_template
        self.models = models
        self.exception_class = exception_class
        logger.info("GRADEAssessor initialized")

    def assess_study(
        self,
        paper: Any,
        characteristics: Any,
        hypothesis: Optional[Any] = None
    ) -> Any:
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
        # Issue #24 fix: Standardized input validation
        if not paper:
            raise ValueError("paper cannot be None")
        if not characteristics:
            raise ValueError("characteristics cannot be None")

        try:
            logger.info(f"Starting GRADE assessment for paper: {paper.paper_id}")

            # Get relevant sections
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

            # Determine starting level
            starting_level = self._determine_starting_level(
                characteristics.study_design
            )

            logger.debug(f"GRADE starting level: {starting_level.value}")

            # Format prompt
            prompt = self.prompt_template.format(
                study_design=characteristics.study_design,
                methods=methods[:4000],
                results=results[:3000]
            )

            # Get LLM assessment
            logger.debug("Calling LLM for GRADE assessment")
            response = self.llm.complete_with_json(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1
            )

            data = response["json_data"]

            # Parse domain assessments
            GRADEDomain = self.models['GRADEDomain']
            GRADEDomainAssessment = self.models['GRADEDomainAssessment']
            GRADELevel = self.models['GRADELevel']
            GRADEAssessment = self.models['GRADEAssessment']

            domain_assessments = []
            for domain_data in data["domains"]:
                domain_enum = GRADEDomain(domain_data["domain"])
                domain_assess = GRADEDomainAssessment(
                    domain=domain_enum,
                    rating=domain_data["rating"],
                    justification=domain_data["justification"],
                    confidence=float(domain_data["confidence"]),
                    key_evidence=domain_data.get("key_evidence", [])
                )
                domain_assessments.append(domain_assess)

            logger.debug(f"Assessed {len(domain_assessments)} GRADE domains")

            # Parse final grade
            final_grade_str = data["final_grade"]
            overall_certainty = GRADELevel(final_grade_str)

            # Calculate downgrades
            downgrades_by_domain = {}
            total_downgrades = 0
            for d in domain_assessments:
                if d.rating == "serious":
                    downgrades_by_domain[d.domain.value] = 1
                    total_downgrades += 1
                elif d.rating == "very_serious":
                    downgrades_by_domain[d.domain.value] = 2
                    total_downgrades += 2

            # Parse upgrades
            upgrades = data.get("upgrades", {})

            logger.debug(
                f"GRADE complete: {overall_certainty.value}, "
                f"downgrades={total_downgrades}, "
                f"upgrades={sum(upgrades.values())}"
            )

            # Create assessment
            assessment = GRADEAssessment(
                overall_certainty=overall_certainty,
                starting_level=GRADELevel(data["starting_level"]),
                domain_assessments=domain_assessments,
                total_downgrades=total_downgrades,
                downgrades_by_domain=downgrades_by_domain,
                upgrades=upgrades,
                summary=data["summary"],
                overall_confidence=float(data["overall_confidence"])
            )

            return assessment

        except KeyError as e:
            raise self.exception_class(
                f"Failed to parse GRADE assessment: missing key {e}"
            ) from e

        except ValueError as e:
            raise self.exception_class(
                f"Invalid GRADE assessment value: {e}"
            ) from e

        except Exception as e:
            logger.error(f"Error in GRADE assessment: {e}", exc_info=True)
            raise self.exception_class(
                f"GRADE assessment failed: {e}"
            ) from e

    def _determine_starting_level(self, study_design: Any) -> Any:
        """
        Determine starting GRADE level based on study design.

        Per GRADE methodology:
        - RCTs start at HIGH
        - Observational studies start at LOW
        - Systematic reviews/meta-analyses not assessed (would assess underlying studies)

        Args:
            study_design: Type of study design (enum)

        Returns:
            Starting GRADE level

        Raises:
            ValueError: If study design is unknown/unsupported
        """
        StudyDesign = self.models['StudyDesign']
        GRADELevel = self.models['GRADELevel']

        design_to_level = {
            StudyDesign.RCT: GRADELevel.HIGH,
            StudyDesign.COHORT: GRADELevel.LOW,
            StudyDesign.CASE_CONTROL: GRADELevel.LOW,
            StudyDesign.CROSS_SECTIONAL: GRADELevel.VERY_LOW,
            StudyDesign.CASE_SERIES: GRADELevel.VERY_LOW,
            # Note: SYSTEMATIC_REVIEW and META_ANALYSIS should assess underlying studies
            StudyDesign.SYSTEMATIC_REVIEW: GRADELevel.LOW,
            StudyDesign.META_ANALYSIS: GRADELevel.HIGH,  # Can start high if based on RCTs
            StudyDesign.OTHER: GRADELevel.VERY_LOW,
        }

        if study_design not in design_to_level:
            raise ValueError(
                f"Unknown or unsupported study design: {study_design}. "
                f"Supported designs: {list(design_to_level.keys())}"
            )

        level = design_to_level[study_design]
        logger.debug(f"Study design {study_design} → starting level {level}")
        return level
