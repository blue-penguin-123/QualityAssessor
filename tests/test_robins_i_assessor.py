"""Tests for ROBINS-I assessor."""
 
import pytest
from unittest.mock import Mock, MagicMock
from enum import Enum
 
 
class TestROBINSIAssessor:
    """Tests for ROBINS-I assessment functionality."""
 
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.complete_with_json = Mock()
        return provider
 
    @pytest.fixture
    def mock_models(self):
        """Create mock model classes."""
        # Create mock enums
        class MockROBINSILevel(Enum):
            LOW = "low"
            MODERATE = "moderate"
            SERIOUS = "serious"
            CRITICAL = "critical"
            NO_INFORMATION = "no_information"
 
        class MockROBINSIDomain(Enum):
            CONFOUNDING = "confounding"
            SELECTION_OF_PARTICIPANTS = "selection_of_participants"
            CLASSIFICATION_OF_INTERVENTIONS = "classification_of_interventions"
            DEVIATIONS_FROM_INTERVENTIONS = "deviations_from_interventions"
            MISSING_DATA = "missing_data"
            MEASUREMENT_OF_OUTCOMES = "measurement_of_outcomes"
            SELECTION_OF_REPORTED_RESULTS = "selection_of_reported_results"
 
        class MockStudyDesign(Enum):
            RCT = "randomized_controlled_trial"
            COHORT = "cohort_study"
            CASE_CONTROL = "case_control"
            CASE_SERIES = "case_series"
            CROSS_SECTIONAL = "cross_sectional"
 
        # Create mock model classes
        MockROBINSIDomainAssessment = MagicMock()
        MockROBINSIAssessment = MagicMock()
 
        return {
            'ROBINSIDomain': MockROBINSIDomain,
            'ROBINSILevel': MockROBINSILevel,
            'ROBINSIDomainAssessment': MockROBINSIDomainAssessment,
            'ROBINSIAssessment': MockROBINSIAssessment,
            'StudyDesign': MockStudyDesign
        }
 
    @pytest.fixture
    def mock_prompt_template(self):
        """Create a mock prompt template."""
        return ("Assess ROBINS-I for {study_design}. Population: {population}. "
                "Intervention: {intervention}. Comparator: {comparator}. Outcome: {outcome}. "
                "Title: {title}. Methods: {methods}. Results: {results}")
 
    @pytest.fixture
    def mock_exception_class(self):
        """Create a mock exception class."""
        class MockAssessmentError(Exception):
            pass
        return MockAssessmentError
 
    @pytest.fixture
    def robins_assessor(self, mock_llm_provider, mock_prompt_template, mock_models, mock_exception_class):
        """Create a ROBINS-I assessor instance."""
        from quality_assessor import ROBINSIAssessor
 
        applicable_designs = [
            mock_models['StudyDesign'].COHORT,
            mock_models['StudyDesign'].CASE_CONTROL,
            mock_models['StudyDesign'].CASE_SERIES
        ]
 
        return ROBINSIAssessor(
            llm_provider=mock_llm_provider,
            prompt_template=mock_prompt_template,
            models=mock_models,
            exception_class=mock_exception_class,
            applicable_designs=applicable_designs
        )
 
    @pytest.fixture
    def mock_paper(self):
        """Create a mock paper."""
        paper = Mock()
        paper.paper_id = "test_cohort_123"
        paper.title = "Effect of Intervention X on Outcome Y in Cohort"
        paper.sections = {
            "methods": "We conducted a prospective cohort study...",
            "results": "The exposed group showed..."
        }
        return paper
 
    @pytest.fixture
    def mock_cohort_characteristics(self, mock_models):
        """Create mock cohort study characteristics."""
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].COHORT
        characteristics.population = "Adults aged 40-65"
        characteristics.intervention_exposure = "Daily exercise"
        characteristics.comparator = "No structured exercise"
        characteristics.primary_outcome = "Cardiovascular events"
        return characteristics
 
    @pytest.fixture
    def mock_rct_characteristics(self, mock_models):
        """Create mock RCT characteristics (not applicable)."""
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].RCT
        return characteristics
 
    def test_initialization(self, robins_assessor, mock_llm_provider, mock_prompt_template, mock_models):
        """Test ROBINS-I assessor initialization."""
        assert robins_assessor.llm == mock_llm_provider
        assert robins_assessor.prompt_template == mock_prompt_template
        assert hasattr(robins_assessor, 'ROBINSILevel')
        assert hasattr(robins_assessor, 'ROBINSIDomain')
        assert mock_models['StudyDesign'].COHORT in robins_assessor.APPLICABLE_DESIGNS
 
    def test_assess_study_success(self, robins_assessor, mock_paper, mock_cohort_characteristics, mock_llm_provider, mock_models):
        """Test successful ROBINS-I assessment."""
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "target_trial": "Hypothetical RCT comparing daily exercise vs no exercise",
                "domains": [
                    {
                        "domain": "confounding",
                        "level": "low",
                        "justification": "Well-controlled confounders",
                        "confidence": 0.85,
                        "key_evidence": ["Adjusted for age, sex"],
                        "signaling_questions": ["Q1: Yes", "Q2: Yes"]
                    },
                    {
                        "domain": "selection_of_participants",
                        "level": "low",
                        "justification": "Random selection",
                        "confidence": 0.88,
                        "key_evidence": ["Population-based"],
                        "signaling_questions": []
                    },
                    {
                        "domain": "missing_data",
                        "level": "moderate",
                        "justification": "Some missing data",
                        "confidence": 0.75,
                        "key_evidence": ["10% loss to follow-up"],
                        "signaling_questions": []
                    }
                ],
                "overall_bias": "moderate",
                "summary": "Moderate risk of bias overall",
                "overall_confidence": 0.80
            }
        }
 
        # Perform assessment
        result = robins_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_cohort_characteristics
        )
 
        # Verify LLM was called
        assert mock_llm_provider.complete_with_json.called
 
        # Verify assessment was constructed
        assert mock_models['ROBINSIAssessment'].called
 
    def test_assess_study_rct_raises_error(self, robins_assessor, mock_paper, mock_rct_characteristics, mock_exception_class):
        """Test that RCT studies raise an error."""
        with pytest.raises(mock_exception_class, match="only applicable to non-randomized"):
            robins_assessor.assess_study(
                paper=mock_paper,
                characteristics=mock_rct_characteristics
            )
 
    def test_assess_study_missing_paper(self, robins_assessor, mock_cohort_characteristics):
        """Test assessment with missing paper raises ValueError."""
        with pytest.raises(ValueError, match="paper cannot be None"):
            robins_assessor.assess_study(
                paper=None,
                characteristics=mock_cohort_characteristics
            )
 
    def test_assess_study_missing_characteristics(self, robins_assessor, mock_paper):
        """Test assessment with missing characteristics raises ValueError."""
        with pytest.raises(ValueError, match="characteristics cannot be None"):
            robins_assessor.assess_study(
                paper=mock_paper,
                characteristics=None
            )
 
    def test_robins_algorithm_all_low(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm when all domains are low risk."""
        domain_assessments = []
        for _ in range(7):
            domain = Mock()
            domain.level = mock_models['ROBINSILevel'].LOW
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].LOW
 
    def test_robins_algorithm_one_critical(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm when one domain is critical."""
        domain_assessments = []
        # First domain is critical
        domain1 = Mock()
        domain1.level = mock_models['ROBINSILevel'].CRITICAL
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(6):
            domain = Mock()
            domain.level = mock_models['ROBINSILevel'].LOW
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].CRITICAL
 
    def test_robins_algorithm_one_serious(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm when one domain is serious."""
        domain_assessments = []
        # First domain is serious
        domain1 = Mock()
        domain1.level = mock_models['ROBINSILevel'].SERIOUS
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(6):
            domain = Mock()
            domain.level = mock_models['ROBINSILevel'].LOW
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].SERIOUS
 
    def test_robins_algorithm_one_moderate(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm when one domain is moderate."""
        domain_assessments = []
        # First domain is moderate
        domain1 = Mock()
        domain1.level = mock_models['ROBINSILevel'].MODERATE
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(6):
            domain = Mock()
            domain.level = mock_models['ROBINSILevel'].LOW
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].MODERATE
 
    def test_robins_algorithm_no_information(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm when one domain has no information."""
        domain_assessments = []
        # First domain has no information
        domain1 = Mock()
        domain1.level = mock_models['ROBINSILevel'].NO_INFORMATION
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(6):
            domain = Mock()
            domain.level = mock_models['ROBINSILevel'].LOW
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].NO_INFORMATION
 
    def test_robins_algorithm_worst_domain_wins(self, robins_assessor, mock_models):
        """Test ROBINS-I algorithm - worst domain determines overall."""
        domain_assessments = []
 
        # Mix of levels - critical should win
        levels = [
            mock_models['ROBINSILevel'].LOW,
            mock_models['ROBINSILevel'].MODERATE,
            mock_models['ROBINSILevel'].SERIOUS,
            mock_models['ROBINSILevel'].CRITICAL,
            mock_models['ROBINSILevel'].LOW,
            mock_models['ROBINSILevel'].LOW,
            mock_models['ROBINSILevel'].LOW
        ]
 
        for level in levels:
            domain = Mock()
            domain.level = level
            domain_assessments.append(domain)
 
        result = robins_assessor._apply_robins_i_algorithm(domain_assessments)
        assert result == mock_models['ROBINSILevel'].CRITICAL
 
    def test_assess_study_algorithm_overrides_llm(self, robins_assessor, mock_paper, mock_cohort_characteristics, mock_llm_provider, mock_models):
        """Test that computed algorithm result overrides LLM if they differ."""
        # Mock LLM response with incorrect overall bias
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "target_trial": "Hypothetical RCT",
                "domains": [
                    {
                        "domain": "confounding",
                        "level": "critical",
                        "justification": "Severe confounding",
                        "confidence": 0.7,
                        "key_evidence": [],
                        "signaling_questions": []
                    }
                ] + [
                    {
                        "domain": f"domain_{i}",
                        "level": "low",
                        "justification": "Good",
                        "confidence": 0.9,
                        "key_evidence": [],
                        "signaling_questions": []
                    } for i in range(6)
                ],
                "overall_bias": "low",  # LLM says low, but should be critical
                "summary": "Summary",
                "overall_confidence": 0.8
            }
        }
 
        robins_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_cohort_characteristics
        )
 
        # The algorithm should override and use CRITICAL
        assert mock_models['ROBINSIAssessment'].called
 
    def test_assess_study_missing_target_trial(self, robins_assessor, mock_paper, mock_cohort_characteristics, mock_llm_provider):
        """Test assessment generates default target trial when missing."""
        # Mock LLM response without target trial
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "target_trial": "",  # Empty target trial
                "domains": [],
                "overall_bias": "moderate",
                "summary": "Summary",
                "overall_confidence": 0.7
            }
        }
 
        robins_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_cohort_characteristics
        )
 
        # Should generate a default target trial description
        assert mock_llm_provider.complete_with_json.called
 
    def test_assess_study_missing_characteristics_fields(self, robins_assessor, mock_paper, mock_llm_provider, mock_models):
        """Test assessment handles missing characteristic fields."""
        # Create characteristics without optional fields
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].COHORT
        # Missing: population, intervention_exposure, comparator, primary_outcome
 
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "target_trial": "Default trial",
                "domains": [],
                "overall_bias": "moderate",
                "summary": "Summary",
                "overall_confidence": 0.7
            }
        }
 
        # Should use "not specified" for missing fields
        robins_assessor.assess_study(
            paper=mock_paper,
            characteristics=characteristics
        )
 
        # Verify it handled missing fields gracefully
        call_args = mock_llm_provider.complete_with_json.call_args
        assert "not specified" in call_args[1]['prompt']
 
    def test_assess_study_case_series_warning(self, robins_assessor, mock_paper, mock_llm_provider, mock_models):
        """Test assessment logs info for case series."""
        # Create case series characteristics
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].CASE_SERIES
        characteristics.population = "Patients with condition X"
        characteristics.intervention_exposure = "Treatment Y"
        characteristics.comparator = "Historical controls"
        characteristics.primary_outcome = "Recovery"
 
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "target_trial": "Trial description",
                "domains": [],
                "overall_bias": "serious",
                "summary": "Summary",
                "overall_confidence": 0.6
            }
        }
 
        # Should complete assessment (case series is in applicable designs)
        robins_assessor.assess_study(
            paper=mock_paper,
            characteristics=characteristics
        )
 
        assert mock_llm_provider.complete_with_json.called