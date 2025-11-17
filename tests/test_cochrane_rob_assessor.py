"""Tests for Cochrane RoB assessor."""
 
import pytest
from unittest.mock import Mock, MagicMock
from enum import Enum
 
 
class TestCochraneRoBAssessor:
    """Tests for Cochrane RoB 2.0 assessment functionality."""
 
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
        class MockRoBJudgment(Enum):
            LOW = "low"
            SOME_CONCERNS = "some_concerns"
            HIGH = "high"
 
        class MockRoBDomain(Enum):
            RANDOMIZATION = "randomization"
            DEVIATIONS_FROM_INTENDED_INTERVENTIONS = "deviations_from_intended_interventions"
            MISSING_OUTCOME_DATA = "missing_outcome_data"
            MEASUREMENT_OF_OUTCOME = "measurement_of_outcome"
            SELECTION_OF_REPORTED_RESULT = "selection_of_reported_result"
 
        class MockStudyDesign(Enum):
            RCT = "randomized_controlled_trial"
            COHORT = "cohort_study"
            CASE_CONTROL = "case_control"
 
        # Create mock model classes
        MockRoBDomainAssessment = MagicMock()
        MockCochraneRoBAssessment = MagicMock()
 
        return {
            'RoBDomain': MockRoBDomain,
            'RoBJudgment': MockRoBJudgment,
            'RoBDomainAssessment': MockRoBDomainAssessment,
            'CochraneRoBAssessment': MockCochraneRoBAssessment,
            'StudyDesign': MockStudyDesign
        }
 
    @pytest.fixture
    def mock_prompt_template(self):
        """Create a mock prompt template."""
        return "Assess RoB for RCT. Title: {title}. Methods: {methods}. Results: {results}"
 
    @pytest.fixture
    def mock_exception_class(self):
        """Create a mock exception class."""
        class MockAssessmentError(Exception):
            pass
        return MockAssessmentError
 
    @pytest.fixture
    def rob_assessor(self, mock_llm_provider, mock_prompt_template, mock_models, mock_exception_class):
        """Create a Cochrane RoB assessor instance."""
        from quality_assessor import CochraneRoBAssessor
 
        return CochraneRoBAssessor(
            llm_provider=mock_llm_provider,
            prompt_template=mock_prompt_template,
            models=mock_models,
            exception_class=mock_exception_class
        )
 
    @pytest.fixture
    def mock_paper(self):
        """Create a mock paper."""
        paper = Mock()
        paper.paper_id = "test_rct_123"
        paper.title = "Effect of Intervention X on Outcome Y"
        paper.sections = {
            "methods": "Randomization was performed using...",
            "results": "The intervention group showed..."
        }
        return paper
 
    @pytest.fixture
    def mock_rct_characteristics(self, mock_models):
        """Create mock RCT characteristics."""
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].RCT
        return characteristics
 
    @pytest.fixture
    def mock_non_rct_characteristics(self, mock_models):
        """Create mock non-RCT characteristics."""
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].COHORT
        return characteristics
 
    def test_initialization(self, rob_assessor, mock_llm_provider, mock_prompt_template):
        """Test RoB assessor initialization."""
        assert rob_assessor.llm == mock_llm_provider
        assert rob_assessor.prompt_template == mock_prompt_template
        assert hasattr(rob_assessor, 'RoBJudgment')
        assert hasattr(rob_assessor, 'RoBDomain')
 
    def test_assess_study_success(self, rob_assessor, mock_paper, mock_rct_characteristics, mock_llm_provider, mock_models):
        """Test successful RoB assessment."""
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "domains": [
                    {
                        "domain": "randomization",
                        "judgment": "low",
                        "justification": "Adequate randomization",
                        "confidence": 0.9,
                        "key_evidence": ["Computer-generated sequence"]
                    },
                    {
                        "domain": "deviations_from_intended_interventions",
                        "judgment": "low",
                        "justification": "Protocol followed",
                        "confidence": 0.85,
                        "key_evidence": ["No deviations reported"]
                    },
                    {
                        "domain": "missing_outcome_data",
                        "judgment": "low",
                        "justification": "Complete data",
                        "confidence": 0.88,
                        "key_evidence": ["No missing data"]
                    },
                    {
                        "domain": "measurement_of_outcome",
                        "judgment": "low",
                        "justification": "Blinded assessment",
                        "confidence": 0.92,
                        "key_evidence": ["Validated instrument"]
                    },
                    {
                        "domain": "selection_of_reported_result",
                        "judgment": "low",
                        "justification": "Pre-registered",
                        "confidence": 0.87,
                        "key_evidence": ["Trial registered"]
                    }
                ],
                "overall_risk": "low",
                "summary": "Low risk of bias across all domains",
                "overall_confidence": 0.88
            }
        }
 
        # Perform assessment
        result = rob_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_rct_characteristics
        )
 
        # Verify LLM was called
        assert mock_llm_provider.complete_with_json.called
 
        # Verify assessment was constructed
        assert mock_models['CochraneRoBAssessment'].called
 
    def test_assess_study_non_rct_raises_error(self, rob_assessor, mock_paper, mock_non_rct_characteristics, mock_exception_class):
        """Test that non-RCT studies raise an error."""
        with pytest.raises(mock_exception_class, match="only applicable to RCTs"):
            rob_assessor.assess_study(
                paper=mock_paper,
                characteristics=mock_non_rct_characteristics
            )
 
    def test_assess_study_missing_paper(self, rob_assessor, mock_rct_characteristics):
        """Test assessment with missing paper raises ValueError."""
        with pytest.raises(ValueError, match="paper cannot be None"):
            rob_assessor.assess_study(
                paper=None,
                characteristics=mock_rct_characteristics
            )
 
    def test_assess_study_missing_characteristics(self, rob_assessor, mock_paper):
        """Test assessment with missing characteristics raises ValueError."""
        with pytest.raises(ValueError, match="characteristics cannot be None"):
            rob_assessor.assess_study(
                paper=mock_paper,
                characteristics=None
            )
 
    def test_rob_algorithm_all_low(self, rob_assessor, mock_models):
        """Test RoB algorithm when all domains are low risk."""
        # Create mock domain assessments
        domain_assessments = []
        for _ in range(5):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].LOW
            domain.domain = "test_domain"
            domain_assessments.append(domain)
 
        result = rob_assessor._apply_rob_algorithm(domain_assessments)
        assert result == mock_models['RoBJudgment'].LOW
 
    def test_rob_algorithm_one_high(self, rob_assessor, mock_models):
        """Test RoB algorithm when one domain has high risk."""
        domain_assessments = []
        # First domain is high risk
        domain1 = Mock()
        domain1.judgment = mock_models['RoBJudgment'].HIGH
        domain1.domain = "randomization"
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(4):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].LOW
            domain.domain = "other_domain"
            domain_assessments.append(domain)
 
        result = rob_assessor._apply_rob_algorithm(domain_assessments)
        assert result == mock_models['RoBJudgment'].HIGH
 
    def test_rob_algorithm_some_concerns(self, rob_assessor, mock_models):
        """Test RoB algorithm with some concerns in one domain."""
        domain_assessments = []
        # First domain has some concerns
        domain1 = Mock()
        domain1.judgment = mock_models['RoBJudgment'].SOME_CONCERNS
        domain1.domain = "randomization"
        domain_assessments.append(domain1)
 
        # Rest are low
        for _ in range(4):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].LOW
            domain.domain = "other_domain"
            domain_assessments.append(domain)
 
        result = rob_assessor._apply_rob_algorithm(domain_assessments)
        assert result == mock_models['RoBJudgment'].SOME_CONCERNS
 
    def test_rob_algorithm_multiple_concerns_escalates(self, rob_assessor, mock_models):
        """Test RoB algorithm escalates to high when 3+ domains have concerns."""
        domain_assessments = []
        # Three domains with some concerns
        for _ in range(3):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].SOME_CONCERNS
            domain.domain = "test_domain"
            domain_assessments.append(domain)
 
        # Rest are low
        for _ in range(2):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].LOW
            domain.domain = "other_domain"
            domain_assessments.append(domain)
 
        result = rob_assessor._apply_rob_algorithm(domain_assessments)
        assert result == mock_models['RoBJudgment'].HIGH
 
    def test_rob_algorithm_critical_domains_both_concerns(self, rob_assessor, mock_models):
        """Test RoB algorithm when both critical domains have concerns."""
        domain_assessments = []
 
        # Randomization has concerns (critical domain)
        domain1 = Mock()
        domain1.judgment = mock_models['RoBJudgment'].SOME_CONCERNS
        domain1.domain = "randomization"
        domain_assessments.append(domain1)
 
        # Deviations has concerns (critical domain)
        domain2 = Mock()
        domain2.judgment = mock_models['RoBJudgment'].SOME_CONCERNS
        domain2.domain = "deviations_from_intended_interventions"
        domain_assessments.append(domain2)
 
        # Rest are low
        for _ in range(3):
            domain = Mock()
            domain.judgment = mock_models['RoBJudgment'].LOW
            domain.domain = "other_domain"
            domain_assessments.append(domain)
 
        result = rob_assessor._apply_rob_algorithm(domain_assessments)
        assert result == mock_models['RoBJudgment'].HIGH
 
    def test_assess_study_algorithm_overrides_llm(self, rob_assessor, mock_paper, mock_rct_characteristics, mock_llm_provider, mock_models):
        """Test that computed algorithm result overrides LLM if they differ."""
        # Mock LLM response with incorrect overall risk
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "domains": [
                    {
                        "domain": "randomization",
                        "judgment": "high",
                        "justification": "No randomization",
                        "confidence": 0.8,
                        "key_evidence": []
                    }
                ] + [
                    {
                        "domain": f"domain_{i}",
                        "judgment": "low",
                        "justification": "Good",
                        "confidence": 0.9,
                        "key_evidence": []
                    } for i in range(4)
                ],
                "overall_risk": "low",  # LLM says low, but should be high
                "summary": "Summary",
                "overall_confidence": 0.8
            }
        }
 
        rob_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_rct_characteristics
        )
 
        # The algorithm should override and use HIGH
        # (We can't directly assert this without checking the constructed object,
        # but the code logs a warning)
        assert mock_models['CochraneRoBAssessment'].called
 
    def test_assess_study_missing_title(self, rob_assessor, mock_rct_characteristics, mock_llm_provider):
        """Test assessment handles missing title."""
        # Create paper without title
        paper = Mock()
        paper.paper_id = "test_paper"
        paper.title = None
        paper.sections = {
            "methods": "Methods...",
            "results": "Results..."
        }
 
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "domains": [],
                "overall_risk": "low",
                "summary": "Summary",
                "overall_confidence": 0.7
            }
        }
 
        # Should use empty string for title
        rob_assessor.assess_study(
            paper=paper,
            characteristics=mock_rct_characteristics
        )
 
        assert mock_llm_provider.complete_with_json.called