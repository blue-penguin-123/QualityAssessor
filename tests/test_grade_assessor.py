"""Tests for GRADE assessor."""
 
import pytest
from unittest.mock import Mock, MagicMock
from enum import Enum
 
 
class TestGRADEAssessor:
    """Tests for GRADE assessment functionality."""
 
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
        class MockGRADELevel(Enum):
            HIGH = "high"
            MODERATE = "moderate"
            LOW = "low"
            VERY_LOW = "very_low"
 
        class MockGRADEDomain(Enum):
            RISK_OF_BIAS = "risk_of_bias"
            INCONSISTENCY = "inconsistency"
            INDIRECTNESS = "indirectness"
            IMPRECISION = "imprecision"
            PUBLICATION_BIAS = "publication_bias"
 
        class MockStudyDesign(Enum):
            RCT = "randomized_controlled_trial"
            COHORT = "cohort_study"
            CASE_CONTROL = "case_control"
            CROSS_SECTIONAL = "cross_sectional"
            CASE_SERIES = "case_series"
            SYSTEMATIC_REVIEW = "systematic_review"
            META_ANALYSIS = "meta_analysis"
            OTHER = "other"
 
        # Create mock model classes
        MockGRADEDomainAssessment = MagicMock()
        MockGRADEAssessment = MagicMock()
 
        return {
            'GRADELevel': MockGRADELevel,
            'GRADEDomain': MockGRADEDomain,
            'GRADEDomainAssessment': MockGRADEDomainAssessment,
            'GRADEAssessment': MockGRADEAssessment,
            'StudyDesign': MockStudyDesign
        }
 
    @pytest.fixture
    def mock_prompt_template(self):
        """Create a mock prompt template."""
        return "Assess GRADE for {study_design}. Methods: {methods}. Results: {results}"
 
    @pytest.fixture
    def mock_exception_class(self):
        """Create a mock exception class."""
        class MockAssessmentError(Exception):
            pass
        return MockAssessmentError
 
    @pytest.fixture
    def grade_assessor(self, mock_llm_provider, mock_prompt_template, mock_models, mock_exception_class):
        """Create a GRADE assessor instance."""
        from quality_assessor import GRADEAssessor
 
        return GRADEAssessor(
            llm_provider=mock_llm_provider,
            prompt_template=mock_prompt_template,
            models=mock_models,
            exception_class=mock_exception_class
        )
 
    @pytest.fixture
    def mock_paper(self):
        """Create a mock paper."""
        paper = Mock()
        paper.paper_id = "test_paper_123"
        paper.sections = {
            "methods": "This is a randomized controlled trial...",
            "results": "The primary outcome showed significant improvement..."
        }
        return paper
 
    @pytest.fixture
    def mock_characteristics(self, mock_models):
        """Create mock study characteristics."""
        characteristics = Mock()
        characteristics.study_design = mock_models['StudyDesign'].RCT
        return characteristics
 
    def test_initialization(self, grade_assessor, mock_llm_provider, mock_prompt_template):
        """Test GRADE assessor initialization."""
        assert grade_assessor.llm == mock_llm_provider
        assert grade_assessor.prompt_template == mock_prompt_template
        assert hasattr(grade_assessor, 'GRADELevel')
        assert hasattr(grade_assessor, 'GRADEDomain')
        assert hasattr(grade_assessor, 'GRADEAssessment')
 
    def test_assess_study_success(self, grade_assessor, mock_paper, mock_characteristics, mock_llm_provider, mock_models):
        """Test successful GRADE assessment."""
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "starting_level": "high",
                "domains": [
                    {
                        "domain": "risk_of_bias",
                        "rating": "not_serious",
                        "justification": "Low risk of bias",
                        "confidence": 0.9,
                        "key_evidence": ["Randomization was adequate"]
                    },
                    {
                        "domain": "inconsistency",
                        "rating": "not_serious",
                        "justification": "Consistent results",
                        "confidence": 0.85,
                        "key_evidence": ["All outcomes aligned"]
                    }
                ],
                "final_grade": "high",
                "upgrades": {},
                "summary": "High quality evidence",
                "overall_confidence": 0.88
            }
        }
 
        # Perform assessment
        result = grade_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_characteristics,
            hypothesis=None
        )
 
        # Verify LLM was called
        assert mock_llm_provider.complete_with_json.called
        call_args = mock_llm_provider.complete_with_json.call_args
 
        # Verify prompt was formatted correctly
        assert "randomized_controlled_trial" in call_args[1]['prompt']
 
        # Verify GRADEAssessment was constructed
        assert mock_models['GRADEAssessment'].called
 
    def test_assess_study_missing_paper(self, grade_assessor, mock_characteristics):
        """Test assessment with missing paper raises ValueError."""
        with pytest.raises(ValueError, match="paper cannot be None"):
            grade_assessor.assess_study(
                paper=None,
                characteristics=mock_characteristics
            )
 
    def test_assess_study_missing_characteristics(self, grade_assessor, mock_paper):
        """Test assessment with missing characteristics raises ValueError."""
        with pytest.raises(ValueError, match="characteristics cannot be None"):
            grade_assessor.assess_study(
                paper=mock_paper,
                characteristics=None
            )
 
    def test_assess_study_missing_methods_section(self, grade_assessor, mock_characteristics, mock_llm_provider):
        """Test assessment handles missing methods section."""
        # Create paper without methods section
        paper = Mock()
        paper.paper_id = "test_paper"
        paper.sections = {
            "results": "Results here..."
        }
 
        # Mock LLM response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "starting_level": "low",
                "domains": [],
                "final_grade": "low",
                "upgrades": {},
                "summary": "Low quality",
                "overall_confidence": 0.5
            }
        }
 
        # Should not raise error, just use empty string for methods
        grade_assessor.assess_study(
            paper=paper,
            characteristics=mock_characteristics
        )
 
        # Verify it was called successfully
        assert mock_llm_provider.complete_with_json.called
 
    def test_determine_starting_level_rct(self, grade_assessor, mock_models):
        """Test starting level determination for RCT."""
        starting_level = grade_assessor._determine_starting_level(
            mock_models['StudyDesign'].RCT
        )
        assert starting_level == mock_models['GRADELevel'].HIGH
 
    def test_determine_starting_level_cohort(self, grade_assessor, mock_models):
        """Test starting level determination for cohort study."""
        starting_level = grade_assessor._determine_starting_level(
            mock_models['StudyDesign'].COHORT
        )
        assert starting_level == mock_models['GRADELevel'].LOW
 
    def test_determine_starting_level_case_series(self, grade_assessor, mock_models):
        """Test starting level determination for case series."""
        starting_level = grade_assessor._determine_starting_level(
            mock_models['StudyDesign'].CASE_SERIES
        )
        assert starting_level == mock_models['GRADELevel'].VERY_LOW
 
    def test_assess_study_with_downgrades(self, grade_assessor, mock_paper, mock_characteristics, mock_llm_provider, mock_models):
        """Test assessment with domain downgrades."""
        # Mock LLM response with serious issues
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "starting_level": "high",
                "domains": [
                    {
                        "domain": "risk_of_bias",
                        "rating": "serious",
                        "justification": "High risk of bias",
                        "confidence": 0.7,
                        "key_evidence": ["No blinding"]
                    },
                    {
                        "domain": "imprecision",
                        "rating": "very_serious",
                        "justification": "Wide confidence intervals",
                        "confidence": 0.6,
                        "key_evidence": ["Small sample size"]
                    }
                ],
                "final_grade": "low",
                "upgrades": {},
                "summary": "Quality downgraded due to bias and imprecision",
                "overall_confidence": 0.65
            }
        }
 
        result = grade_assessor.assess_study(
            paper=mock_paper,
            characteristics=mock_characteristics
        )
 
        # Verify assessment was created
        assert mock_models['GRADEAssessment'].called
 
    def test_assess_study_llm_error_handling(self, grade_assessor, mock_paper, mock_characteristics, mock_llm_provider, mock_exception_class):
        """Test error handling when LLM call fails."""
        # Make LLM raise an exception
        mock_llm_provider.complete_with_json.side_effect = Exception("LLM API error")
 
        # Should raise custom exception class
        with pytest.raises(mock_exception_class):
            grade_assessor.assess_study(
                paper=mock_paper,
                characteristics=mock_characteristics
            )
 
    def test_assess_study_invalid_json_response(self, grade_assessor, mock_paper, mock_characteristics, mock_llm_provider, mock_exception_class):
        """Test error handling for invalid JSON response."""
        # Mock incomplete response
        mock_llm_provider.complete_with_json.return_value = {
            "json_data": {
                "starting_level": "high"
                # Missing required fields
            }
        }
 
        # Should raise custom exception for missing keys
        with pytest.raises(mock_exception_class):
            grade_assessor.assess_study(
                paper=mock_paper,
                characteristics=mock_characteristics
            )