"""Quality gates for sampled technical-indicator text generation outputs."""

from __future__ import annotations

import re

from priorstock.config import TextGenerationConfig
from priorstock.exceptions import TextGenerationValidationError

DIGIT_PATTERN = re.compile(r"\d")
NUMBER_WORD_PATTERN = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|first|second|third)\b",
    flags=re.IGNORECASE,
)
ZONE_TERMS = [
    "deeply oversold",
    "oversold",
    "neutral-low",
    "neutral low",
    "neutral-high",
    "neutral high",
    "overbought",
    "deeply overbought",
]
TREND_SPACING_TERMS = [
    "compressed",
    "tight",
    "narrow",
    "clustered",
    "fanned",
    "spread",
    "separated",
]
TREND_CONVERGENCE_TERMS = [
    "converging",
    "diverging",
    "parallel",
]
MA_REFERENCE_TERMS = [
    "moving average",
    "moving averages",
    "ma cluster",
    "average cluster",
    "average stack",
]
MOMENTUM_ZERO_TERMS = [
    "above zero",
    "below zero",
    "positive side of zero",
    "negative side of zero",
    "north of zero",
    "south of zero",
]
MOMENTUM_DIRECTION_TERMS = [
    "positive",
    "negative",
]
MOMENTUM_HISTOGRAM_TERMS = [
    "histogram",
    "bar expansion",
    "bar contraction",
]
MOMENTUM_DYNAMIC_TERMS = [
    "expanding",
    "contracting",
    "widening",
    "narrowing",
]
CONSISTENCY_TERMS = [
    "consistent",
    "coherent",
    "aligned",
    "divergent",
    "divergence",
    "inconsistent",
    "at odds",
]
KDJ_REFERENCE_TERMS = [
    "kdj",
]
KD_RELATIONSHIP_TERMS = [
    "k above d",
    "k below d",
    "d above k",
    "d below k",
    "k-d",
    "k over d",
    "k under d",
]
J_REFERENCE_PATTERN = re.compile(r"\bj\b", flags=re.IGNORECASE)
RSI_REFERENCE_TERMS = [
    "rsi",
]
LEAD_LAG_TERMS = [
    "lead",
    "leads",
    "leading",
    "lag",
    "lags",
    "lagging",
]
AGREEMENT_TERMS = [
    "agree",
    "agrees",
    "agreement",
    "disagree",
    "disagrees",
    "disagreement",
]
VOLATILITY_REGIME_TERMS = [
    "squeeze",
    "tight",
    "normal",
    "wide",
    "extreme",
    "expanded",
    "compressed",
    "bandwidth",
]
PRICE_BAND_POSITION_TERMS = [
    "lower band",
    "middle band",
    "mid band",
    "upper band",
    "within the bands",
    "inside the bands",
    "outside the bands",
    "below the lower band",
    "above the upper band",
    "lower region",
    "midpoint",
    "upper region",
]
VOLATILITY_DYNAMIC_TERMS = [
    "contracting",
    "expanding",
]
VOLUME_LEVEL_TERMS = [
    "very low",
    "below average",
    "average",
    "above average",
    "surge",
]
SYSTEM_COUNT_TERMS = [
    "all systems",
    "most systems",
    "several systems",
    "few systems",
    "the systems",
]
STRUCTURAL_PATTERN_TERMS = [
    "compressed spring",
    "momentum exhaustion",
    "structural transition",
    "mature trend extension",
    "range equilibrium",
    "divergent stress",
    "micro-state",
    "microstate",
    "pattern",
]


def _contains_any_term(response_text: str, candidate_terms: list[str]) -> bool:
    """Return whether the response text contains at least one term from one controlled vocabulary set."""

    lowered_text = response_text.lower()
    return any(candidate_term in lowered_text for candidate_term in candidate_terms)


def _contains_price_position_relative_to_moving_average_cluster(response_text: str) -> bool:
    """Return whether the text states price position relative to the moving-average cluster."""

    lowered_text = response_text.lower()
    cluster_is_named = _contains_any_term(lowered_text, MA_REFERENCE_TERMS)
    price_position_is_named = "price" in lowered_text and any(
        keyword in lowered_text for keyword in ["above", "below", "inside", "within", "through", "relative to"]
    )
    return cluster_is_named and price_position_is_named


def _contains_momentum_position_relative_to_zero(response_text: str) -> bool:
    """Return whether the text states the momentum system's relation to zero or sign."""

    lowered_text = response_text.lower()
    return _contains_any_term(lowered_text, MOMENTUM_ZERO_TERMS) or _contains_any_term(
        lowered_text, MOMENTUM_DIRECTION_TERMS
    )


def _contains_histogram_dynamic(response_text: str) -> bool:
    """Return whether the text explicitly discusses histogram dynamics."""

    lowered_text = response_text.lower()
    return _contains_any_term(lowered_text, MOMENTUM_HISTOGRAM_TERMS) and _contains_any_term(
        lowered_text, MOMENTUM_DYNAMIC_TERMS
    )


def _contains_kdj_detail(response_text: str) -> bool:
    """Return whether the text separately covers KDJ zone, K-D relation, and J behavior."""

    lowered_text = response_text.lower()
    return (
        _contains_any_term(lowered_text, KDJ_REFERENCE_TERMS)
        and _contains_any_term(lowered_text, ZONE_TERMS)
        and _contains_any_term(lowered_text, KD_RELATIONSHIP_TERMS)
        and J_REFERENCE_PATTERN.search(lowered_text) is not None
    )


def _contains_rsi_detail(response_text: str) -> bool:
    """Return whether the text separately covers RSI zone and short-vs-long lead-lag structure."""

    lowered_text = response_text.lower()
    return (
        _contains_any_term(lowered_text, RSI_REFERENCE_TERMS)
        and _contains_any_term(lowered_text, ZONE_TERMS)
        and _contains_any_term(lowered_text, LEAD_LAG_TERMS)
    )


def _contains_explicit_oscillator_agreement(response_text: str) -> bool:
    """Return whether the text explicitly states KDJ/RSI agreement or disagreement."""

    lowered_text = response_text.lower()
    return (
        "kdj" in lowered_text
        and "rsi" in lowered_text
        and _contains_any_term(lowered_text, AGREEMENT_TERMS)
    )


def _contains_volatility_relation_to_trend_or_momentum(response_text: str) -> bool:
    """Return whether the text links volatility dynamics to trend or momentum."""

    lowered_text = response_text.lower()
    mentions_volatility_dynamics = _contains_any_term(lowered_text, VOLATILITY_DYNAMIC_TERMS)
    mentions_trend_or_momentum = "trend" in lowered_text or "momentum" in lowered_text
    return mentions_volatility_dynamics and mentions_trend_or_momentum


def _contains_cross_system_counting_language(response_text: str) -> bool:
    """Return whether the text summarizes how many systems are coherent or contradictory."""

    lowered_text = response_text.lower()
    return _contains_any_term(lowered_text, SYSTEM_COUNT_TERMS) or (
        "systems" in lowered_text and _contains_any_term(lowered_text, CONSISTENCY_TERMS)
    )


def _validate_local_indicator_snapshot(indicator_snapshot: dict, text_generation_config: TextGenerationConfig) -> dict:
    """Validate that locally computed Bollinger-derived values are numerically self-consistent."""

    upper_value = float(indicator_snapshot["upper"])
    middle_value = float(indicator_snapshot["mid"])
    lower_value = float(indicator_snapshot["lower"])
    close_value = float(indicator_snapshot["close"])
    expected_bandwidth = float(indicator_snapshot["bollinger_bandwidth"])
    expected_percent_b = float(indicator_snapshot["bollinger_percent_b"])

    bandwidth_is_defined = middle_value != 0.0
    percent_b_is_defined = upper_value != lower_value

    derived_bandwidth = None
    derived_percent_b = None
    bandwidth_error = None
    percent_b_error = None

    if bandwidth_is_defined:
        derived_bandwidth = (upper_value - lower_value) / middle_value
        bandwidth_error = abs(derived_bandwidth - expected_bandwidth)
    if percent_b_is_defined:
        derived_percent_b = (close_value - lower_value) / (upper_value - lower_value)
        percent_b_error = abs(derived_percent_b - expected_percent_b)

    passed = True
    if text_generation_config.require_local_indicator_consistency_check:
        passed = (
            derived_bandwidth is not None
            and derived_percent_b is not None
            and bandwidth_error is not None
            and percent_b_error is not None
            and bandwidth_error <= text_generation_config.bandwidth_validation_tolerance
            and percent_b_error <= text_generation_config.percent_b_validation_tolerance
        )

    return {
        "passed": passed,
        "derived_bandwidth": derived_bandwidth,
        "expected_bandwidth": expected_bandwidth,
        "bandwidth_error": bandwidth_error,
        "derived_percent_b": derived_percent_b,
        "expected_percent_b": expected_percent_b,
        "percent_b_error": percent_b_error,
    }


def _compute_dimension_coverage(response_text: str) -> dict[str, bool]:
    """Check whether the current prompt's five analytical dimensions are substantively covered."""

    lowered_text = response_text.lower()
    return {
        "trend_structure": (
            _contains_any_term(lowered_text, MA_REFERENCE_TERMS)
            and _contains_any_term(lowered_text, TREND_SPACING_TERMS)
            and _contains_any_term(lowered_text, TREND_CONVERGENCE_TERMS)
            and _contains_price_position_relative_to_moving_average_cluster(lowered_text)
        ),
        "momentum_profile": (
            _contains_momentum_position_relative_to_zero(lowered_text)
            and _contains_histogram_dynamic(lowered_text)
            and _contains_any_term(lowered_text, CONSISTENCY_TERMS)
            and "trend" in lowered_text
        ),
        "oscillator_regime": (
            _contains_kdj_detail(lowered_text)
            and _contains_rsi_detail(lowered_text)
            and _contains_explicit_oscillator_agreement(lowered_text)
        ),
        "volatility_context": (
            _contains_any_term(lowered_text, VOLATILITY_REGIME_TERMS)
            and _contains_any_term(lowered_text, PRICE_BAND_POSITION_TERMS)
            and _contains_volatility_relation_to_trend_or_momentum(lowered_text)
        ),
        "cross_indicator_synthesis": (
            _contains_cross_system_counting_language(lowered_text)
            and _contains_any_term(lowered_text, STRUCTURAL_PATTERN_TERMS)
        ),
        "volume_context": _contains_any_term(lowered_text, VOLUME_LEVEL_TERMS),
    }


def validate_generated_text_record(generated_record: dict, text_generation_config: TextGenerationConfig) -> dict:
    """Validate one generated qualitative paragraph against the current prompt contract."""

    generated_text = str(generated_record["generated_text"])
    lowered_text = generated_text.lower()
    word_count = len(generated_text.split())

    forbidden_phrase_matches = [
        forbidden_phrase
        for forbidden_phrase in text_generation_config.banned_phrases
        if forbidden_phrase.lower() in lowered_text
    ]
    contains_digit_characters = DIGIT_PATTERN.search(generated_text) is not None
    contains_number_words = NUMBER_WORD_PATTERN.search(generated_text) is not None
    contains_percent_symbol = "%" in generated_text
    paragraph_segments = [segment.strip() for segment in generated_text.splitlines() if segment.strip()]
    is_single_paragraph = len(paragraph_segments) == 1

    dimension_coverage = _compute_dimension_coverage(generated_text)
    local_indicator_validation = _validate_local_indicator_snapshot(
        generated_record["indicator_snapshot"],
        text_generation_config,
    )

    passed = (
        not forbidden_phrase_matches
        and all(dimension_coverage.values())
        and local_indicator_validation["passed"]
        and (not text_generation_config.require_zero_digit_output or not contains_digit_characters)
        and (not text_generation_config.forbid_number_word_output or not contains_number_words)
        and (not text_generation_config.require_zero_digit_output or not contains_percent_symbol)
        and (not text_generation_config.require_single_paragraph_output or is_single_paragraph)
        and (
            not text_generation_config.enable_word_count_check
            or (
                text_generation_config.minimum_output_word_count
                <= word_count
                <= text_generation_config.maximum_output_word_count
            )
        )
    )

    return {
        "record_id": generated_record["record_id"],
        "stock_id": generated_record["stock_id"],
        "trade_date": generated_record["trade_date"],
        "passed": passed,
        "word_count": word_count,
        "forbidden_phrase_matches": forbidden_phrase_matches,
        "contains_digit_characters": contains_digit_characters,
        "contains_number_words": contains_number_words,
        "contains_percent_symbol": contains_percent_symbol,
        "is_single_paragraph": is_single_paragraph,
        "dimension_coverage": dimension_coverage,
        "derived_bandwidth": local_indicator_validation["derived_bandwidth"],
        "expected_bandwidth": local_indicator_validation["expected_bandwidth"],
        "bandwidth_error": local_indicator_validation["bandwidth_error"],
        "derived_percent_b": local_indicator_validation["derived_percent_b"],
        "expected_percent_b": local_indicator_validation["expected_percent_b"],
        "percent_b_error": local_indicator_validation["percent_b_error"],
        "generated_text": generated_text,
    }


def summarize_validation_results(validation_results: list[dict]) -> dict:
    """Aggregate multiple validation records into a summary payload."""

    passed_count = sum(1 for record in validation_results if record["passed"])
    return {
        "sample_count": len(validation_results),
        "passed_count": passed_count,
        "failed_count": len(validation_results) - passed_count,
        "all_passed": passed_count == len(validation_results),
    }


def raise_if_validation_failed(summary_payload: dict) -> None:
    """Raise one explicit exception when the sampled validation batch is not fully clean."""

    if not summary_payload["all_passed"]:
        raise TextGenerationValidationError(
            "Technical-indicator text validation failed. "
            f"Passed {summary_payload['passed_count']} of {summary_payload['sample_count']} sampled outputs."
        )
