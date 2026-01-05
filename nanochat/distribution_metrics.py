"""Distribution comparison metrics for ICML 2026 evaluation.

Computes KL divergence, total variation, chi-squared, and coverage metrics
to compare field value distributions between generated samples and ground truth.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DistributionComparison:
    """Results of comparing two distributions for a single field."""
    field_name: str
    value_type: str

    # Counts
    real_total: int
    gen_total: int
    unique_real: int
    unique_gen: int

    # Metrics
    kl_divergence: float          # KL(real || gen), lower is better
    reverse_kl: float             # KL(gen || real)
    total_variation: float        # TV distance, [0, 1], lower is better
    chi_squared: float            # Chi-squared statistic
    chi_squared_pvalue: float     # p-value for chi-squared test

    # Coverage
    coverage: float               # Fraction of real values that appear in generated
    mode_match: bool              # Whether modes match
    real_mode: Optional[str]
    gen_mode: Optional[str]

    # Top values
    real_top3: List[Tuple[str, int]]  # Top 3 values in real distribution
    gen_top3: List[Tuple[str, int]]   # Top 3 values in generated distribution

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "field_name": self.field_name,
            "value_type": self.value_type,
            "real_total": self.real_total,
            "gen_total": self.gen_total,
            "unique_real": self.unique_real,
            "unique_gen": self.unique_gen,
            "kl_divergence": self.kl_divergence,
            "reverse_kl": self.reverse_kl,
            "total_variation": self.total_variation,
            "chi_squared": self.chi_squared,
            "chi_squared_pvalue": self.chi_squared_pvalue,
            "coverage": self.coverage,
            "mode_match": self.mode_match,
            "real_mode": self.real_mode,
            "gen_mode": self.gen_mode,
            "real_top3": self.real_top3,
            "gen_top3": self.gen_top3,
        }


def counts_to_probs(counts: Dict[str, int], smoothing: float = 1e-10) -> Dict[str, float]:
    """Convert count dictionary to probability dictionary with smoothing.

    Args:
        counts: Dictionary of value -> count
        smoothing: Small value added to prevent zero probabilities

    Returns:
        Dictionary of value -> probability
    """
    total = sum(counts.values()) + smoothing * len(counts)
    return {k: (v + smoothing) / total for k, v in counts.items()}


def compute_kl_divergence(
    p_counts: Dict[str, int],
    q_counts: Dict[str, int],
    smoothing: float = 1e-10,
) -> float:
    """Compute KL divergence KL(P || Q).

    KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))

    Lower values indicate Q is closer to P.

    Args:
        p_counts: Count dictionary for P (typically real/reference distribution)
        q_counts: Count dictionary for Q (typically generated distribution)
        smoothing: Smoothing for zero probabilities

    Returns:
        KL divergence in nats (natural log)
    """
    # Get union of all values
    all_values = set(p_counts.keys()) | set(q_counts.keys())

    # Add smoothing to both distributions
    p_total = sum(p_counts.values()) + smoothing * len(all_values)
    q_total = sum(q_counts.values()) + smoothing * len(all_values)

    kl = 0.0
    for v in all_values:
        p_v = (p_counts.get(v, 0) + smoothing) / p_total
        q_v = (q_counts.get(v, 0) + smoothing) / q_total
        if p_v > 0:
            kl += p_v * math.log(p_v / q_v)

    return kl


def compute_total_variation(
    p_counts: Dict[str, int],
    q_counts: Dict[str, int],
) -> float:
    """Compute Total Variation distance.

    TV(P, Q) = 0.5 * sum_x |P(x) - Q(x)|

    Values in [0, 1], lower is better.

    Args:
        p_counts: Count dictionary for P
        q_counts: Count dictionary for Q

    Returns:
        Total variation distance
    """
    all_values = set(p_counts.keys()) | set(q_counts.keys())

    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())

    if p_total == 0 or q_total == 0:
        return 1.0  # Maximum distance if one is empty

    tv = 0.0
    for v in all_values:
        p_v = p_counts.get(v, 0) / p_total
        q_v = q_counts.get(v, 0) / q_total
        tv += abs(p_v - q_v)

    return tv / 2.0


def compute_chi_squared(
    real_counts: Dict[str, int],
    gen_counts: Dict[str, int],
    min_expected: float = 5.0,
) -> Tuple[float, float]:
    """Compute chi-squared statistic and p-value.

    Tests whether generated distribution matches real distribution.
    Uses generated counts as observed, real counts (scaled) as expected.

    Args:
        real_counts: Count dictionary for real/reference distribution
        gen_counts: Count dictionary for generated distribution
        min_expected: Minimum expected count (values below this are grouped)

    Returns:
        Tuple of (chi_squared_statistic, p_value)
    """
    try:
        from scipy.stats import chisquare
    except ImportError:
        return (float('nan'), float('nan'))

    all_values = set(real_counts.keys()) | set(gen_counts.keys())

    real_total = sum(real_counts.values())
    gen_total = sum(gen_counts.values())

    if real_total == 0 or gen_total == 0:
        return (float('nan'), float('nan'))

    # Scale real counts to match gen total
    scale = gen_total / real_total

    observed = []
    expected = []
    for v in sorted(all_values):
        obs = gen_counts.get(v, 0)
        exp = real_counts.get(v, 0) * scale

        # Skip values with very low expected counts
        if exp < min_expected and obs == 0:
            continue

        observed.append(obs)
        expected.append(max(exp, 0.1))  # Avoid division by zero

    if len(observed) < 2:
        return (float('nan'), float('nan'))

    try:
        stat, pvalue = chisquare(observed, expected)
        return (float(stat), float(pvalue))
    except Exception:
        return (float('nan'), float('nan'))


def compute_coverage(
    real_counts: Dict[str, int],
    gen_counts: Dict[str, int],
) -> float:
    """Compute coverage: fraction of real values that appear in generated.

    Args:
        real_counts: Count dictionary for real distribution
        gen_counts: Count dictionary for generated distribution

    Returns:
        Fraction of real values present in generated, [0, 1]
    """
    if not real_counts:
        return 1.0  # No values to cover

    real_values = set(real_counts.keys())
    gen_values = set(gen_counts.keys())

    covered = len(real_values & gen_values)
    return covered / len(real_values)


def compare_distributions(
    real_counts: Dict[str, int],
    gen_counts: Dict[str, int],
    field_name: str,
    value_type: str = "unknown",
) -> DistributionComparison:
    """Compare two distributions and compute all metrics.

    Args:
        real_counts: Count dictionary for real/reference distribution
        gen_counts: Count dictionary for generated distribution
        field_name: Name of the field being compared
        value_type: Type of values ("bool", "enum", "string", etc.)

    Returns:
        DistributionComparison with all metrics
    """
    # Compute metrics
    kl = compute_kl_divergence(real_counts, gen_counts)
    reverse_kl = compute_kl_divergence(gen_counts, real_counts)
    tv = compute_total_variation(real_counts, gen_counts)
    chi2, pvalue = compute_chi_squared(real_counts, gen_counts)
    coverage = compute_coverage(real_counts, gen_counts)

    # Modes
    real_mode = max(real_counts, key=real_counts.get) if real_counts else None
    gen_mode = max(gen_counts, key=gen_counts.get) if gen_counts else None
    mode_match = real_mode == gen_mode

    # Top 3
    real_top3 = sorted(real_counts.items(), key=lambda x: -x[1])[:3]
    gen_top3 = sorted(gen_counts.items(), key=lambda x: -x[1])[:3]

    return DistributionComparison(
        field_name=field_name,
        value_type=value_type,
        real_total=sum(real_counts.values()),
        gen_total=sum(gen_counts.values()),
        unique_real=len(real_counts),
        unique_gen=len(gen_counts),
        kl_divergence=kl,
        reverse_kl=reverse_kl,
        total_variation=tv,
        chi_squared=chi2,
        chi_squared_pvalue=pvalue,
        coverage=coverage,
        mode_match=mode_match,
        real_mode=real_mode,
        gen_mode=gen_mode,
        real_top3=real_top3,
        gen_top3=gen_top3,
    )


@dataclass
class SchemaComparisonResult:
    """Results of comparing distributions across all fields for a schema."""
    schema_name: str
    real_samples: int
    gen_samples: int
    field_comparisons: Dict[str, DistributionComparison]

    # Aggregate metrics
    mean_kl: float = 0.0
    mean_tv: float = 0.0
    mean_coverage: float = 0.0
    mode_match_rate: float = 0.0

    def __post_init__(self):
        """Compute aggregate metrics."""
        if self.field_comparisons:
            comparisons = list(self.field_comparisons.values())
            n = len(comparisons)

            self.mean_kl = sum(c.kl_divergence for c in comparisons) / n
            self.mean_tv = sum(c.total_variation for c in comparisons) / n
            self.mean_coverage = sum(c.coverage for c in comparisons) / n
            self.mode_match_rate = sum(1 for c in comparisons if c.mode_match) / n

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "schema_name": self.schema_name,
            "real_samples": self.real_samples,
            "gen_samples": self.gen_samples,
            "mean_kl": self.mean_kl,
            "mean_tv": self.mean_tv,
            "mean_coverage": self.mean_coverage,
            "mode_match_rate": self.mode_match_rate,
            "field_comparisons": {
                k: v.to_dict() for k, v in self.field_comparisons.items()
            },
        }

    def summary_table(self) -> str:
        """Generate a markdown summary table."""
        lines = [
            f"## Distribution Comparison: {self.schema_name}",
            f"",
            f"Real samples: {self.real_samples}, Generated samples: {self.gen_samples}",
            f"",
            f"### Aggregate Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean KL Divergence | {self.mean_kl:.4f} |",
            f"| Mean Total Variation | {self.mean_tv:.4f} |",
            f"| Mean Coverage | {self.mean_coverage:.2%} |",
            f"| Mode Match Rate | {self.mode_match_rate:.2%} |",
            f"",
            f"### Per-Field Results",
            f"| Field | KL | TV | Coverage | Mode Match | Real Mode | Gen Mode |",
            f"|-------|----|----|----------|------------|-----------|----------|",
        ]

        for name, c in sorted(self.field_comparisons.items()):
            match = "Yes" if c.mode_match else "No"
            lines.append(
                f"| {name} | {c.kl_divergence:.4f} | {c.total_variation:.4f} | "
                f"{c.coverage:.2%} | {match} | {c.real_mode} | {c.gen_mode} |"
            )

        return "\n".join(lines)


def compare_extraction_results(
    real_result,  # ExtractionResult
    gen_result,   # ExtractionResult
) -> SchemaComparisonResult:
    """Compare two ExtractionResults (from field_extractors).

    Args:
        real_result: ExtractionResult from real/validation data
        gen_result: ExtractionResult from generated samples

    Returns:
        SchemaComparisonResult with all field comparisons
    """
    field_comparisons = {}

    # Compare each field that exists in both
    all_fields = set(real_result.field_distributions.keys()) | set(gen_result.field_distributions.keys())

    for field_name in all_fields:
        real_dist = real_result.field_distributions.get(field_name)
        gen_dist = gen_result.field_distributions.get(field_name)

        real_counts = real_dist.counts if real_dist else {}
        gen_counts = gen_dist.counts if gen_dist else {}
        value_type = (real_dist.value_type if real_dist else
                      gen_dist.value_type if gen_dist else "unknown")

        # Skip fields with no data
        if not real_counts and not gen_counts:
            continue

        comparison = compare_distributions(
            real_counts=real_counts,
            gen_counts=gen_counts,
            field_name=field_name,
            value_type=value_type,
        )
        field_comparisons[field_name] = comparison

    return SchemaComparisonResult(
        schema_name=real_result.schema_name,
        real_samples=real_result.num_valid,
        gen_samples=gen_result.num_valid,
        field_comparisons=field_comparisons,
    )
