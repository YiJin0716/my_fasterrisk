"""
Lung-RADS® v2022 Classification Algorithm
==========================================
Reference: ACR Lung-RADS v2022 (November 2022)
https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads

Usage
-----
Instantiate a `Nodule` for each finding on the CT, then call
`classify_exam(nodules, ...)` to get the exam-level Lung-RADS category.

All sizes are in **mm** (mean diameter) unless otherwise noted.

Limitations / Assumptions
--------------------------
* This implementation covers the primary size/morphology decision tree.
  Clinical edge-cases that require radiologist judgement (e.g. infectious
  vs. malignant appearance, specific PET/CT recommendations, risk-model
  calculators) are flagged in the output notes rather than auto-resolved.
* Growth is defined as > 1.5 mm increase in mean diameter within 12 months
  (per Note 6 of the official spec).
* The 'S' modifier (significant incidental finding unrelated to lung cancer)
  must be set by the caller; this algorithm does not detect incidental findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NoduleType(str, Enum):
    SOLID = "solid"
    PART_SOLID = "part_solid"          # also called sub-solid
    NON_SOLID = "non_solid"            # pure ground-glass nodule (GGN)
    JUXTAPLEURAL = "juxtapleural"      # pleural-contact solid nodule
    AIRWAY = "airway"                  # endo-bronchial / airway nodule
    ATYPICAL_CYST = "atypical_cyst"    # thick-walled or multilocular cyst


class AirwayLevel(str, Enum):
    SUBSEGMENTAL = "subsegmental"
    SEGMENTAL_OR_MORE_PROXIMAL = "segmental_or_more_proximal"


class CystType(str, Enum):
    THICK_WALLED = "thick_walled"       # unilocular, wall ≥ 2 mm
    MULTILOCULAR = "multilocular"       # internal septations


class DetectionContext(str, Enum):
    BASELINE = "baseline"               # first exam / first detection
    NEW = "new"                         # new nodule on follow-up
    GROWING = "growing"                 # existing nodule that has grown (> 1.5 mm / 12 mo)
    STABLE = "stable"                   # existing nodule, no significant growth
    DECREASED = "decreased"             # existing nodule, smaller than prior


class LungRadsCategory(str, Enum):
    CAT_0 = "0"
    CAT_1 = "1"
    CAT_2 = "2"
    CAT_3 = "3"
    CAT_4A = "4A"
    CAT_4B = "4B"
    CAT_4X = "4X"


CATEGORY_RANK: dict[LungRadsCategory, int] = {
    LungRadsCategory.CAT_0:  0,
    LungRadsCategory.CAT_1:  1,
    LungRadsCategory.CAT_2:  2,
    LungRadsCategory.CAT_3:  3,
    LungRadsCategory.CAT_4A: 4,
    LungRadsCategory.CAT_4B: 5,
    LungRadsCategory.CAT_4X: 6,
}

CATEGORY_MANAGEMENT: dict[LungRadsCategory, str] = {
    LungRadsCategory.CAT_0:  "Additional imaging needed; 1–3 month LDCT if infectious/inflammatory process suspected",
    LungRadsCategory.CAT_1:  "12-month screening LDCT",
    LungRadsCategory.CAT_2:  "12-month screening LDCT",
    LungRadsCategory.CAT_3:  "6-month LDCT",
    LungRadsCategory.CAT_4A: "3-month LDCT; PET/CT may be considered if solid nodule or solid component ≥ 8 mm",
    LungRadsCategory.CAT_4B: "Diagnostic chest CT ± contrast; PET/CT may be considered; tissue sampling and/or referral for further clinical evaluation",
    LungRadsCategory.CAT_4X: "Management per category (3 or 4) plus evaluation of additional suspicious features",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Nodule:
    """
    Represents a single pulmonary nodule or airway/cystic finding.

    Parameters
    ----------
    nodule_type : NoduleType
        Morphological category of the nodule.
    context : DetectionContext
        Whether this is a baseline finding, new on follow-up, growing, stable,
        or decreased.
    mean_diameter_mm : float
        Mean diameter in mm = (long axis + short axis) / 2.
        Required for all nodule types except pure benign calcified/fat lesions.
    solid_component_mm : float, optional
        Mean diameter of the solid component in mm.
        Required for PART_SOLID nodules.
    has_benign_calcification : bool
        Complete, central, popcorn, or concentric ring calcification → Cat 1.
    is_fat_containing : bool
        Fat density on CT (e.g. hamartoma) → Cat 1.
    airway_level : AirwayLevel, optional
        For AIRWAY nodules: subsegmental vs segmental-or-more-proximal.
    cyst_type : CystType, optional
        For ATYPICAL_CYST nodules.
    cyst_wall_growing : bool
        For thick-walled cysts: wall thickness / nodularity is growing.
    cyst_multilocular_growing : bool
        Growing multilocular cyst (mean diameter increasing).
    cyst_increased_loculation : bool
        Increased loculation or new/increased opacity in multilocular cyst.
    prior_category : str, optional
        Lung-RADS category assigned at the prior exam (e.g. "3", "4A", "4B").
        Used to apply stability down-grade rules.
    has_additional_suspicious_features : bool
        Spiculation, lymphadenopathy, frank metastatic disease, or GGN that
        doubles in size in 1 year → upgrade to 4X.
    notes : list[str]
        Free-text notes added during classification.
    """
    nodule_type: NoduleType
    context: DetectionContext = DetectionContext.BASELINE
    mean_diameter_mm: Optional[float] = None
    solid_component_mm: Optional[float] = None

    # Benign feature overrides
    has_benign_calcification: bool = False
    is_fat_containing: bool = False

    # Airway-nodule specifics
    airway_level: Optional[AirwayLevel] = None

    # Cyst specifics
    cyst_type: Optional[CystType] = None
    cyst_wall_growing: bool = False
    cyst_multilocular_growing: bool = False
    cyst_increased_loculation: bool = False

    # Follow-up context
    prior_category: Optional[str] = None

    # 4X upgrade
    has_additional_suspicious_features: bool = False

    # Output
    notes: list[str] = field(default_factory=list)


@dataclass
class NoduleResult:
    """Classification result for a single nodule."""
    nodule: Nodule
    category: LungRadsCategory
    management: str
    notes: list[str]


@dataclass
class ExamResult:
    """Exam-level Lung-RADS result (highest-category nodule drives the score)."""
    overall_category: LungRadsCategory
    s_modifier: bool
    management: str
    nodule_results: list[NoduleResult]
    notes: list[str]

    def __str__(self) -> str:
        cat = self.overall_category.value
        s = "S" if self.s_modifier else ""
        lines = [
            f"Lung-RADS® v2022: Category {cat}{s}",
            f"Management: {self.management}",
        ]
        if self.notes:
            lines.append("Exam notes:")
            for n in self.notes:
                lines.append(f"  • {n}")
        lines.append("")
        for i, r in enumerate(self.nodule_results, 1):
            lines.append(f"  Nodule {i}: [{r.nodule.nodule_type.value}] → Category {r.category.value}")
            for n in r.notes:
                lines.append(f"    – {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def _classify_nodule(n: Nodule) -> NoduleResult:
    """Classify a single nodule and return its Lung-RADS category."""
    notes: list[str] = list(n.notes)
    cat: LungRadsCategory

    d = n.mean_diameter_mm
    s = n.solid_component_mm
    ctx = n.context

    # ------------------------------------------------------------------
    # Benign-feature overrides → Category 1
    # ------------------------------------------------------------------
    if n.has_benign_calcification:
        notes.append("Complete/central/popcorn/concentric-ring calcification → Category 1")
        cat = LungRadsCategory.CAT_1
        return _apply_4x_modifier(n, cat, notes)

    if n.is_fat_containing:
        notes.append("Fat-containing nodule (e.g. hamartoma) → Category 1")
        cat = LungRadsCategory.CAT_1
        return _apply_4x_modifier(n, cat, notes)

    # ------------------------------------------------------------------
    # Prior-exam stability down-grades (Notes 3a / 4A follow-up rule)
    # ------------------------------------------------------------------
    if n.prior_category == "3" and ctx in (DetectionContext.STABLE, DetectionContext.DECREASED):
        notes.append("Category 3 lesion stable/decreased on 6-month follow-up → down-graded to Category 2")
        cat = LungRadsCategory.CAT_2
        return _apply_4x_modifier(n, cat, notes)

    if n.prior_category == "4A" and ctx in (DetectionContext.STABLE, DetectionContext.DECREASED):
        notes.append("Category 4A lesion stable/decreased on 3-month follow-up → down-graded to Category 3")
        cat = LungRadsCategory.CAT_3
        return _apply_4x_modifier(n, cat, notes)

    if n.prior_category == "4B" and ctx in (DetectionContext.STABLE, DetectionContext.DECREASED):
        notes.append(
            "Category 4B lesion stable/decreased and proven benign after workup → Category 2 (caller must confirm benign work-up)"
        )
        cat = LungRadsCategory.CAT_2
        return _apply_4x_modifier(n, cat, notes)

    # ------------------------------------------------------------------
    # Classify by nodule type
    # ------------------------------------------------------------------
    if n.nodule_type == NoduleType.JUXTAPLEURAL:
        cat = _classify_juxtapleural(d, ctx, notes)

    elif n.nodule_type == NoduleType.SOLID:
        cat = _classify_solid(d, ctx, notes)

    elif n.nodule_type == NoduleType.PART_SOLID:
        if s is None:
            notes.append(
                "WARNING: solid_component_mm not provided for part-solid nodule; "
                "classification may be incomplete."
            )
        cat = _classify_part_solid(d, s, ctx, notes)

    elif n.nodule_type == NoduleType.NON_SOLID:
        cat = _classify_non_solid(d, ctx, notes)

    elif n.nodule_type == NoduleType.AIRWAY:
        cat = _classify_airway(n.airway_level, ctx, notes)

    elif n.nodule_type == NoduleType.ATYPICAL_CYST:
        cat = _classify_atypical_cyst(
            cyst_type=n.cyst_type,
            context=ctx,
            wall_growing=n.cyst_wall_growing,
            multilocular_growing=n.cyst_multilocular_growing,
            increased_loculation=n.cyst_increased_loculation,
            notes=notes,
        )

    else:
        raise ValueError(f"Unknown nodule type: {n.nodule_type}")

    return _apply_4x_modifier(n, cat, notes)


def _apply_4x_modifier(n: Nodule, cat: LungRadsCategory, notes: list[str]) -> NoduleResult:
    """Upgrade to 4X if additional suspicious features are present on a Cat 3/4 nodule."""
    if n.has_additional_suspicious_features:
        rank = CATEGORY_RANK[cat]
        if rank >= CATEGORY_RANK[LungRadsCategory.CAT_3]:
            notes.append(
                "Additional suspicious features present (spiculation, lymphadenopathy, "
                "metastatic disease, or GGN doubling in 1 year) → upgraded to Category 4X"
            )
            cat = LungRadsCategory.CAT_4X
        else:
            notes.append(
                "Additional suspicious features noted but nodule is Category 1 or 2; "
                "4X upgrade applies only to Category 3 or 4 nodules."
            )
    return NoduleResult(
        nodule=n,
        category=cat,
        management=CATEGORY_MANAGEMENT[cat],
        notes=notes,
    )


# ------------------------------------------------------------------
# Per-type helpers
# ------------------------------------------------------------------

def _classify_juxtapleural(d: Optional[float], ctx: DetectionContext, notes: list[str]) -> LungRadsCategory:
    """
    Juxtapleural solid nodule: smooth margins; oval, lentiform, or triangular.
    < 10 mm at baseline or new → Category 2.
    Otherwise treat as solid nodule (caller should switch nodule_type to SOLID).
    """
    if d is None:
        notes.append("WARNING: mean_diameter_mm not provided for juxtapleural nodule.")
        return LungRadsCategory.CAT_2

    if ctx in (DetectionContext.BASELINE, DetectionContext.NEW) and d < 10.0:
        notes.append(f"Juxtapleural nodule {d} mm (< 10 mm), baseline/new → Category 2")
        return LungRadsCategory.CAT_2

    notes.append(
        f"Juxtapleural nodule {d} mm does not meet Category 2 criteria; "
        "classifying as solid nodule."
    )
    return _classify_solid(d, ctx, notes)


def _classify_solid(d: Optional[float], ctx: DetectionContext, notes: list[str]) -> LungRadsCategory:
    if d is None:
        notes.append("WARNING: mean_diameter_mm not provided for solid nodule.")
        return LungRadsCategory.CAT_2

    # --- Category 2 ---
    if ctx == DetectionContext.BASELINE and d < 6.0:
        notes.append(f"Solid nodule {d} mm < 6 mm at baseline → Category 2")
        return LungRadsCategory.CAT_2

    if ctx == DetectionContext.NEW and d < 4.0:
        notes.append(f"New solid nodule {d} mm < 4 mm → Category 2")
        return LungRadsCategory.CAT_2

    # --- Category 3 ---
    if ctx == DetectionContext.BASELINE and 6.0 <= d < 8.0:
        notes.append(f"Solid nodule {d} mm (6–<8 mm) at baseline → Category 3")
        return LungRadsCategory.CAT_3

    if ctx == DetectionContext.NEW and 4.0 <= d < 6.0:
        notes.append(f"New solid nodule {d} mm (4–<6 mm) → Category 3")
        return LungRadsCategory.CAT_3

    # --- Category 4A ---
    if ctx == DetectionContext.BASELINE and 8.0 <= d < 15.0:
        notes.append(f"Solid nodule {d} mm (8–<15 mm) at baseline → Category 4A")
        return LungRadsCategory.CAT_4A

    if ctx == DetectionContext.GROWING and d < 8.0:
        notes.append(f"Growing solid nodule {d} mm (< 8 mm) → Category 4A")
        return LungRadsCategory.CAT_4A

    if ctx == DetectionContext.NEW and 6.0 <= d < 8.0:
        notes.append(f"New solid nodule {d} mm (6–<8 mm) → Category 4A")
        return LungRadsCategory.CAT_4A

    # --- Category 4B ---
    if ctx == DetectionContext.BASELINE and d >= 15.0:
        notes.append(f"Solid nodule {d} mm (≥ 15 mm) at baseline → Category 4B")
        return LungRadsCategory.CAT_4B

    if ctx in (DetectionContext.NEW, DetectionContext.GROWING) and d >= 8.0:
        notes.append(f"New or growing solid nodule {d} mm (≥ 8 mm) → Category 4B")
        return LungRadsCategory.CAT_4B

    # Stable / decreased solid nodule without a prior_category context
    notes.append(f"Solid nodule {d} mm, {ctx.value}: applying size-based baseline criteria.")
    return _classify_solid(d, DetectionContext.BASELINE, notes)


def _classify_part_solid(
    d: Optional[float],
    s: Optional[float],
    ctx: DetectionContext,
    notes: list[str],
) -> LungRadsCategory:
    if d is None:
        notes.append("WARNING: mean_diameter_mm not provided for part-solid nodule.")
        return LungRadsCategory.CAT_3

    s_val = s if s is not None else 0.0

    # --- Category 2 ---
    if ctx == DetectionContext.BASELINE and d < 6.0:
        notes.append(f"Part-solid nodule total {d} mm < 6 mm at baseline → Category 2")
        return LungRadsCategory.CAT_2

    # --- Category 3 ---
    if ctx == DetectionContext.BASELINE and d >= 6.0 and s_val < 6.0:
        notes.append(
            f"Part-solid nodule: total {d} mm (≥ 6 mm), solid component {s_val} mm (< 6 mm) at baseline → Category 3"
        )
        return LungRadsCategory.CAT_3

    if ctx == DetectionContext.NEW and d < 6.0:
        notes.append(f"New part-solid nodule total {d} mm < 6 mm → Category 3")
        return LungRadsCategory.CAT_3

    # --- Category 4A ---
    if ctx == DetectionContext.BASELINE and d >= 6.0 and 6.0 <= s_val < 8.0:
        notes.append(
            f"Part-solid nodule: total {d} mm (≥ 6 mm), solid component {s_val} mm (6–<8 mm) at baseline → Category 4A"
        )
        return LungRadsCategory.CAT_4A

    # New or growing with solid component < 4 mm → 4A
    if ctx in (DetectionContext.NEW, DetectionContext.GROWING) and s_val < 4.0:
        notes.append(
            f"New or growing part-solid nodule with solid component {s_val} mm < 4 mm → Category 4A"
        )
        return LungRadsCategory.CAT_4A

    # --- Category 4B ---
    if ctx == DetectionContext.BASELINE and s_val >= 8.0:
        notes.append(
            f"Part-solid nodule solid component {s_val} mm (≥ 8 mm) at baseline → Category 4B"
        )
        return LungRadsCategory.CAT_4B

    if ctx in (DetectionContext.NEW, DetectionContext.GROWING) and s_val >= 4.0:
        notes.append(
            f"New or growing part-solid nodule with solid component {s_val} mm (≥ 4 mm) → Category 4B"
        )
        return LungRadsCategory.CAT_4B

    notes.append(
        f"Part-solid nodule {d} mm / solid component {s_val} mm, {ctx.value}: "
        "defaulting to size-based baseline criteria."
    )
    return _classify_part_solid(d, s, DetectionContext.BASELINE, notes)


def _classify_non_solid(d: Optional[float], ctx: DetectionContext, notes: list[str]) -> LungRadsCategory:
    """Pure ground-glass nodule (GGN)."""
    if d is None:
        notes.append("WARNING: mean_diameter_mm not provided for non-solid (GGN) nodule.")
        return LungRadsCategory.CAT_2

    # --- Category 2 ---
    if ctx in (DetectionContext.BASELINE, DetectionContext.NEW, DetectionContext.GROWING) and d < 30.0:
        notes.append(f"Non-solid GGN {d} mm < 30 mm (baseline/new/growing) → Category 2")
        return LungRadsCategory.CAT_2

    if ctx in (DetectionContext.STABLE, DetectionContext.DECREASED) and d >= 30.0:
        notes.append(f"Non-solid GGN {d} mm ≥ 30 mm but stable/slowly growing → Category 2 (Note 7)")
        return LungRadsCategory.CAT_2

    # --- Category 3 ---
    if ctx in (DetectionContext.BASELINE, DetectionContext.NEW) and d >= 30.0:
        notes.append(f"Non-solid GGN {d} mm ≥ 30 mm at baseline or new → Category 3")
        return LungRadsCategory.CAT_3

    # Growing ≥ 30 mm GGN
    notes.append(f"Non-solid GGN {d} mm ≥ 30 mm growing → Category 3")
    return LungRadsCategory.CAT_3


def _classify_airway(
    level: Optional[AirwayLevel],
    ctx: DetectionContext,
    notes: list[str],
) -> LungRadsCategory:
    if level is None:
        notes.append("WARNING: airway_level not specified; defaulting to subsegmental.")
        level = AirwayLevel.SUBSEGMENTAL

    if level == AirwayLevel.SUBSEGMENTAL:
        notes.append("Subsegmental airway nodule (baseline/new/stable) → Category 2")
        return LungRadsCategory.CAT_2

    # Segmental or more proximal
    if ctx == DetectionContext.BASELINE:
        notes.append("Segmental/proximal airway nodule at baseline → Category 4A")
        return LungRadsCategory.CAT_4A

    if ctx in (DetectionContext.STABLE, DetectionContext.GROWING):
        notes.append(
            "Segmental/proximal airway nodule stable or growing on follow-up → Category 4B; "
            "clinical evaluation (typically bronchoscopy) recommended"
        )
        return LungRadsCategory.CAT_4B

    # New segmental/proximal airway nodule – treat as baseline
    notes.append("New segmental/proximal airway nodule → Category 4A")
    return LungRadsCategory.CAT_4A


def _classify_atypical_cyst(
    cyst_type: Optional[CystType],
    context: DetectionContext,
    wall_growing: bool,
    multilocular_growing: bool,
    increased_loculation: bool,
    notes: list[str],
) -> LungRadsCategory:
    """
    Note: Thin-walled unilocular cysts (wall < 2 mm) are Category 1 / benign.
    Only thick-walled or multilocular cysts reach this function.
    """
    if cyst_type is None:
        notes.append("WARNING: cyst_type not specified for atypical cyst; defaulting to thick-walled.")
        cyst_type = CystType.THICK_WALLED

    # --- Category 4B escalations ---
    if cyst_type == CystType.THICK_WALLED and wall_growing:
        notes.append("Thick-walled cyst with growing wall thickness/nodularity → Category 4B")
        return LungRadsCategory.CAT_4B

    if cyst_type == CystType.MULTILOCULAR and multilocular_growing:
        notes.append("Growing multilocular cyst (mean diameter) → Category 4B")
        return LungRadsCategory.CAT_4B

    if cyst_type == CystType.MULTILOCULAR and increased_loculation:
        notes.append(
            "Multilocular cyst with increased loculation or new/increased opacity → Category 4B"
        )
        return LungRadsCategory.CAT_4B

    # --- Category 3 ---
    if cyst_type == CystType.THICK_WALLED and context == DetectionContext.GROWING:
        notes.append("Thick-walled cyst with growing cystic component (mean diameter) → Category 3")
        return LungRadsCategory.CAT_3

    if cyst_type == CystType.THICK_WALLED:
        # Baseline or new thick-walled cyst
        notes.append("Thick-walled cyst (baseline) → Category 4A")
        return LungRadsCategory.CAT_4A

    if cyst_type == CystType.MULTILOCULAR:
        if context == DetectionContext.BASELINE:
            notes.append("Multilocular cyst at baseline → Category 4A")
            return LungRadsCategory.CAT_4A
        if context == DetectionContext.NEW:
            notes.append("Thin- or thick-walled cyst that becomes multilocular → Category 4A")
            return LungRadsCategory.CAT_4A

    notes.append("Atypical cyst: no escalation criteria met → Category 3")
    return LungRadsCategory.CAT_3


# ---------------------------------------------------------------------------
# Exam-level classification
# ---------------------------------------------------------------------------

def classify_exam(
    nodules: list[Nodule],
    s_modifier: bool = False,
    prior_exams_pending: bool = False,
    suspected_infectious_inflammatory: bool = False,
) -> ExamResult:
    """
    Classify an entire CT exam per Lung-RADS v2022.

    The overall category is driven by the highest-suspicion nodule (Note 1).
    Category 0 is automatically assigned when prior exams are pending or
    an infectious/inflammatory process is suspected.

    Parameters
    ----------
    nodules : list[Nodule]
        All lung nodules / findings on the exam.  Pass an empty list for a
        completely negative scan.
    s_modifier : bool
        Set True if there is a clinically significant incidental finding
        unrelated to lung cancer.
    prior_exams_pending : bool
        Set True if a prior chest CT is being located for comparison → Cat 0.
    suspected_infectious_inflammatory : bool
        Set True if findings are suggestive of an infectious/inflammatory
        process → Cat 0, recommend 1–3 month LDCT.

    Returns
    -------
    ExamResult
    """
    exam_notes: list[str] = []

    # Category 0 overrides
    if prior_exams_pending:
        exam_notes.append("Prior chest CT being located for comparison → Category 0 (temporary)")
        return ExamResult(
            overall_category=LungRadsCategory.CAT_0,
            s_modifier=s_modifier,
            management=CATEGORY_MANAGEMENT[LungRadsCategory.CAT_0],
            nodule_results=[],
            notes=exam_notes,
        )

    if suspected_infectious_inflammatory:
        exam_notes.append(
            "Findings suggestive of infectious/inflammatory process → Category 0; "
            "recommend 1–3 month LDCT follow-up"
        )
        return ExamResult(
            overall_category=LungRadsCategory.CAT_0,
            s_modifier=s_modifier,
            management=CATEGORY_MANAGEMENT[LungRadsCategory.CAT_0],
            nodule_results=[],
            notes=exam_notes,
        )

    # No nodules → Category 1
    if not nodules:
        exam_notes.append("No lung nodules identified → Category 1")
        return ExamResult(
            overall_category=LungRadsCategory.CAT_1,
            s_modifier=s_modifier,
            management=CATEGORY_MANAGEMENT[LungRadsCategory.CAT_1],
            nodule_results=[],
            notes=exam_notes,
        )

    # Classify each nodule individually
    results: list[NoduleResult] = [_classify_nodule(n) for n in nodules]

    # Overall category = highest-ranked individual nodule (Note 1)
    best: NoduleResult = max(results, key=lambda r: CATEGORY_RANK[r.category])
    overall = best.category

    return ExamResult(
        overall_category=overall,
        s_modifier=s_modifier,
        management=CATEGORY_MANAGEMENT[overall],
        nodule_results=results,
        notes=exam_notes,
    )


# ---------------------------------------------------------------------------
# Convenience helper: volume ↔ diameter conversion
# ---------------------------------------------------------------------------

def diameter_to_volume_mm3(mean_diameter_mm: float) -> float:
    """Approximate sphere volume from mean diameter (mm³)."""
    import math
    r = mean_diameter_mm / 2.0
    return (4 / 3) * math.pi * r ** 3


def volume_to_diameter_mm(volume_mm3: float) -> float:
    """Approximate mean diameter from sphere volume (mm)."""
    import math
    return 2.0 * (volume_mm3 * 3 / (4 * math.pi)) ** (1 / 3)


def is_growing(prior_mm: float, current_mm: float) -> bool:
    """
    Returns True if nodule meets the Lung-RADS growth definition:
    > 1.5 mm increase in mean diameter within a 12-month interval (Note 6).
    """
    return (current_mm - prior_mm) > 1.5


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        # --- Negative exam ---
        ("Negative exam (no nodules)", classify_exam([])),

        # --- Cat 1: calcified ---
        ("Calcified nodule", classify_exam([
            Nodule(
                nodule_type=NoduleType.SOLID,
                mean_diameter_mm=8.0,
                has_benign_calcification=True,
            )
        ])),

        # --- Cat 2: small solid baseline ---
        ("Small solid 4 mm baseline", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=4.0,
                   context=DetectionContext.BASELINE)
        ])),

        # --- Cat 3: solid 7 mm baseline ---
        ("Solid 7 mm baseline", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=7.0,
                   context=DetectionContext.BASELINE)
        ])),

        # --- Cat 4A: solid 10 mm baseline ---
        ("Solid 10 mm baseline", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=10.0,
                   context=DetectionContext.BASELINE)
        ])),

        # --- Cat 4B: solid 18 mm baseline ---
        ("Solid 18 mm baseline", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=18.0,
                   context=DetectionContext.BASELINE)
        ])),

        # --- Cat 4B with 4X upgrade ---
        ("Solid 18 mm + spiculation (4X)", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=18.0,
                   context=DetectionContext.BASELINE,
                   has_additional_suspicious_features=True)
        ])),

        # --- Part solid ---
        ("Part-solid 8 mm total / 7 mm solid (4A)", classify_exam([
            Nodule(nodule_type=NoduleType.PART_SOLID, mean_diameter_mm=8.0,
                   solid_component_mm=7.0, context=DetectionContext.BASELINE)
        ])),

        # --- GGN ≥ 30 mm new ---
        ("GGN 35 mm new (Cat 3)", classify_exam([
            Nodule(nodule_type=NoduleType.NON_SOLID, mean_diameter_mm=35.0,
                   context=DetectionContext.NEW)
        ])),

        # --- Airway: segmental, baseline ---
        ("Segmental airway nodule baseline (4A)", classify_exam([
            Nodule(nodule_type=NoduleType.AIRWAY,
                   airway_level=AirwayLevel.SEGMENTAL_OR_MORE_PROXIMAL,
                   context=DetectionContext.BASELINE)
        ])),

        # --- Atypical cyst: thick-walled, growing wall ---
        ("Thick-walled cyst growing wall (4B)", classify_exam([
            Nodule(nodule_type=NoduleType.ATYPICAL_CYST,
                   cyst_type=CystType.THICK_WALLED,
                   cyst_wall_growing=True,
                   context=DetectionContext.GROWING)
        ])),

        # --- Prior Cat 3, now stable (down-grade to 2) ---
        ("Prior Cat 3 stable → Cat 2", classify_exam([
            Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=7.0,
                   context=DetectionContext.STABLE, prior_category="3")
        ])),

        # --- S-modifier ---
        ("Cat 2 + S modifier", classify_exam(
            [Nodule(nodule_type=NoduleType.SOLID, mean_diameter_mm=4.0,
                    context=DetectionContext.BASELINE)],
            s_modifier=True
        )),
    ]

    for title, result in examples:
        print(f"{'='*60}")
        print(f"EXAMPLE: {title}")
        print(result)
        print()
