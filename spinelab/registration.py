"""Rigid registration between two series of the same session.

Why this exists, given that NIfTI affines already put both series in the same
patient coordinate frame: the structures being measured are 2-4 mm across. A
facet joint is smaller than the inter-series patient motion you get from
breathing and shifting between two acquisitions. Header geometry gives an
excellent starting point and is usually right to within a couple of millimetres —
which at this scale can move the region of interest off the joint entirely.

So: header geometry as the initial transform, then a rigid (6 DOF) refinement
driven by mutual information (the two series have different contrast), then a QC
report of how far it moved. If the correction is large, that is a sign the
registration failed rather than that the patient moved 3 cm, and the caller is
told so instead of getting a silently wrong result.

SimpleITK is imported lazily: it is only needed when this stage runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Above these, a "correction" is far more likely to be a failed optimisation
#: than real inter-series motion.
MAX_PLAUSIBLE_TRANSLATION_MM = 15.0
MAX_PLAUSIBLE_ROTATION_DEG = 10.0


@dataclass
class RegistrationResult:
    applied: bool
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_magnitude_mm: float = 0.0
    rotation_deg: float = 0.0
    metric_before: float | None = None
    metric_after: float | None = None
    reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "translation_mm": [round(float(v), 3) for v in self.translation_mm],
            "translation_magnitude_mm": round(float(self.translation_magnitude_mm), 3),
            "rotation_deg": round(float(self.rotation_deg), 3),
            "metric_before": None if self.metric_before is None else round(float(self.metric_before), 5),
            "metric_after": None if self.metric_after is None else round(float(self.metric_after), 5),
            "improved": (self.metric_after is not None and self.metric_before is not None
                         and self.metric_after < self.metric_before),
            "reason": self.reason,
            "notes": self.notes,
        }


def rigid_register(fixed_path: str, moving_path: str, *, sampling_percentage: float = 0.2,
                   iterations: int = 200):
    """Refine the header alignment of `moving` onto `fixed` with a rigid transform.

    Returns (transform, RegistrationResult). The transform maps points in the
    fixed image's space to the moving image's space, which is what
    `sitk.Resample` wants; use `apply_to_label_volume` to bring a mask defined in
    the moving image's space onto the fixed grid.

    Mattes mutual information is used because the two series have different
    contrast (e.g. T2 against STIR), where intensity-difference metrics fail.
    """
    import SimpleITK as sitk

    fixed = sitk.Cast(sitk.ReadImage(str(fixed_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_path)), sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_percentage, seed=1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    metric_before = reg.MetricEvaluate(fixed, moving)
    try:
        transform = reg.Execute(fixed, moving)
    except RuntimeError as exc:
        return initial, RegistrationResult(
            applied=False, reason=f"optimiser failed: {str(exc)[:200]}",
            metric_before=metric_before,
        )

    metric_after = reg.GetMetricValue()
    euler = sitk.Euler3DTransform(transform)
    tx, ty, tz = euler.GetTranslation()
    magnitude = float((tx ** 2 + ty ** 2 + tz ** 2) ** 0.5)
    rotation = _rotation_magnitude_deg(euler)

    result = RegistrationResult(
        applied=True,
        translation_mm=(tx, ty, tz),
        translation_magnitude_mm=magnitude,
        rotation_deg=rotation,
        metric_before=metric_before,
        metric_after=metric_after,
        notes=[f"stop condition: {reg.GetOptimizerStopConditionDescription()}"],
    )

    if magnitude > MAX_PLAUSIBLE_TRANSLATION_MM or rotation > MAX_PLAUSIBLE_ROTATION_DEG:
        result.applied = False
        result.reason = (
            f"implausible correction ({magnitude:.1f} mm, {rotation:.1f} deg) — treating as a "
            "failed registration and keeping the header alignment"
        )
        return initial, result
    if metric_after is not None and metric_before is not None and metric_after > metric_before:
        result.applied = False
        result.reason = "registration did not improve the metric — keeping the header alignment"
        return initial, result
    return transform, result


def _rotation_magnitude_deg(euler) -> float:
    """Total rotation angle of a 3D Euler transform, in degrees."""
    import math

    import numpy as np

    matrix = np.array(euler.GetMatrix(), dtype=float).reshape(3, 3)
    # Rotation angle from the trace: trace = 1 + 2*cos(theta)
    cos_theta = (float(np.trace(matrix)) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_theta)))))


def apply_to_label_volume(label_path: str, fixed_path: str, transform, out_path: str) -> str:
    """Resample a label volume onto the fixed grid, nearest neighbour.

    Nearest neighbour is not a preference here, it is a requirement: label ids are
    categorical, and any interpolation invents ids that mean other structures.
    """
    import SimpleITK as sitk

    fixed = sitk.ReadImage(str(fixed_path))
    labels = sitk.ReadImage(str(label_path))
    resampled = sitk.Resample(labels, fixed, transform, sitk.sitkNearestNeighbor, 0,
                              labels.GetPixelID())
    sitk.WriteImage(resampled, str(out_path))
    return str(out_path)
