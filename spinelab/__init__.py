"""spinelab — inference-only spine MRI analysis pipeline.

RESEARCH USE ONLY. Nothing in this package is a diagnosis, and no stage here has
been validated as a diagnostic device. Every number that leaves this pipeline
carries an evidence level (see spinelab.evidence) so a reader can tell a
segmentation-model output from a raw intensity heuristic.
"""

__version__ = "0.3.0"

DISCLAIMER_RU = (
    "Исследовательский инструмент. НЕ диагноз и НЕ медицинское заключение. "
    "Все находки требуют проверки врачом-рентгенологом на исходных изображениях."
)
DISCLAIMER_EN = (
    "Research tool. NOT a diagnosis. Every finding must be verified by a "
    "radiologist on the source images."
)
