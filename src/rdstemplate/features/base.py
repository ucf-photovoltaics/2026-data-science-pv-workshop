"""FeatureExtractor ABC and name-string registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

EXTRACTOR_REGISTRY: dict[str, type] = {}


def register_extractor(name: str):
    """Class decorator that registers a FeatureExtractor under *name*."""

    def deco(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls

    return deco


class FeatureExtractor(ABC):
    """Extract features for ONE (sample_id, exposure_step) observation.

    Subclass this and decorate with ``@register_extractor('your_name')`` to make
    the extractor addressable from the YAML config.
    """

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def extract(self, sample_id: str, exposure_step, data) -> dict:
        """Return {feature_name: value} for ONE (sample_id, exposure_step) pair.

        *data* is whatever the corresponding loader returns (DataFrame, ndarray, …).
        Return an empty dict if *data* is None or the step cannot be processed —
        the pipeline will record NaNs for the missing features.
        """
