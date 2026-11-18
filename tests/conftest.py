"""Hypothesis profiles: deterministic by default, ``HYPOTHESIS_PROFILE=fuzz`` for a long random search."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "check", derandomize=True, max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("fuzz", max_examples=5_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "check"))
