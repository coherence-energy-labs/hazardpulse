"""Verification baselines: what the operational forecast system actually did.

Modules here produce the *comparison arm* for HazardPulse verification -- the
decisions of human forecasters and operational guidance, evaluated on exactly
the same (lat, lon, UTC instant) tuples the HazardPulse models score.

* :mod:`hazardpulse.verification.nws_warnings` -- NWS storm-based tornado
  warnings (VTEC ``TO.W``), every polygon state over each warning's life.
"""
