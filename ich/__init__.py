"""Traumatic intracranial haemorrhage segmentation on head CT (PhysioNet CT-ICH + BHSD).

Every stage writes its outputs to disk and skips work that is already done, so a
disconnected Colab session loses at most one epoch.
"""
__version__ = "3.0.0"
