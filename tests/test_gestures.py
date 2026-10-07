"""Lightweight smoke tests for project imports and metadata.

Camera/MediaPipe integration is intentionally not exercised in CI because it
requires a physical webcam.
"""

from src.gestures import NUMBER_INFO

def test_numbers_1_to_10():
    assert list(NUMBER_INFO) == list(range(1, 11))

def test_number_metadata():
    assert NUMBER_INFO[6][:2] == ("六", "liù")
    assert NUMBER_INFO[9][:2] == ("九", "jiǔ")
    assert NUMBER_INFO[10][:2] == ("十", "shí")
