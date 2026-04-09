from src.utils.geo import haversine_km


def test_zurich_to_jfk():
    # ZRH 47.4647, 8.5492  →  JFK 40.6413, -73.7781
    d = haversine_km(47.4647, 8.5492, 40.6413, -73.7781)
    assert 6200 < d < 6500


def test_zero_distance():
    assert haversine_km(0, 0, 0, 0) == 0
