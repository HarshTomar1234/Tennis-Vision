# Cricket field dimensions (in meters)
PITCH_LENGTH = 20.12  # 22 yards
PITCH_WIDTH = 3.05    # 10 feet
CREASE_LENGTH = 2.64  # 8 feet 8 inches
CREASE_WIDTH = 0.05   # 2 inches

# Boundary dimensions (approximate for different grounds)
BOUNDARY_RADIUS_MIN = 55  # Minimum boundary distance
BOUNDARY_RADIUS_MAX = 90  # Maximum boundary distance

# Wicket dimensions
WICKET_HEIGHT = 0.71  # 28 inches
WICKET_WIDTH = 0.23   # 9 inches
BAIL_LENGTH = 0.11    # 4.31 inches

# Player positions
STRIKER_END = "striker"
NON_STRIKER_END = "non_striker"
BOWLER_END = "bowler"

# Cricket shot zones
SHOT_ZONES = {
    'STRAIGHT': 0,
    'OFF_SIDE': 1,
    'LEG_SIDE': 2,
    'BEHIND_WICKET': 3
}

# Ball types
BALL_TYPES = {
    'FAST': 'Fast',
    'MEDIUM': 'Medium',
    'SPIN': 'Spin',
    'SLOWER': 'Slower'
}

# Shot types
SHOT_TYPES = {
    'DEFENSIVE': 'Defensive',
    'DRIVE': 'Drive',
    'CUT': 'Cut',
    'PULL': 'Pull',
    'SWEEP': 'Sweep',
    'HOOK': 'Hook',
    'FLICK': 'Flick',
    'LOFT': 'Loft',
    'SIX': 'Six',
    'FOUR': 'Four'
}

# Player heights for reference (in meters)
BATSMAN_HEIGHT_METRES = 1.75
BOWLER_HEIGHT_METRES = 1.80
WICKET_KEEPER_HEIGHT_METRES = 1.70
FIELDER_HEIGHT_METRES = 1.75