SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48

PLAYER_1_HEIGHT_METRES = 1.88
PLAYER_2_HEIGHT_METRES = 1.91

# Fastest recorded serve is ~263 km/h (Sam Groth, 2012); groundstrokes are far lower.
# Anything above this from peak-velocity estimation is tracking noise, not a real shot.
MAX_REALISTIC_BALL_SPEED_KMH = 260.0

# A groundstroke cannot reach serve speeds, and using one bound for both let a 242 km/h
# "Forehand" through on the reference clip. The fastest serve on record is about 263 km/h
# and the fastest forehand about 193, so a non-serve flight above this is a reconstruction
# artifact rather than a remarkable shot: usually two events joined across a contact the
# detector missed, which makes the flight look shorter in time than it really was.
MAX_REALISTIC_GROUNDSTROKE_KMH = 200.0
