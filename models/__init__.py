"""
Marker so setuptools treats models/ as a package and ships the small trained weights
inside the wheel.

Only hit_bounce_classifier.json travels this way. It is a few hundred bytes of logistic
regression coefficients with no other distribution channel, and the pipeline silently
degrades to a weaker heuristic without it. The large weights (TrackNet, the court model,
MediaPipe pose) are fetched at runtime by scripts/download_models.py into the working
directory, because they are tens to hundreds of megabytes and two of them are not ours
to redistribute.
"""
