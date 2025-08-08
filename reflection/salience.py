"""
“Was this moment worth keeping?”

We will combine 4 signals:

    Novelty: is this event different from the last few events?
    Action intensity: did the agent do much (mouse/keys)?
    Affect: strong audio mood/music confidence?
    Eventness: your is_event flag.

This will translate to a formula like:

    salience = 0.35*novelty + 0.25*action + 0.25*affect + 0.15*eventness

"""