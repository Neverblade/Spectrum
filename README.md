# Spectrum

5/6 - Added more tests to test noises and conflicts.

5/5 - It's been somewhat tested - look at sanity_tests.py to see what it succeeds at doing right now. I've also added in a set_constants() function in spectrum.py that allows you to change the preset constants of the game. Call this BEFORE you call reset(). Lastly I've brought an ActionSpace implementation so you can use that to sample randomly and check that actions are legal.

Still needed: more sanity checks, particularly for noise and collision checking. More helper functions that make life easier. render() implementation.
