# Follow-up diagnosis, declared after weight panel 331682

The approximate solver improves 16/36 crops versus 33/36 for the reference,
despite maximum crop spectral estimation error below 9e-7 relative. Every full
matrix run stopped after a rejected proposal. This motivates a diagnostic of
direction changes, not a new configuration election.

On the same 36 crops, at their all-E2M1 baseline, compute the exact top right
singular vector in float64. Evaluate all eight individual toggles with exact
SVD. Record each toggle's change along the original worst direction, actual
spectral change, and overlap between old and new worst directions. Count cases
where the best directional proposal is harmful in spectral error, and whether
another individually useful toggle exists. Reconstruct candidates from the
same full matrices and verify hashes against the original run.

No existing map, objective, solver parameter, or result changes. These are
post-panel mechanism diagnostics, not independent held-out validation. Small
crop results do not prove the same mechanism dominates full-matrix behavior.
