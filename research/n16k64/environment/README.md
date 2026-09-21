# Frozen Python environments

`main.lock.txt` is the common lock used by the primary, PPL-extension,
mechanism, follow-up, and boundary/corruption campaigns. Its SHA-256 is
`030ac0bb23566ee08904f45cfa5f9d824890b80a6ab22e9fbd01ab934f39f77e`.

`historical.lock.txt` is the separate historical-comparison environment from
the primary campaign. Its SHA-256 is
`1ef2ff426e14e1748bbe97bf58e704dd5248211c39a506462652428bf04c0945`.

These files record Python dependencies. Container and driver provenance remains
in the campaign reports; the locks alone do not recreate GPU hardware, model
weights, datasets, or immutable maps.
