# MIMII DUE pilot protocol

This repository does not use the official unsupervised DCASE/MIMII DUE protocol.

Reason:
- the current repo uses a supervised binary source model;
- therefore MIMII DUE is used here as a custom supervised cross-domain pilot.

Current split mapping:
- source/train = source normal training clips + part of labeled source-domain test clips
- source/val = held-out part of labeled source-domain test clips
- target/adapt = target normal training clips
- target/test = labeled target-domain test clips

Claim boundary:
- this is a domain-shift evaluation pilot inside the current methodology;
- this is not an official benchmark claim against DCASE leaderboard protocols.