# MIMII DUE pilot protocol

This repository does not use the official unsupervised DCASE / MIMII DUE protocol.

## Why

The current repository uses a supervised binary source model and treats MIMII DUE as a real-data cross-domain pilot inside the broader methodology.

Therefore, the MIMII DUE stage should be interpreted as:

- a custom supervised cross-domain pilot,
- not an official DCASE leaderboard-style benchmark,
- and not a direct comparison against published unsupervised challenge systems.

## Current split mapping

- `source/train` = source normal training clips plus the selected labeled source-domain training portion used by the current supervised setup
- `source/val` = held-out labeled source-domain validation portion
- `target/adapt` = target normal training clips
- `target/test` = labeled target-domain test clips

## Current role in the repository

The role of this pilot is:

- to provide a non-saturated real target split,
- to compare `raw_only`, `tfr_only`, and `fused` on real data,
- and to test whether optional SFDA provides a stable gain under this shift.

## Current claim boundary

The current MIMII DUE pilot supports the following cautious interpretation:

- representation choice matters on real data;
- no universal representation winner should be claimed from the current pilot;
- optional SFDA is not universally beneficial in the current setup.

The current MIMII DUE pilot does **not** support:

- an official DCASE benchmark claim,
- a claim of a new anomalous-sound adaptation algorithm,
- or a universal claim that adaptation improves real-data transfer.