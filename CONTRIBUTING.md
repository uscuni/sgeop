# Contributing Guidelines

## Procedure

1. *After* opening an issue and discussing with the development team, create a PR with the proposed changes. 
2. If [testing fails](https://github.com/uscuni/sgeop/actions/runs/11368511561) due to an update in the code base:
3. Observed data is [saved as artifacts](https://github.com/uscuni/sgeop/actions/runs/11368511561#artifacts) from the workflow and can be download locally.
4. We determine the `ci_artifacts-ubuntu-latest-py312_sgeop-latest` data as the "truth."
5. After comparison of the current "known" data with new data from (3.), if new data is "truthier," update your PR with the new "known" data.

## Code Structure

Code should be linted and formatted via `ruff`. With the [`.pre-commit` hooks](https://github.com/uscuni/sgeop/blob/main/.pre-commit-config.yaml) we have code in commits will be formatted and linted automatically once [`pre-commit` is installed](https://pre-commit.com/#installation).
