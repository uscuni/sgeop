# Contributing Guidelines

## Procedure

1. *After* opening an issue and discussing with the development team, create a PR with the proposed changes. 
2. If [testing fails](https://github.com/uscuni/sgeop/actions/runs/11368511561) due to an update in the code base:
3. Observed data is [saved as artifacts](https://github.com/uscuni/sgeop/actions/runs/11368511561#artifacts) from the workflow and can be download locally.
4. We determine the `ci_artifacts-ubuntu-latest-py312_sgeop-latest` data as the "truth."
5. After comparison of the current "known" data with new data from (3.), if new data is "truthier," update your PR with the new "known" data.

## Handling Edge Cases in Testing

Edge cases will crop up in full-scale FUA testing that we can ignore (following a thorough investigation – e.g. [`sgeop#77`](https://github.com/uscuni/sgeop/issues/77)) during testing. Once it is determined the geometry in question is not caused by a bug on our end, it can be added to the `KNOWN_BAD_GEOMS` collection in `tests/conftest.py`. This collection is a dictionary keyed by `<NAME>_CODE` of the city/FUA where the values are lists of index locations of simplified edges that can be ignored if they fail equality testing. As an example, see our initial "bad" geometries [here](https://github.com/uscuni/sgeop/blob/1be6b44b1a06d52453ecbaee205ae649101c4ea4/sgeop/tests/conftest.py#L25-L39), which were due to a variant number of coordinates in those resultant simplified edges created by [different versions of `shapely`](https://github.com/uscuni/sgeop/pull/67#issuecomment-2457333724).

## Code Structure

Code should be linted and formatted via `ruff`. With the [`.pre-commit` hooks](https://github.com/uscuni/sgeop/blob/main/.pre-commit-config.yaml) we have code in commits will be formatted and linted automatically once [`pre-commit` is installed](https://pre-commit.com/#installation).
