# Contributing to IduEdu

This guide covers the day-to-day workflow for humans and coding agents. The
short version: **write [Conventional Commits](https://www.conventionalcommits.org/),
open a PR, squash-merge it into `main`, and the release happens automatically.**
Never bump the version or create a tag by hand.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (package manager; `uv.lock` is generated locally and is not committed)
- `make` (optional but every command below has a `make` shortcut)

## Setup

```bash
uv sync --all-groups     # or: make install-dev
```

## Local development loop

| Task                    | Command             | Notes                                             |
|-------------------------|---------------------|---------------------------------------------------|
| Fast tests (no network) | `make test`         | default loop; excludes Overpass tests             |
| Unit tests only         | `make test-unit`    |                                                   |
| Network tests           | `make test-network` | calls the Overpass API; needs internet            |
| All tests               | `make test-all`     | `pytest --run-network`                            |
| Coverage (terminal)     | `make coverage`     | fast subset                                       |
| Coverage for CI         | `make coverage-xml` | **includes network tests**, writes `coverage.xml` |
| Format                  | `make format`       | isort + black                                     |
| Format check            | `make format-check` | what CI enforces                                  |
| Lint                    | `make lint`         | pylint                                            |
| Build docs              | `make docs`         | Sphinx → `docs/_build/html`                       |

Line length is 120. Imports are isort-sorted (`__init__.py` files are skipped).

## Commit messages (Conventional Commits)

Every commit — and, because we **squash-merge**, every **PR title** — must follow
Conventional Commits. The PR title becomes the squash commit on `main`, and that
message is what drives the next version.

```
<type>(<optional scope>): <description>

[optional body]

[optional footer, e.g. BREAKING CHANGE: ...]
```

### How the version is bumped

The next version is computed from the commit types since the last release tag:

| Commit                                                       | Example                                | Version effect              |
|--------------------------------------------------------------|----------------------------------------|-----------------------------|
| `fix:` / `perf:`                                             | `fix(overpass): retry on 504`          | **patch** (`1.2.1 → 1.2.2`) |
| `feat:`                                                      | `feat(graph): add od_matrix threshold` | **minor** (`1.2.1 → 1.3.0`) |
| `feat!:` or `BREAKING CHANGE:` footer                        | `feat(graph)!: return UrbanGraph`      | **major** (`1.2.1 → 2.0.0`) |
| `docs:` `test:` `chore:` `refactor:` `ci:` `style:` `build:` | `test(graph): expand coverage`         | **no release**              |

If a batch of merged commits contains several types, the **highest** bump wins
(one `feat!` in the batch → a single major bump). A push to `main` that contains
only no-release types produces no release at all.

Preview what would be released without changing anything:

```bash
make version-next     # prints the next version PSR would compute (no-op)
make version          # prints the current version
```

## Releases (automated — do not do this by hand)

On every push to `main`, `.github/workflows/release.yml`:

1. runs a fast (non-network) build/test gate;
2. runs [python-semantic-release](https://python-semantic-release.readthedocs.io/),
   which computes the next version, updates `iduedu/_version.py` and
   `CHANGELOG.md`, commits `chore(release): vX.Y.Z [skip ci]`, tags `vX.Y.Z`, and
   creates the GitHub Release;
3. publishes the built distributions to PyPI via Trusted Publishing (OIDC).

The version lives in a **single source of truth**, `iduedu/_version.py`
(`VERSION = "x.y.z"`); `pyproject.toml` reads it dynamically through hatchling.
There is no manual `pyproject.toml` version, no `sync_version`/`check_version`
step, and no manual `git tag`. Do not edit `iduedu/_version.py` or `CHANGELOG.md`
by hand — the release automation owns them.

## Pull request checklist

- [ ] PR title is a valid Conventional Commit (it becomes the squash commit).
- [ ] `make format-check` and `make test` pass locally.
- [ ] New behavior has tests (`_unit` for fast tests, `_network` for Overpass).
- [ ] Public API changes are reachable via `from iduedu import ...`.
- [ ] Breaking changes use `!` / `BREAKING CHANGE:` and note the migration.
