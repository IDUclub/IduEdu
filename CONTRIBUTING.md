# Contributing to IduEdu

This guide covers the day-to-day workflow for humans and coding agents. The
short version: **write [Conventional Commits](https://www.conventionalcommits.org/),
open a PR, squash-merge it into the official `IDUclub/IduEdu` `main` branch, and the release happens
automatically.**
Never bump the version or create a tag by hand.

## Requirements

- Python 3.11 through 3.14
- a stable [Rust toolchain](https://rustup.rs/) (1.82 or newer): the graph kernels are a Rust crate in `rust/`,
  compiled with maturin into `iduedu._rust` when the package is installed from source
- [uv](https://docs.astral.sh/uv/) (package manager; `uv.lock` is generated locally and is not committed)
- `make` (optional but every command below has a `make` shortcut)

## Setup

```bash
uv sync --all-groups     # or: make install-dev; also rebuilds the Rust extension after changes in rust/
```

## Local development loop

| Task                    | Command             | Notes                                             |
|-------------------------|---------------------|---------------------------------------------------|
| Fast tests (no network) | `make test`         | default loop; excludes Overpass tests             |
| Unit tests only         | `make test-unit`    |                                                   |
| Network tests           | `make test-network` | Overpass, or a Geofabrik extract if it is down    |
| All tests               | `make test-all`     | `pytest --run-network`                            |
| Coverage (terminal)     | `make coverage`     | fast subset                                       |
| Coverage for CI         | `make coverage-xml` | **includes network tests**, writes `coverage.xml` |
| Format                  | `make format`       | isort + black                                     |
| Format check            | `make format-check` | what CI enforces                                  |
| Lint                    | `make lint`         | pylint                                            |
| Rust core               | `make rust-check`   | `cargo fmt --check`, clippy, `cargo test` (CI)    |
| Build docs              | `make docs`         | English and Russian → `docs/_build/html`, `/ru`   |
| Update translations     | `make docs-i18n`    | new and changed text into `docs/locale/ru`        |
| Check translations      | `make docs-i18n-check` | what CI enforces on pull requests              |

Line length is 120. Imports are isort-sorted (`__init__.py` files are skipped).

Tests are grouped by behavior: graph data/adjacency/methods, routing,
NetworkX compatibility, OSM builders, Overpass, GTFS, and optional PBF support.
Keep a distinct test for a regression or public contract; avoid repeating the
same search across every combination of unrelated options. Shared graph and
GTFS factories live in `tests/factories.py`. Optional-dependency tests stay
separate so missing an extra cannot skip core graph tests.

The routing and component kernels are compiled Rust (`rust/`, see `rust/README.md`). `cargo test` covers them
on their own, and `tests/test_rust_unit.py` checks the bindings against SciPy; coverage measures the Python layer.
Keep `iduedu/_rust.pyi` in sync when the bindings change. The early pytest plugin in `tests/__init__.py` loads
third-party numerical extensions before coverage tracing starts, including on
Windows; imports and execution of `iduedu` remain measured.
Branch coverage is explicitly enabled in pytest options so subprocesses inherit
the same measurement mode even when their working directory has no project config.

## Documentation in two languages

The documentation is written in English and translated into Russian with Sphinx's gettext tools. Every page,
every text cell of the example notebooks and every docstring that the API reference shows has its translation in
`docs/locale/ru/LC_MESSAGES/<page>.po`; the Russian site is built into `docs/_build/html/ru` and both sites link
to each other from the sidebar. A change of English text is not finished until it is translated:

1. Edit the English source (`.md`, `.rst`, notebook markdown, docstrings).
2. Run `make docs-i18n` (`python docs/i18n.py update`). New messages get an empty `msgstr`; changed ones keep the
   old translation, marked `#, fuzzy`, with the previous English text as `#| msgid` for comparison.
3. Translate the empty entries and fix the fuzzy ones, deleting the `#, fuzzy` line (any `.po` editor works, e.g.
   Poedit, or a text editor). Keep markup, roles, links and literals as in the English text; write decimal commas
   in prose (`0,5 м`) but not in code.
4. Run `make docs-i18n-check` (CI runs it on every pull request) and `make docs`, and look at the changed pages in
   both languages.

Translations of a MyST page (`configuration.md`, `performance.md`) are parsed as Markdown, including the docstrings
that page shows through `autoclass`: write a role there as `` {func}`name` ``, not `` :func:`name` ``, or it renders
as plain text. A parameter description with a nested list appears in a catalogue as a whole and as its parts;
`make docs-i18n` keeps the whole one unchanged so that Sphinx translates the parts, which you translate.

### Pages, figures and notebooks

The pages follow the topics of the sidebar: `docs/building/`, `docs/graph/`, `docs/routing/` and `docs/analysis/`.
A method page has the sections Overview, Inputs, Method, Outputs, Functions, Figures, Limitations, References and
Example notebook; its functions are listed in an `autosummary` with `:toctree: ../api/generated`, so every function
has its own reference page and its docstring is translated once.

The figures are drawn by `docs/figures/make_figures.py` from Vasilyevsky Island (OSM relation 1114252), read from
the Geofabrik extract of the Northwestern Federal District (about 650 MB, downloaded once into `.iduedu_cache`):

```bash
uv run python docs/figures/make_figures.py                  # every figure
uv run python docs/figures/make_figures.py routing_search   # only the named ones
```

The PNG files land in `docs/figures/out/` (not committed). Pages link them from the `assets` branch,
`https://raw.githubusercontent.com/IDUclub/IduEdu/assets/<name>.png`; a maintainer commits new or changed figures
there. The example notebooks are stored with their outputs (`nb_execution_mode = "off"`); run a changed notebook
before committing it.

## Commit messages (Conventional Commits)

Every commit — and, because we **squash-merge**, every **PR title** — must follow
Conventional Commits. The PR title becomes the squash commit on `main`, and that
message is what drives the next version.

```
<type>(<optional scope>): <description>

[optional body]

[optional footer, e.g. BREAKING CHANGE: ...]
```

The squash commit body is copied into the GitHub Release and `CHANGELOG.md`. python-semantic-release joins
consecutive lines of a paragraph into one, so separate every point with a blank line; a wrapped list without
blank lines becomes a single run-on paragraph.

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

In the official `IDUclub/IduEdu` repository, every successful `Tests and Coverage` run for a push to
`main` starts `.github/workflows/release.yml`:

1. runs [python-semantic-release](https://python-semantic-release.readthedocs.io/),
   which computes the next version, updates `iduedu/_version.py` and
   `CHANGELOG.md`, commits `chore(release): vX.Y.Z [skip ci]`, tags `vX.Y.Z`, and
   creates the GitHub Release;
2. publishes the built distributions to PyPI via Trusted Publishing (OIDC);
3. lets `.github/workflows/docs.yml` rebuild and publish the official documentation to `gh-pages`.

Development forks may run tests and documentation previews, but they do not publish releases or deploy
the official documentation.

The version lives in a **single source of truth**, `iduedu/_version.py`
(`VERSION = "x.y.z"`); `pyproject.toml` reads it dynamically through hatchling.
There is no manual `pyproject.toml` version, no `sync_version`/`check_version`
step, and no manual `git tag`. Do not edit `iduedu/_version.py` by hand — the release
automation owns it. `CHANGELOG.md` is updated in place: each release is inserted below
`<!-- version list -->` and earlier entries are left untouched, so a released entry may be rewritten by hand
when the generated text needs it.

## Pull request checklist

- [ ] PR title is a valid Conventional Commit (it becomes the squash commit).
- [ ] `make format-check` and `make test` pass locally.
- [ ] New behavior has tests (`_unit` for fast tests, `_network` for Overpass).
- [ ] Public API changes are reachable via `from iduedu import ...`.
- [ ] Breaking changes use `!` / `BREAKING CHANGE:` and note the migration.
- [ ] Changed English text (pages, notebook text, public docstrings) is translated: `make docs-i18n`, translate,
      `make docs-i18n-check`.
