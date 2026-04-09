# agents.md — Blog Repository Guide

## Project Overview

This is the personal blog of **Jonny Law**, a Senior AI Research Engineer at Neo4j with a PhD in Bayesian Statistics. The blog is titled *"Bayesian Statistics and Functional Programming"* and is published at [https://jonnylaw.rocks](https://jonnylaw.rocks).

It is built with [Quarto](https://quarto.org), a scientific and technical publishing system. Posts cover Bayesian inference, functional programming (Scala), R data science, Python deep learning, and cloud/infrastructure engineering. The rendered site is deployed to GitHub Pages.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Site framework | [Quarto](https://quarto.org) |
| R dependency management | [renv](https://rstudio.github.io/renv/) |
| Python dependency management | [uv](https://docs.astral.sh/uv/) (per-post virtual environments) |
| Containerised build | Docker (`rocker/rstudio` base image) |
| CI / Deployment | GitHub Actions → GitHub Pages |
| Theme | Quarto `litera` HTML theme |

---

## Repository Structure

```
.
├── _quarto.yml                  # Site-level Quarto configuration
├── index.qmd                    # Homepage (post listing)
├── about.qmd                    # About page
├── styles.css                   # Custom CSS overrides
├── build.sh                     # Docker-based build script
├── Dockerfile                   # Build environment definition
├── renv.lock                    # R package lock file
├── renv/                        # renv bootstrap files
├── .Rprofile                    # R startup (activates renv)
├── .github/workflows/main.yml   # CI pipeline (deploy to GitHub Pages)
├── posts/                       # One subdirectory per blog post
│   ├── _metadata.yml            # Shared post metadata defaults
│   └── <date>-<slug>/
│       ├── index.qmd            # Post content (R/Python/Markdown)
│       ├── pyproject.toml       # (Python posts) uv project file
│       ├── uv.lock              # (Python posts) uv lock file
│       └── .venv/               # (Python posts) virtual environment
├── notebooks/                   # Standalone exploration notebooks & scripts
│   └── data/                    # Raw data files used by notebooks
└── _site/                       # Rendered output (git-ignored, uploaded to Pages)
```

---

## Building & Previewing

### Full build via Docker (recommended for reproducibility)

```bash
./build.sh
```

This builds the Docker image (installs all R and Python dependencies) and renders the entire site into `_site/`.

### Local preview (requires Quarto, R, and uv installed)

```bash
# Render everything
quarto render

# Serve locally
python3 -m http.server 8000 --directory _site
# Then open http://localhost:8000
```

### Live preview (may have Python env issues)

```bash
quarto preview
```

---

## Adding a New Post

1. **Create the post directory** following the naming convention:

   ```
   posts/YYYY-MM-DD-post-slug/
   ```

2. **Create `index.qmd`** with front matter:

   ```yaml
   ---
   title: "Your Post Title"
   description: "A short description shown in the listing."
   date: "YYYY-MM-DD"
   categories:
     - Bayesian
     - R          # or Python, Scala, Statistics, etc.
   ---
   ```

3. **Set up language dependencies** (see sections below).

4. Quarto will automatically pick up the new post in the homepage listing (sorted by date descending).

---

## Managing Python Dependencies

Each Python post has its own isolated `uv` environment in its post directory.

```
posts/<date>-<slug>/
├── index.qmd
├── pyproject.toml
├── uv.lock
└── .venv/
```

**Create a new Python post environment:**

```bash
cd posts/<date>-<slug>
uv init            # creates pyproject.toml
uv venv            # creates .venv
uv add numpy pandas scipy  # add dependencies
```

Quarto automatically detects and uses the `.venv` in the same directory as the `.qmd` file when executing Python chunks.

**Interactive development in Positron / VS Code:**

1. Open Command Palette (`Cmd+Shift+P`)
2. Select *"Python: Select Interpreter"*
3. Choose `posts/<slug>/.venv/bin/python`
4. Open the `.qmd` file and run cells with `Cmd+Enter`

---

## Managing R Dependencies

R packages are managed globally for the project using `renv`.

```bash
# Restore all packages from renv.lock
Rscript -e 'renv::restore()'

# After installing a new package, snapshot the lock file
Rscript -e 'renv::snapshot()'
```

The `.Rprofile` in the project root automatically activates `renv` when R starts.

---

## Deployment

The site is deployed to **GitHub Pages** via the GitHub Actions workflow at `.github/workflows/main.yml`.

- **Trigger:** push to `main` (or `fix-rendering`) branch
- **Process:** uploads the pre-rendered `_site/` directory as a Pages artifact, then deploys it
- **Note:** The site must be rendered *before* pushing; the CI workflow does **not** run `quarto render` — it only uploads and deploys the already-rendered `_site/` directory. Render locally (or via `build.sh`) and commit `_site/` before pushing.

---

## Post Categories & Themes

The blog covers these main topic areas:

| Category | Description |
|---|---|
| **Bayesian** | Bayesian inference, MCMC (Metropolis, Gibbs, HMC), state space models, Stan |
| **R** | Data wrangling, visualisation (`tidyverse`), statistical modelling in R |
| **Scala** | Functional programming, Akka Streams, Akka HTTP, Breeze |
| **Python** | NumPy/SciPy, deep learning (PyTorch / numpyro), entity embeddings |
| **DeepLearning** | Neural networks, MC Dropout uncertainty, entity embeddings |
| **Statistics** | Frequentist methods, A/B testing, survival analysis, model comparison |
| **tidytuesday** | Weekly R data visualisation challenges |
| **MachineLearning** | Hierarchical models, model selection |
| **Infrastructure** | Terraform, GitOps, cloud IaC |

---

## Configuration

**`_quarto.yml`** — top-level site settings:

```yaml
project:
  type: website

website:
  title: "Bayesian Inference and Functional Programming"
  navbar:
    right:
      - about.qmd
      - icon: github
        href: https://github.com/jonnylaw

format:
  html:
    theme: litera
    css: styles.css

site-url: https://jonnylaw.rocks
```

**`posts/_metadata.yml`** — default front-matter applied to all posts (e.g., freeze settings).

---

## Key Files at a Glance

| File | Purpose |
|---|---|
| `_quarto.yml` | Site configuration |
| `index.qmd` | Homepage post listing |
| `about.qmd` | Author bio page |
| `build.sh` | Docker-based full build |
| `Dockerfile` | Build image (R + uv + renv) |
| `renv.lock` | R package versions |
| `.github/workflows/main.yml` | CI/CD to GitHub Pages |
| `styles.css` | Custom CSS |
