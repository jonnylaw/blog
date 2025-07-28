# Blog

Jonny Law's Weblog. Rare updated.

## Building

This blog is built locally using [Quarto](https://quarto.org).

You can run `./build.sh` to build everything in the project.

### Managing Python environments

Python environments are managed by [uv](https://docs.astral.sh/uv/) a super fast Python package and project manager written in rust. Each post which requires Python has it's own environment.

```
my_post/
├── .venv
├── index.qmd
├── pyproject.toml
└── uv.lock
```

Create this environment by:

- Initializing the project, `uv init`
- Populate the dependencies in `pyproject.toml`
- Create the virtualenv using `uv venv`

Quarto will automatically use the virtual environment in the same folder as the `.qmd` file to run `python` chunks when running `quarto render`.

### Managing R dependencies

R dependencies are managed by the `renv` package. Each post which requires R has it's own environment.
