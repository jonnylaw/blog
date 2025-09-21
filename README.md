# Blog

Jonny Law's Weblog.

## Building

This blog is built locally using [Quarto](https://quarto.org).

You can run `./build.sh` to build everything in the project.

## Previewing

To preview the site with proper CSS and navigation:

1. Render the entire site:
   ```bash
   quarto render
   ```

2. Serve the site locally:
   ```bash
   python3 -m http.server 8000 --directory _site
   ```

3. Open in browser: http://localhost:8000

Alternatively, for live preview (may have Python environment issues):
```bash
quarto preview
```

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

To run Python cells interactively in Positron:

1. Open Command Palette (Cmd+Shift+P)
2. Select "Python: Select Interpreter"
3. Choose the UV environment: `posts/[post-name]/.venv/bin/python`
4. Open the `.qmd` file and run cells with Cmd+Enter

### Managing R dependencies

R dependencies are managed by the `renv` package.
