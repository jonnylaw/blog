#!/bin/bash -eu

docker build -t blog-build .
docker run --rm -v "$(pwd)":/home/rstudio/project -w /home/rstudio/project blog-build bash -c "quarto render"