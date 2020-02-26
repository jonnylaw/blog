FROM rocker/r-apt:disco

# Define environment variables for virtualenvs
ENV WORKON_HOME /opt/virtualenvs
ENV PYTHON_VENV_PATH $WORKON_HOME/r-reticulate

RUN apt-get update \
      && apt-get install -y --no-install-recommends \
      r-cran-rstan \
      r-cran-brms \
      r-cran-coda \
      r-cran-dplyr \
      r-cran-furrr \
      r-cran-ggmcmc \
      r-cran-ggplot2 \
      r-cran-ggthemes \
      r-cran-here \
      r-cran-htmltab \
      r-cran-janitor \
      r-cran-knitr \
      r-cran-leaflet \
      r-cran-magrittr \
      r-cran-microbenchmark \
      r-cran-readr \
      r-cran-reticulate \
      r-cran-tibble \
      r-cran-lubridate \
      libpython3-dev \
      python3-venv \
      curl \
      default-jdk \
      libxml2-dev \
      libssl-dev \
      libcurl4-openssl-dev \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/

# Add virtualenv to path
# ENV PATH ${PYTHON_VENV_PATH}/bin:${PATH}
## And set ENV for R! It doesn't read from the environment...
#RUN echo "PATH=${PATH}" >> /usr/local/lib/R/etc/Renviron && \
#    echo "WORKON_HOME=${WORKON_HOME}" >> /usr/local/lib/R/etc/Renviron && \
#    echo "RETICULATE_PYTHON_ENV=${PYTHON_VENV_PATH}" >> /usr/local/lib/R/etc/Renviron

# Copy contents of repo to container
COPY . blog/.

# Install Dependencies
RUN Rscript -e "install.packages('remotes', repos = 'https://demo.rstudiopm.com/all/__linux__/bionic/latest')"
RUN Rscript -e "remotes::install_deps(pkg = 'blog', dependencies = TRUE, repos = 'https://demo.rstudiopm.com/all/__linux__/bionic/latest')"
RUN Rscript -e "pkgbuild::build(path = 'blog'); install.packages('jonnylaw_0.1.0.tar.gz', type = 'source', repos = NULL)"

# Compile Blog
RUN cd blog && Rscript -e "blogdown::build_site()"
