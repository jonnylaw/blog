FROM rocker/tidyverse:3.6.2

RUN  apt-get update \
  && apt-get install software-properties-common \
  && add-apt-repository ppa:ubuntugis/ppa \
  && apt-get install -y --no-install-recommends \
  libpython3-dev \
  python3-venv \
  pandoc \
  curl \
  default-jdk \
  libxml2-dev \
  libssl-dev \
  libudunits2-dev \
  libgdal-dev \
  libcurl4-openssl-dev \
  libv8-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/
  
# Restore R packages from local cache
RUN Rscript -e "Sys.setenv(RENV_PATHS_CACHE = "/renv/library/R-3.6/x86_64-apple-darwin15.6.0"); renv::restore()"

# Install Ammonite
RUN sh -c '(echo "#!/usr/bin/env sh" && curl -L https://github.com/lihaoyi/Ammonite/releases/download/2.0.4/2.12-2.0.4) > /usr/local/bin/amm && chmod +x /usr/local/bin/amm'

# Copy directory
# Including renv cache
COPY . .

# Run scala examples
RUN ./notebooks/run_scala.sh

# Render site
RUN Rscript -e "options(blogdown.subdir = 'blog'); blogdown::install_hugo(); blogdown::build_site()"