FROM rocker/r-apt:disco

WORKDIR /usr/local/app

RUN  apt-get update \
  && apt-get install software-properties-common \
  && add-apt-repository ppa:ubuntugis/ppa \
  && apt-get install -y --no-install-recommends \
  hugo \
  pandoc \
  curl \
  libpython3-dev \
  python3-venv \
  default-jdk \
  libxml2-dev \
  libssl-dev \
  libudunits2-dev \
  libgdal-dev \
  libcurl4-openssl-dev \
  libmagick++-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/
  
# Copy directory
# Including renv cache
COPY . .
  
# Restore R packages from local cache
RUN Rscript -e "Sys.setenv(RENV_PATHS_CACHE = '/usr/local/app/renv/library/R-4.0/x86_64-apple-darwin15.6.0'); renv::restore(); tinytex::install_tinytex(); install.packages('blogdown')"

# Install Ammonite
RUN sh -c '(echo "#!/usr/bin/env sh" && curl -L https://github.com/lihaoyi/Ammonite/releases/download/2.0.4/2.12-2.0.4) > /usr/local/bin/amm && chmod +x /usr/local/bin/amm'

# Build site
RUN Rscript -e "options(blogdown.subdir = 'blog'); blogdown::build_site()"