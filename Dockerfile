FROM rocker/rstudio

# install renv
RUN R -e 'install.packages("renv", repos="http://cran.rstudio.com", dependencies=TRUE, lib="/usr/local/lib/R/site-library");'
# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

USER rstudio

# copy lock file & install deps
COPY --chown=rstudio:rstudio renv.lock /home/rstudio/project/
COPY --chown=rstudio:rstudio .Rprofile /home/rstudio/project/
COPY --chown=rstudio:rstudio renv /home/rstudio/project/renv
RUN R -e 'renv::restore(project="/home/rstudio/project");'

# copy the rest of the directory
# .dockerignore can ignore some files/folders if desirable
COPY --chown=rstudio:rstudio . /home/rstudio/project

USER root