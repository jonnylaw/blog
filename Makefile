blog: scala
	Rscript -e "Sys.setenv(RSTUDIO_PANDOC = '/Applications/RStudio.app/Contents/MacOS/pandoc'); blogdown::build_site(); blogdown::render_site()"

scala:
	RUN /usr/local/app/notebooks/run_scala.sh