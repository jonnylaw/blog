blog:
	Rscript -e "Sys.setenv(RSTUDIO_PANDOC = '/Applications/RStudio.app/Contents/MacOS/pandoc'); blogdown::build_site(); blogdown::render_site()"

container:
	docker build . --file Dockerfile --tag image
