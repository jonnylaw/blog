blog: 
	Rscript -e "rmarkdown::render_site()"

scala:
	RUN /usr/local/app/notebooks/run_scala.sh
	
docker:
	docker build -f localbuild.Dockerfile -t local-blog-build .
	docker run --rm -it -v $PWD/docs:/usr/local/app/docs local-blog-build make blog