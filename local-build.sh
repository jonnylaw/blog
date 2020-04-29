docker build -f localbuild.Dockerfile -t local-blog-build .
docker run --rm -it -v $PWD/docs:/usr/local/app/docs local-blog-build make blog