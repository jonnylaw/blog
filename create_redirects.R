# list names of post folders
posts <- list.dirs(
  path = here::here("posts"),
  full.names = FALSE,
  recursive = FALSE
)

# extract the slugs
slugs <- gsub("^.*_", "", posts)

# lines to insert to a netlify _redirect file
redirects <- paste0("/", slugs, " ", "/posts/", posts)

# write the _redirect file
writeLines(redirects, here::here("_site", "_redirects"))
