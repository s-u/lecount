lecount <- function(m) {
    if (!is.matrix(m) || length(dim(m)) != 2 || diff(dim(m) != 0))
        stop("`m' must be a square matrix")
    .Call(C_lecount, as.integer(m), as.double(dim(m)[1]))
}
