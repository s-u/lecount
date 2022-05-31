\name{lecount}
\alias{lecount}
\title{
  Counting linear extensions of partially ordered sets.
}
\description{
  \code{lecount} counts linear extensions of partially ordered sets
  (posets). It includes a collection of algorithms, which are documented
  in the paper Counting linear extensions of sparse posets (IJCAI'16).
}
\usage{
lecount(m)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{m}{logical or integer square matrix. Each poset is represented
    as a 0-1 matrix, where a 1 at \code{[i,j]} indicates
    \code{i < j}. Any value other than 0 or \code{FALSE} is treated as 1
    in this context.}
}
\value{
  Resulting count.
}
\details{
  This R package is a wrapper of LEcount 16.04.20 C++ code by Kustaa
  Kangas.
}
%\references{
%}
\author{
  Kustaa Kangas
}
%\note{
%}
%\seealso{
%}
\examples{
m = matrix(c(
0,0,0,0,0,0,0,0,0,0,
0,0,0,1,0,0,0,0,1,0,
0,1,0,0,0,1,1,0,0,0,
0,0,0,0,0,1,0,0,0,0,
0,1,0,1,0,1,0,0,1,0,
1,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,0,1,0,0,0
),10,,TRUE)
lecount(m)
}
\keyword{manip}
