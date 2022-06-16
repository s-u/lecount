# Process this file with autoconf to produce a configure script.
AC_INIT([lecount],[0.1],[Simon.Urbanek@r-project.org])
AC_CONFIG_SRCDIR([src/glue.cpp])
AC_CONFIG_HEADERS([src/config.h])

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

RBIN="${R_HOME}/bin/R"

# pick all flags for testing from R
: ${CC=`"${RBIN}" CMD config CC`}
: ${CXX=`"${RBIN}" CMD config CXX`}
: ${CFLAGS=`"${RBIN}" CMD config CFLAGS`}
: ${CXXFLAGS=`"${RBIN}" CMD config CXXFLAGS`}
: ${CPPFLAGS=`"${RBIN}" CMD config CPPFLAGS`}
: ${LDFLAGS=`"${RBIN}" CMD config LDFLAGS`}
: ${CPP="$CC -E"}
: ${PKGCONFIG=pkg-config}

# honor PKG_xx overrides
LIBS="${LIBS} ${PKG_LIBS}"
# for CPPFLAGS we will superfluously double R's flags
# since we'll set PKG_CPPFLAGS with this, but that shouldn't hurt
CPPFLAGS="${CPPFLAGS} ${PKG_CPPFLAGS}"

AC_SUBST(R_HOME)

AC_ARG_VAR([PKG_CPPFLAGS],[additional pre-processor flags])
AC_ARG_VAR([PKG_LIBS],[additional linker library flags])
AC_ARG_VAR([PKGCONFIG],[command to use for flags detection if none are supplied, defaults to pkg-config])

if test x"${PKG_CPPFLAGS}" = x -a x"${PKG_LIBS}" = x -a x"$PKGCONFIG" != x; then
   AC_MSG_CHECKING([for $PKGCONFIG])
   if $PKGCONFIG --version >/dev/null 2>&1; then
      AC_MSG_RESULT(yes)
      AC_MSG_CHECKING([whether it knows about openssl])
      if $PKGCONFIG gmpxx; then
         AC_MSG_RESULT(yes)
	 PC_CFLAGS=`$PKGCONFIG --cflags gmpxx`
	 PC_LIBS=`$PKGCONFIG --libs gmpxx`
	 AC_MSG_NOTICE([CFLAGS: ${PC_CFLAGS}])
	 AC_MSG_NOTICE([LIBS: ${PC_LIBS}])
	 CPPFLAGS="${CPPFLAGS} ${PC_CFLAGS}"
	 LIBS="${LIBS} ${PC_LIBS}"
      else
         AC_MSG_RESULT(no)
      fi
   else
      AC_MSG_RESULT(no)
   fi
fi

# Checks for programs.
AC_LANG([C++])
AC_PROG_CXX

AC_CHECK_HEADER([gmpxx.h],,[AC_MSG_ERROR([Failed to find usable gmpxx.h headers])])

#[AC_SEARCH_LIBS(SSL_CTX_load_verify_locations, [ssl openssl],,
#  [AC_MSG_ERROR([Cannot find usable SSL library])])],
#[AC_MSG_ERROR([Failed to find usable SSL library. Please install openssl-dev or equivalent and/or set PKG_LIBS if not in default location.])]
#)

AC_SUBST(LIBS)
AC_SUBST(CPPFLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
