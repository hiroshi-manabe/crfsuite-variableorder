# $Id:$

lib_LTLIBRARIES = libcrf.la

EXTRA_DIST = \
	crf.vcproj 

libcrfincludedir = $(includedir)
libcrfinclude_HEADERS = \
	include/crfsuite.h

libcrf_la_SOURCES = \
	src/dictionary.c \
	src/logging.c \
	src/logging.h \
	src/params.c \
	src/params.h \
	src/quark.c \
	src/quark.h \
	src/rumavl.c \
	src/rumavl.h \
	src/mt19937ar.c \
	src/mt19937ar.h \
	src/crfvo.c \
	src/crfvo.h \
	src/crfvo_context.c \
	src/crfvo_feature.c \
	src/crfvo_learn.c \
	src/crfvo_learn_lbfgs.c \
	src/crfvo_preprocess.c \
	src/crfvo_model.c \
	src/crfvo_tag.c \
	src/crf.c

libcrf_la_CFLAGS = -I./include -I$(top_builddir)/lib/cqdb/include 

libcrf_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

libcrf_la_LIBADD = \
	$(top_builddir)/lib/cqdb/libcqdb.la

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
