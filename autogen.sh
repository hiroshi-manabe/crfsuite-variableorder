#!/bin/sh
# $Id: autogen.sh 39 2007-11-30 14:26:42Z naoaki $

if [ "$1" = "--force" ];
then
    FORCE=--force
    NOFORCE=
    FORCE_MISSING=--force-missing
else
    FORCE=
    NOFORCE=--no-force
    FORCE_MISSING=
fi

libtoolize --copy $FORCE 2>&1 | sed '/^You should/d' || {
    echo "libtoolize failed!"
    exit 1
}

aclocal $FORCE || {
    echo "aclocal failed!"
    exit 1
}

autoheader $FORCE || {
    echo "autoheader failed!"
    exit 1
}

automake -a -c $NOFORCE || {
    echo "automake failed!"
    exit 1
}

autoconf $FORCE || {
    echo "autoconf failed!"
    exit 1
}
