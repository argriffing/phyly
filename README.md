under construction...


Requirements
------------

The code has been tested only on Linux, and the installation requires
[autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html)
which should be available as `autotools-dev`
on Linux distributions based on debian.

The package depends on these C libraries:
 * [arb](https://github.com/fredrik-johansson/arb)
   -- C library for arbitrary-precision interval arithmetic
 * [flint2](https://github.com/wbhart/flint2)
   -- <b>F</b>ast <b>Li</b>brary for <b>N</b>umber <b>T</b>heory
 * [gmp](https://gmplib.org/)
   -- The GNU Multiple Precision Arithmetic Library
 * [jansson](https://github.com/akheron/jansson)
   -- C library for encoding, decoding and manipulating JSON data

The [jq](https://stedolan.github.io/jq/) tool may be used for json filtering.


Installation
------------

Something like the usual autotools installation commands should
work if you are lucky:

```shell
$ ./autogen.sh
$ ./configure CPPFLAGS='-I/path/to/include/flint'
$ make
$ make check
$ make install
```

The extra CPPFLAGS path is due to the
[idiosyncratic](https://github.com/fredrik-johansson/arb/issues/24)
way that arb includes the flint2 headers.

To use a configuration tuned to your specific machine architecture,
use a `configure` command like the following:

```shell
$ ./configure CFLAGS='-O3 -march=native -ffast-math' CPPFLAGS='-I/path/to/include/flint'
```
