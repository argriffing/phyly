Requirements
------------

The code has been tested only on Linux, and the installation requires
[autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html)
which should be available as `autotools-dev`
on Linux distributions based on debian.

The package depends on recent versions of these C libraries:
 * [jansson](https://github.com/akheron/jansson)
   -- C library for encoding, decoding and manipulating JSON data
 * [gmp](https://gmplib.org/)
   -- The GNU Multiple Precision Arithmetic Library
 * [flint2](https://github.com/wbhart/flint2)
   -- <b>F</b>ast <b>Li</b>brary for <b>N</b>umber <b>T</b>heory
 * [arb](https://github.com/fredrik-johansson/arb)
   -- C library for arbitrary-precision interval arithmetic

The [jq](https://stedolan.github.io/jq/) tool may be used for json filtering.


Installation
------------

Something like the usual autotools installation commands should
work if you are lucky:

```shell
$ ./autogen.sh
$ ./configure --prefix=/my/prefix CFLAGS='-march=native -O3 -ffast-math -g -Wall -Wextra'
$ make
$ make check
$ make install
```

A rudimentary Python interface can be installed using the `setup.py`
script as follows:

```shell
$ cd src
src$ python setup.py install --user
src$ cd..
$ nosetests
```

Features
--------

These features are available through command-line programs
with executable names like `arbplf-newton-update` which
operate via JSON files on stdin and stdout.
If the Python extension is installed, the features are available in the
`arbplf` module with function names like `arbplf_newton_update`
using a JSON string API.

 - ll -- log likelihood
 - deriv -- derivative of log likelihood w.r.t. edge rate coefficients
 - hess -- hessian matrix of log likelihood w.r.t. edge rate coefficients
 - inv-hess -- inverse of hessian matrix
 - marginal -- marginal state distributions at nodes
 - dwell -- linear combinations of dwell time expectations in states on edges
 - trans -- linear combinations of labeled transition count expectations on edges
 - em-update -- update edge rate coefficients using one step of EM
 - newton-update -- update edge rate coefficients using one step of Newton's method
 - newton-delta -- the difference between the newton update and the current values
 - newton-refine -- certify an interior local optimum near an initial guess of edge rate coefficients
