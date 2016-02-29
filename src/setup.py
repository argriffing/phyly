from __future__ import print_function, division

from distutils.core import setup, Extension

import distutils.log

distutils.log.set_verbosity(distutils.log.DEBUG)


myext = Extension(
        'arbplf',
        include_dirs=[
            '/usr/local/include',
            '/usr/local/include/flint'],
        libraries=['gmp', 'flint', 'arb', 'jansson'],
        sources = [
            'arbplf.c',
            'runjson.c',
            'arbplfll.c',
            'arbplfderiv.c',
            'arbplfhess.c',
            'arbplfmarginal.c',
            'csr_graph.c',
            'util.c',
            'model.c',
            'evaluate_site_lhood.c',
            'parsemodel.c',
            'reduction.c',
            'parsereduction.c',
            ])

setup(name="arbplf", version="0.0.1", ext_modules = [myext])
