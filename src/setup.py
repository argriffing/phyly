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
            'csr_graph.c',
            'model.c',
            'parsemodel.c',
            ])

setup(name="arbplf", version="0.0.1", ext_modules = [myext])
