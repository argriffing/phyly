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
            'arbplfdwell.c',
            'arbplfem.c',
            'arbplftrans.c',
            'csr_graph.c',
            'util.c',
            'model.c',
            'ndaccum.c',
            'evaluate_site_lhood.c',
            'evaluate_site_marginal.c',
            'evaluate_site_frechet.c',
            'parsemodel.c',
            'reduction.c',
            'parsereduction.c',
            'arb_mat_extras.c',
            'arb_vec_extras.c',
            'arb_vec_calc.c',
            'arb_vec_calc_quad.c',
            'rosenbrock.c',
            'equilibrium.c',
            'distribution.c',
            'rate_mixture.c',
            'cross_site_ws.c',
            'finite_differences.c',
            'gamma_discretization.c',
            ])

setup(name="arbplf", version="0.0.1",
        scripts=['scripts/tabby'],
        ext_modules = [myext])
