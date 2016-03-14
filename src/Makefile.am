## Process this file with automake to produce Makefile.in

## For Makefile.am noobs like me, see
## https://www.gnu.org/software/automake/manual/html_node/General-Operation.html
## https://www.lrde.epita.fr/~adl/dl/autotools.pdf

# https://www.gnu.org/software/autoconf-archive/ax_valgrind_check.html

bin_PROGRAMS = arbplf-ll arbplf-marginal arbplf-dwell arbplf-trans \
	       arbplf-deriv arbplf-hess arbplf-inv-hess \
	       arbplf-coeff-expect arbplf-newton-point arbplf-newton-delta \
	       arbplf-opt-cert

check_PROGRAMS = t-dummy_test

TESTS = $(check_PROGRAMS)

CORE_SOURCES = \
	dummy_module.c \
	util.c \
	util.h \
	model.c \
	model.h \
	ndaccum.c \
	ndaccum.h \
	evaluate_site_lhood.c \
	evaluate_site_lhood.h \
	arb_vec_extras.c \
	arb_vec_extras.h \
	parsemodel.c \
	parsemodel.h \
	reduction.c \
	reduction.h \
	parsereduction.c \
	parsereduction.h \
	csr_graph.c \
	csr_graph.h

JSON_SOURCES = \
	runjson.c \
	runjson.h

t_dummy_test_SOURCES =  $(CORE_SOURCES) t-dummy_test.c

arbplf_ll_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfll.c arbplfll.h arbplf-ll.c
arbplf_marginal_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfmarginal.c arbplfmarginal.h arbplf-marginal.c
arbplf_dwell_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfdwell.c arbplfdwell.h arbplf-dwell.c
arbplf_coeff_expect_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfcoeffexpect.c arbplfcoeffexpect.h arbplf-coeff-expect.c
arbplf_trans_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplf-trans.c arbplftrans.h arbplftrans.c
arbplf_deriv_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfderiv.c arbplfderiv.h arbplf-deriv.c
arbplf_hess_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-hess.c
arbplf_inv_hess_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-inv-hess.c
arbplf_newton_point_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-newton-point.c
arbplf_newton_delta_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-newton-delta.c
arbplf_opt_cert_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-opt-cert.c
