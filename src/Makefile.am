## Process this file with automake to produce Makefile.in

## For Makefile.am noobs like me, see
## https://www.gnu.org/software/automake/manual/html_node/General-Operation.html
## https://www.lrde.epita.fr/~adl/dl/autotools.pdf

# https://www.gnu.org/software/autoconf-archive/ax_valgrind_check.html

bin_PROGRAMS = arbplf-ll arbplf-marginal arbplf-dwell arbplf-trans \
	       arbplf-deriv arbplf-hess arbplf-inv-hess \
               arbplf-newton-delta arbplf-newton-refine \
	       arbplf-newton-update arbplf-em-update

check_PROGRAMS = t-dummy_test t-newton t-arb_vec_calc_quad

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
	evaluate_site_marginal.c \
	evaluate_site_marginal.h \
	arb_mat_extras.c \
	arb_mat_extras.h \
	arb_vec_extras.c \
	arb_vec_extras.h \
	arb_vec_calc.c \
	arb_vec_calc.h \
	arb_vec_calc_quad.c \
	arb_vec_calc_quad.h \
	rosenbrock.c \
	rosenbrock.h \
	equilibrium.c \
	equilibrium.h \
	cross_site_ws.c \
	cross_site_ws.h \
	finite_differences.c \
	finite_differences.h \
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
t_newton_SOURCES =  $(CORE_SOURCES) t-newton.c
t_arb_vec_calc_quad_SOURCES =  $(CORE_SOURCES) t-arb_vec_calc_quad.c

arbplf_ll_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfll.c arbplfll.h arbplf-ll.c
arbplf_marginal_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfmarginal.c arbplfmarginal.h arbplf-marginal.c
arbplf_dwell_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfdwell.c arbplfdwell.h arbplf-dwell.c
arbplf_trans_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplf-trans.c arbplftrans.h arbplftrans.c
arbplf_deriv_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfderiv.c arbplfderiv.h arbplf-deriv.c
arbplf_hess_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-hess.c
arbplf_inv_hess_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-inv-hess.c
arbplf_newton_update_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-newton-update.c
arbplf_newton_delta_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-newton-delta.c
arbplf_newton_refine_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfhess.c arbplfhess.h arbplf-newton-refine.c
arbplf_em_update_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfem.c arbplfem.h arbplf-em-update.c
