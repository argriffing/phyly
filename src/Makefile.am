## Process this file with automake to produce Makefile.in

## For Makefile.am noobs like me, see
## https://www.gnu.org/software/automake/manual/html_node/General-Operation.html
## https://www.lrde.epita.fr/~adl/dl/autotools.pdf

# https://www.gnu.org/software/autoconf-archive/ax_valgrind_check.html

bin_PROGRAMS = arbplf-ll arbplf-node arbplf-dwell arbplf-trans arbplf-deriv

check_PROGRAMS = t-dummy_test

TESTS = $(check_PROGRAMS)

CORE_SOURCES = \
	dummy_module.c \
	model.c \
	model.h \
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
arbplf_node_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplf-node.c
arbplf_dwell_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplf-dwell.c
arbplf_trans_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplf-trans.c
arbplf_deriv_SOURCES =  $(CORE_SOURCES) $(JSON_SOURCES) arbplfderiv.c arbplfderiv.h arbplf-deriv.c
