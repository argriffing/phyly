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
	arbplfll.c \
	arbplfll.h \
	parsemodel.c \
	parsemodel.h \
	model.c \
	model.h \
	csr_graph.c \
	csr_graph.h

JSON_SOURCES = \
	runjson.c \
	runjson.h


ALL_SOURCES = $(CORE_SOURCES) $(JSON_SOURCES)

t_dummy_test_SOURCES =  $(CORE_SOURCES) t-dummy_test.c

arbplf_ll_SOURCES =  $(ALL_SOURCES) arbplf-ll.c
arbplf_node_SOURCES =  $(ALL_SOURCES) arbplf-node.c
arbplf_dwell_SOURCES =  $(ALL_SOURCES) arbplf-dwell.c
arbplf_trans_SOURCES =  $(ALL_SOURCES) arbplf-trans.c
arbplf_deriv_SOURCES =  $(ALL_SOURCES) arbplf-deriv.c
