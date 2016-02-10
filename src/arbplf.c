#include "Python.h"

#include "jansson.h"

#include "arbplfll.h"
#include "arbplfderiv.h"
#include "runjson.h"


/* The module doc string */
PyDoc_STRVAR(arbplf__doc__, "phylogenetic likelihood evaluation");

/* The function doc string */
PyDoc_STRVAR(arbplf_ll__doc__, "json in -> json out");
PyDoc_STRVAR(arbplf_deriv__doc__, "json in -> json out");

/* The wrapper to the underlying C function */
static PyObject *
py_arbplf_ll(PyObject *self, PyObject *args)
{
    const char *s_in;
    char *s_out;
    PyObject *ret;
    int retcode = 0;

	/* The ':arbplf_ll' is for error messages */
	if (!PyArg_ParseTuple(args, "s:arbplf_ll", &s_in))
		return NULL;
	
	/* Call the C function */
    s_out = NULL;

    /* define the json->json map */
    json_hom_t j_hom;
    j_hom->userdata = NULL;
    j_hom->clear = NULL;
    j_hom->f = arbplf_ll_run;
    {
        /* define the string->string map */
        string_hom_ptr s_hom = json_induced_string_hom(j_hom);
        {
            /* apply the string->string map */
            s_out = s_hom->f(s_hom->userdata, s_in, &retcode);
        }
        free(s_hom);
    }

    if (retcode)
    {
        PyErr_SetString(PyExc_RuntimeError, "arbplf likelihood error");
        ret = NULL;
    }
    else
    {
        ret = Py_BuildValue("s", s_out);
    }

    free(s_out);
    return ret;
}

/* The wrapper to the underlying C function */
static PyObject *
py_arbplf_deriv(PyObject *self, PyObject *args)
{
    const char *s_in;
    char *s_out;
    PyObject *ret;
    int retcode = 0;

	/* The ':arbplf_deriv' is for error messages */
	if (!PyArg_ParseTuple(args, "s:arbplf_deriv", &s_in))
		return NULL;
	
	/* Call the C function */
    s_out = NULL;

    /* define the json->json map */
    json_hom_t j_hom;
    j_hom->userdata = NULL;
    j_hom->clear = NULL;
    j_hom->f = arbplf_deriv_run;
    {
        /* define the string->string map */
        string_hom_ptr s_hom = json_induced_string_hom(j_hom);
        {
            /* apply the string->string map */
            s_out = s_hom->f(s_hom->userdata, s_in, &retcode);
        }
        free(s_hom);
    }

    if (retcode)
    {
        PyErr_SetString(PyExc_RuntimeError, "arbplf likelihood error");
        ret = NULL;
    }
    else
    {
        ret = Py_BuildValue("s", s_out);
    }

    free(s_out);
    return ret;
}

/* A list of all the methods defined by this module. */
/* "arbplf_ll" is the name seen inside of Python */
/* "py_arbplf_ll" is the name of the C function handling the Python call */
/* "METH_VARGS" tells Python how to call the handler */
/* The {NULL, NULL} entry indicates the end of the method definitions */
static PyMethodDef arbplf_methods[] = {
	{"arbplf_ll",  py_arbplf_ll, METH_VARARGS, arbplf_ll__doc__},
	{"arbplf_deriv",  py_arbplf_deriv, METH_VARARGS, arbplf_deriv__doc__},
	{NULL, NULL}      /* sentinel */
};

/* When Python imports a C module named 'X' it loads the module */
/* then looks for a method named "init"+X and calls it.  Hence */
/* for the module "arbplf" the initialization function is */
/* "initarbplf".  The PyMODINIT_FUNC helps with portability */
/* across operating systems and between C and C++ compilers */
PyMODINIT_FUNC
initarbplf(void)
{
	/* There have been several InitModule functions over time */
	Py_InitModule3("arbplf", arbplf_methods, arbplf__doc__);
}
