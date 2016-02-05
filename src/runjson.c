#include <stdio.h>

#include "jansson.h"

#include "runjson.h"


char *jsonwrap(void *userdata, const char *s_in, int *retcode);

char *jsonwrap(void *userdata, const char *s_in, int *retcode)
{
    json_hom_ptr p = userdata;

    json_error_t error;
    const size_t flags = 0;
    json_t *j_in;
    json_t *j_out;
    char *s_out;
    int result;

    result = 0;
    j_in = NULL;
    j_out = NULL;
    s_out = NULL;

    /* convert input string to json using jansson */
    j_in = json_loads(s_in, flags, &error);
    if (!j_in)
    {
        fprintf(stderr, "error on json line %d: %s\n",
            error.line, error.text);
        result = -1;
        goto finish;
    }

    /* call the underlying function to get a new jansson object */
    j_out = p->f(p->userdata, j_in, &result);
    if (result)
    {
        goto finish;
    }

    /* convert the output jansson object to a string */
    if (j_out)
    {
        s_out = json_dumps(j_out, flags);
        if (!s_out)
        {
            fprintf(stderr, "error: failed to dump ");
            fprintf(stderr, "the json object to a string\n");
            result = -1;
            goto finish;
        }
    }

finish:

    *retcode = result;

    /* attempt to free the input and output json objects */
    json_decref(j_in);
    json_decref(j_out);

    /* return the output string which must be freed by the caller */
    return s_out;
}


string_hom_ptr
json_induced_string_hom(json_hom_t hom)
{
    string_hom_ptr p = malloc(sizeof(string_hom_struct));
    p->userdata = hom;
    p->clear = NULL;
    p->f = jsonwrap;
    return p;
}




char *fgets_dynamic(FILE *stream);

/*
 * Reads unlimited input into a string that must be freed by the caller.
 * This is just a utility function.
 */
char *fgets_dynamic(FILE *stream)
{
  int size = 0;
  int capacity = 20;
  char *s = calloc(capacity, sizeof(*s));
  char *tail = s;

  /* keep reading from the command line until EOF */
  while (fgets(tail, capacity - size, stream)) {
    /* fprintf(stderr, "debug: {size: %d, capacity: %d}\n", size, capacity);
     */
    size += strlen(tail);
    if (size == capacity-1) {
      capacity <<= 1;
      s = realloc(s, capacity * sizeof(*s));
      if (!s) {
        fprintf(stderr, "fgets_dynamic: failed to reallocate %d\n", capacity);
        return 0;
      }
    }
    tail = s + size;
  }

  /* return the newly allocated and filled string */
  return s;
}



int
run_string_script(string_hom_t hom)
{
    char *s_in = 0;
    char *s_out = 0;
    int retcode = 0;

    /* read the string from stdin until we find eof */
    s_in = fgets_dynamic(stdin);
    if (!s_in) {
        fprintf(stderr, "failed to read string from stdin\n");
        return -1;
    }

    /* call the string interface wrapper */
    s_out = hom->f(hom->userdata, s_in, &retcode);

    /* free the input string */
    free(s_in);

    /* write the string to stdout */
    /* free the string allocated by the script function */
    if (s_out)
    {
      puts(s_out);
      free(s_out);
    }

    /* return zero if no error */
    return retcode;
}


int
run_json_script(json_hom_t hom)
{
    string_hom_ptr p = json_induced_string_hom(hom);
    int result = run_string_script(p);
    free(p);
    return result;
}
