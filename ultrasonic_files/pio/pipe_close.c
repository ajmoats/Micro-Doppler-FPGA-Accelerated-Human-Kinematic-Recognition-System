#include <windows.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])
{
  
  /* Close file */
  CloseHandle(prhs[0]);
    
}