#include <windows.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])
{
    
  int    port_num;
  char   pipe_name[15];
  HANDLE pipe_hndl;
  
  DWORD  err;  
  
  /* Check for proper number of input arguments */
  if(nrhs!=1){
    mexErrMsgTxt("pipe_open requires one input argument (COM).\n");
  }

  /* Check type of first argument */
  if((!(mxIsNumeric(prhs[0])))||(mxGetNumberOfElements(prhs[0])!=1)){
    mexErrMsgTxt("pipe_read requires the argument to be a scalar.\n.");
  }
  
  /* Establish pipe number and name */
  port_num=(int)mxGetScalar(prhs[0]);
  sprintf(pipe_name,"\\\\.\\PIPE\\DACQ%d",port_num);
  mexPrintf("Pipe name: %s\n",pipe_name);
  
  /* Open pipe */
  pipe_hndl=CreateFile(pipe_name, 
                       GENERIC_READ, 
                       0, 
                       NULL, 
                       OPEN_EXISTING, 
                       0, 
                       NULL);
  if(pipe_hndl==INVALID_HANDLE_VALUE){
    err=GetLastError();
    mexPrintf("Error code %d in CreateFile.\n",err);
    mexErrMsgTxt("pipe_open could not open the pipe.\n");
  }
  
  /* Return handle */
  if(nlhs==1) plhs[0]=mxCreateDoubleScalar((double)((DWORD)(pipe_hndl)));
  else plhs[0]=(mxArray *)NULL;
    
}