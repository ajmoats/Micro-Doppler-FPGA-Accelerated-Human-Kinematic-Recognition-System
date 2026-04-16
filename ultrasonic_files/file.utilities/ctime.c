/* T.S.M
 * 2011/12
 *
 * [WINDOWS time string] = ctime(MATLAB timestamp (uint64)*9.6E-3)
 */
#include <windows.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

char *etime = "Invalid time\n";

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  time_t  timestamp;
  char   *ptr;
  
  /* Check for proper number of input arguments */
  if(nrhs!=1){
    mexErrMsgTxt("ctime requires precisely one input argument (the timestamp).\n");
  }
  /* Check type of first argument */
  if((!(mxIsNumeric(prhs[0])))||(mxGetNumberOfElements(prhs[0])!=1)){
    mexErrMsgTxt("ctime requires the timestamp argument to be an integer between 0 and 2^64-1.\n");
  }
  
  timestamp = (time_t)mxGetScalar(prhs[0]);
  ptr = ctime(&timestamp);
  if(ptr == (char *)NULL){
    //mexErrMsgTxt("timestamp argument is invalid.\n");
    ptr = etime;
  }
  
  if(nlhs==1){
    plhs[0]=mxCreateString(ptr);    
  }else{
    mexErrMsgTxt("give ctime an ouput argument to get the result.\n");
  }
}