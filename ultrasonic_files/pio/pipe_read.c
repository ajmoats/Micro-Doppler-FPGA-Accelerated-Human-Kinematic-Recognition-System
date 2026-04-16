#include <windows.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int tmp,num_req,num_rec;
  mwSize dims[2];
  mxArray *data;
  unsigned char *buf;
  HANDLE hndl;
  DWORD err;

  /* Check for proper number of input arguments */
  if(nrhs!=1){
    mexErrMsgTxt("pipe_read requires one input argument.\n");
  }

  /* Check type of first argument */
  if((!(mxIsNumeric(prhs[0])))||(mxGetNumberOfElements(prhs[0])!=1)){
    mexErrMsgTxt("pipe_read requires the argument to be a scalar.\n.");
  }

  /* Convert argument to a number */
  num_req=(int)mxGetScalar(prhs[0]);

  num_rec=0;
  if(num_req>0){
    /* Allocate array */
    dims[0]=1;
    dims[1]=num_req*1940;
    data=mxCreateNumericArray(2,dims,mxUINT16_CLASS,mxREAL);
    buf=(unsigned char *)mxGetData(data);

    /* Open pipe */
    hndl=CreateFile("\\\\.\\PIPE\\DACQ3",GENERIC_READ,0,NULL,OPEN_EXISTING,0,NULL);
    if(hndl==INVALID_HANDLE_VALUE){
      err=GetLastError();
      mexPrintf("Error code %d in CreateFile.\n",err);
      mxFree(buf);
      mexErrMsgTxt("pipe_read could not open the pipe.\n");
    }

    /* Read data from device into str */
    if(!ReadFile(hndl,&(buf[0]),num_req*1940,(DWORD*)(&num_rec),NULL)){
      err=GetLastError();
      mexPrintf("Error code %d in ReadFile.\n",err);
      mxFree(buf);
      CloseHandle(hndl);
      mexErrMsgTxt("pipe_read could not read from the device.\n");
    }

    /* Close file */
    CloseHandle(hndl);
  }

  /* Return string */
  if(nlhs>0){
    plhs[0]=data;
  }else{
    mxDestroyArray(data);
  }
}
