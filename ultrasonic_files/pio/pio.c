#include <windows.h>
#include "mex.h"
#include "matrix.h"

#define PIO_OPEN  0
#define PIO_CLOSE 1
#define PIO_WRITE 2
#define PIO_READ  3

typedef struct _pio {
  char         pipe[16];
  HANDLE       hndl;
  struct _pio *next;
} pio;

static pio *pio_head=(pio *)NULL;
static char pio_err[1024];

static void pio_exit(void)
{
  pio *ptr1,*ptr2;

  /* Close files and free allocated memory */
  ptr1=pio_head;
  while(ptr1!=(pio *)NULL){
    CloseHandle(ptr1->hndl);
    ptr2=ptr1->next;
    mxFree((void *)ptr1);
    ptr1=ptr2;
  }
  pio_head=(pio *)NULL;
}

pio *pio_find(char *func,const mxArray *arg,char *pipe,pio **prev)
{
  char    name[16];
  HANDLE  hndl;
  int     len;
  pio    *ptr1,*ptr2;
  mxChar *data;

  if(mxIsChar(arg)){
    /* Get string length */
    len=(int)mxGetNumberOfElements(arg);
    if((len!=1)&&(len!=2)){
    form_error:
      sprintf(pio_err,"%s require the pipe number to be of the form 'n' or 'nn'.\n",func);
      mexErrMsgTxt(pio_err);
    }
    /* Check pipe number contents */
    data=(mxChar *)mxGetData(arg);
    if((((char)data[0])<'1')||(((char)data[0])>'9')) goto form_error;
    if(len==2){
      if((((char)data[1])<'0')||(((char)data[1])>'9')) goto form_error;
    }

    /* Build new pipe name */
    name[0] ='\\';
    name[1] ='\\';
    name[2] ='.';
    name[3] ='\\';
    name[4] ='P';
    name[5] ='I';
    name[6] ='P';
    name[7] ='E';
    name[8] ='\\';
    name[9] ='D';
    name[10]='A';
    name[11]='C';
    name[12]='Q';
    name[13]=(char)data[0];
    if(len==2) name[14]=(char)data[1];
    else name[14]='\0';
    name[15]='\0';

    /* Search for a matching pipe name in linked list */
    ptr1=pio_head;
    ptr2=(pio *)NULL;
    while(1){
      if(ptr1==(pio *)NULL){
        /* Not found */
        if(pipe!=(char *)NULL) strcpy(pipe,name);
        if(prev!=(pio **)NULL) (*prev)=(pio *)NULL;
        return((pio *)NULL);
      }
      if(!strcmp(ptr1->pipe,name)){
        /* Found */
        if(prev!=(pio **)NULL) (*prev)=ptr2;
        return(ptr1);
      }
      ptr2=ptr1;
      ptr1=ptr1->next;
    }
  }else if(mxIsNumeric(arg)){
    /* Check size of argument */
    if(mxGetNumberOfElements(arg)!=1){
      /* File descriptor should be a single number */
      sprintf(pio_err,"%s requires that a device handle be a scalar.\n",func);
      mexErrMsgTxt(pio_err);
    }
    /* Convert mxArray to a handle */
    hndl=(HANDLE)((DWORD)mxGetScalar(arg));
    /* Search for a matching file descriptor in linked list */
    ptr1=pio_head;
    ptr2=(pio *)NULL;
    while(1){
      if(ptr1==(pio *)NULL){
        /* Not found */
        if(prev!=(pio **)NULL){
          (*prev)=(pio *)NULL;
        }
        return((pio *)NULL);
      }
      if(ptr1->hndl==hndl){
        /* Found */
        if(prev!=(pio **)NULL){
          (*prev)=ptr2;
        }
        return(ptr1);
      }
      ptr2=ptr1;
      ptr1=ptr1->next;
    }
  }else{
    sprintf(pio_err,"%s requires that the first additional argument be a pipe name or a device handle.\n",func);
    mexErrMsgTxt(pio_err);
  }
}

mxArray *pio_open(int nlhs,int nrhs,const mxArray *prhs[])
{
  pio   *ptr;
  char   pipe[16];
  HANDLE hndl;
  DWORD  err;

  /* Check for proper number of arguments */
  if(nrhs!=2){
    mexErrMsgTxt("pio_open requires one additional argument.\n");
  }
  if((nlhs!=0)&&(nlhs!=1)){
    mexErrMsgTxt("pio_open returns at most one argument.\n");
  }

  /* Convert argument to linked list entry */
  ptr=pio_find("pio_open",prhs[1],pipe,(pio **)NULL);

  /* Check if entry exists */
  if(ptr!=(pio *)NULL){
    if(nlhs==1) return(mxCreateDoubleScalar((double)((DWORD)(ptr->hndl))));
    else return((mxArray *)NULL);
  }
  
  /* Open pipe */
  hndl=CreateFile(pipe,GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
  if(hndl==INVALID_HANDLE_VALUE){
    err=GetLastError();
    mexPrintf("Error code %d in CreateFile.\n",err);
    mexErrMsgTxt("pio_open could not open the pipe.\n");
  }

  /* Create an pio entry */
  if(pio_head==(pio *)NULL){
    pio_head=(pio *)mxMalloc(sizeof(pio));
    ptr=pio_head;
  }else{
    for(ptr=pio_head;ptr->next!=(pio *)NULL;ptr=ptr->next);
    ptr->next=(pio *)mxMalloc(sizeof(pio));
    ptr=ptr->next;
  }
  strcpy(ptr->pipe,pipe);
  ptr->hndl=hndl;
  ptr->next=(pio *)NULL;
  mexMakeMemoryPersistent(ptr);

  /* Return device handle */
  if(nlhs==1) return(mxCreateDoubleScalar((double)((DWORD)(ptr->hndl))));
  else return((mxArray *)NULL);
}

void pio_close(int nlhs,int nrhs,const mxArray *prhs[])
{
  pio *ptr,*prev;

  /* Check for proper number of arguments */
  if(nrhs!=2){
    mexErrMsgTxt("pio_close requires one additional argument.\n");
  }
  if(nlhs!=0){
    mexErrMsgTxt("pio_close returns no argument.\n");
  }

  /* Convert argument to linked list entry */
  ptr=pio_find("pio_close",prhs[1],(char *)NULL,&prev);

  /* Check if entry exists */
  if(ptr==(pio *)NULL){
    /* Pipe not open is not a fatal error */
    mexWarnMsgTxt("pio_close can only close open pipes opened by pio_open.\n");
    return;
  }

  /* Remove entry from list */
  if(prev!=(pio *)NULL){
    prev->next=ptr->next;
  }else{
    pio_head=ptr->next;
  }

  /* Close file */
  CloseHandle(ptr->hndl);

  /* Free memory */
  mxFree(ptr);
}

void pio_write(int nlhs,int nrhs,const mxArray *prhs[])
{
  pio *ptr;
  int tmp,cnt,len;
  unsigned char *str;
  mxChar *data;
  DWORD err;

  /* Check for proper number of arguments */
  if(nrhs!=3){
    mexErrMsgTxt("pio_write requires two additional arguments.\n");
  }
  if(nlhs!=0){
    mexErrMsgTxt("pio_write returns no argument.\n");
  }

  /* Convert argument to linked list entry */
  ptr=pio_find("pio_write",prhs[1],(char *)NULL,(pio **)NULL);

  /* Check if entry exists */
  if(ptr==(pio *)NULL){
    mexErrMsgTxt("pio_write can only write to serial ports opened by pio_open.\n");
  }

  /* Check type of second argument */
  if((!(mxIsChar(prhs[2])))||(mxGetM(prhs[2])!=1)){
    mexErrMsgTxt("pio_write requires the second additional argument to be a string.\n.");
  }

  /* Get string length */
  len=(int)mxGetNumberOfElements(prhs[2]);

  if(len>0){
    /* Convert second argument to string */
    str=(unsigned char *)mxMalloc(len*sizeof(unsigned char));
    data=(mxChar *)mxGetData(prhs[2]);
    for(tmp=0;tmp<len;tmp++){
      str[tmp]=(unsigned char)data[tmp];
    }

    /* Write string to device */
    cnt=0;
    while(cnt<len){
      if(!WriteFile(ptr->hndl,&(str[cnt]),len-cnt,&tmp,NULL)){
        err=GetLastError();
        mexPrintf("Error code %d in Writefile.\n",err);
        mxFree(str);
        mexErrMsgTxt("pio_write could not write to the device.\n");
      }
      cnt=cnt+tmp;
    }
    mxFree(str);
  }
}

mxArray *pio_read(int nlhs,int nrhs,const mxArray *prhs[])
{
  pio *ptr;
  int tmp,cnt,len,nbr;
  unsigned char *str=(unsigned char *)NULL;
  mxChar *data;
  mxArray *ret;
#ifdef MWSIZE_MIN
  /* Newer versions of Matlab use *mwSize for mxCreateCharArray() */
  mwSize size[2];
#else
  /* Older versions of Matlab use *int for mxCreateCharArray() */
  int size[2];
#endif
  DWORD err;

  /* Check for proper number of arguments */
  if(nrhs!=3){
    mexErrMsgTxt("pio_read requires two additional arguments.\n");
  }
  if((nlhs!=0)&&(nlhs!=1)){
    mexErrMsgTxt("pio_read returns returns at most one argument.\n");
  }

  /* Convert argument to linked list entry */
  ptr=pio_find("pio_read",prhs[1],(char *)NULL,(pio **)NULL);

  /* Check if entry exists */
  if(ptr==(pio *)NULL){
    mexErrMsgTxt("pio_read can only read from pipes opened by pio_open.\n");
  }

  /* Check type of second argument */
  if((!(mxIsNumeric(prhs[2])))||(mxGetNumberOfElements(prhs[2])!=1)){
    mexErrMsgTxt("pio_read requires the second additional argument to be a scalar.\n.");
  }

  /* Convert second argument to byte count */
  cnt=(int)mxGetScalar(prhs[2]);

  len=0;
  if(cnt>0){
    /* Allocate string */
    str=(unsigned char *)mxMalloc(cnt*sizeof(unsigned char));

    /* Read data from device */
    while(len<cnt){
      if(!ReadFile(ptr->hndl,&(str[len]),cnt-len,&nbr,NULL)){
        err=GetLastError();
        mexPrintf("Error code %d in ReadFile.\n",err);
        mxFree(str);
        mexErrMsgTxt("pio_read could not read from the device.\n");
      }
      len+=nbr;
    }
  }

  /* Return string */
  if(nlhs==1){
    size[0]=1;
    size[1]=len;
    ret=mxCreateCharArray(2,size);
    data=(mxChar *)mxGetData(ret);
    for(tmp=0;tmp<len;tmp++){
      data[tmp]=(mxChar)str[tmp];
    }
    if(str!=(unsigned char *)NULL) mxFree(str);
    return(ret);
  }else{
    if(str!=(unsigned char *)NULL) mxFree(str);
    return((mxArray *)NULL);
  }
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int op;
  mxArray *ret;

  /* Register cleanup function */
  mexAtExit(pio_exit);

  /* Check for proper number of input arguments */
  if(nrhs<2){
    mexErrMsgTxt("pio requires at least two input arguments.\n");
  }

  /* Check type of first argument */
  if((!(mxIsNumeric(prhs[0])))||(mxGetNumberOfElements(prhs[0])!=1)){
    mexErrMsgTxt("pio requires the operation argument to be an integer between 0 and 2.\n.");
  }
  
  /* Convert first argument to a number */
  op=(int)mxGetScalar(prhs[0]);
  switch(op){
  case PIO_OPEN:
    ret=pio_open(nlhs,nrhs,prhs);
    if(nlhs==1) plhs[0]=ret;
    break;
  case PIO_CLOSE:
    pio_close(nlhs,nrhs,prhs);
    break;
  case PIO_WRITE:
    pio_write(nlhs,nrhs,prhs);
    break;
  case PIO_READ:
    ret=pio_read(nlhs,nrhs,prhs);
    if(nlhs==1) plhs[0]=ret;
    break;
  default:
    mexErrMsgTxt("pio requires the operation argument to be an integer between 0 and 2.\n.");
  }
}
