/*******************************************************************************
 *
 * apc.c
 * This file is part of fplib.
 *
 * SOLUTION OF THE LINEAR MIN-SUM ASSIGNMENT PROBLEM.
 * HUNGARIAN METHOD. COMPLEXITY O(n^3).
 *
 * October 2008:
 * - Original FORTRAN code translated in C LANGUAGE by Andrea Tramontani
 *   Andrea Tramontani
 *   DEIS, University of Bologna
 *   Viale Risorgimento, 2
 *   40136 - Bologna (Italy)
 *
 * October 2015:
 * - motified by Li Zhu <z@zhuli.name>
 *
 * ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>

void init(int n, double *a, int *f, double *u, double *v, int *fb, int *p,
        double INF, int *m_p);
void path(int n, double *a, int *f, double *u, double *v, int *fb, int *rc,
        double *pi, int *lr, int *uc, double INF, int ii, int *jj_p);
void incr(int *f,int *fb,int *rc,int j);


/*

   SOLUTION OF THE LINEAR MIN-SUM ASSIGNMENT PROBLEM.

   HUNGARIAN METHOD. COMPLEXITY O(n^3).


   MEANING OF THE INPUT PARAMETERS:
   n       = NUMBER OF ROWS AND COLUMNS OF THE COST MATRIX.
   a[i][j] = COST OF THE ASSIGNMENT OF ROW  i  TO COLUMN  j .
   INF     = A VERY LARGE INTEGER VALUE, SETTED BY THE USER ACCORDING
             TO THE CHARACTERISTICS OF THE USED MACHINE.
   INF IS THE ONLY MACHINE-DEPENDENT CONSTANT USED AND IT MUST BE STRICTLY GREATER THAN THE MAXIMUM
   ASSIGNMENT COST (E.G., INF MUST BE STRICTLY GREATER THAN THE MAXIMUM VALUE OF THE COST MATRIX a).
   ON RETURN, THE INPUT PARAMETERS ARE UNCHANGED.

   MEANING OF THE OUTPUT PARAMETERS:
   f[i] = COLUMN ASSIGNED TO ROW  i .
   z_p  = COST OF THE OPTIMAL ASSIGNMENT =
   = a[0][f[0]] + a[1][f[1]] + ... + a[n-1][f[n-1]] .

   RETURN VALUE:
   0,  IF THE PROBLEM HAS BEEN SOLVED AND THE OUTPUT PARAMETERS HAVE BEEN PROPERLY SETTED
   -1, IF THE PROBLEM HAS NOT BEEN SOLVED DUE TO A MEMORY ISSUE (NOT ENOUGH AVAILABLE MEMORY)

   ALL THE PARAMETERS ARE INTEGERS.
   VECTOR  f  MUST BE DIMENSIONED AT LEAST AT  n , MATRIX  a
   AT LEAST AT  (n,n) .

   THE CODE IS BASED ON THE HUNGARIAN METHOD AS DESCRIBED BY
   LAWLER (COMBINATORIAL OPTIMIZATION : NETWORKS AND
   MATROIDS, HOLT, RINEHART AND WINSTON, NEW YORK, 1976).
   THE ALGORITHMIC PASCAL-LIKE DESCRIPTION OF THE CODE IS
   GIVEN IN G.CARPANETO, S.MARTELLO AND P.TOTH, ALGORITHMS AND
   CODES FOR THE ASSIGNMENT PROBLEM, ANNALS OF OPERATIONS
   RESEARCH XX, 1988.

   SUBROUTINE APC DETERMINES THE INITIAL DUAL AND PARTIAL
   PRIMAL SOLUTIONS AND THEN SEARCHES FOR AUGMENTING PATHS
   UNTIL ALL ROWS AND COLUMNS ARE ASSIGNED.

   MEANING OF THE MAIN INTERNAL VARIABLES:
   fb[j] = ROW ASSIGNED TO COLUMN  j .
   m     = NUMBER OF INITIAL ASSIGNMENTS.
   u[i]  = DUAL VARIABLE ASSOCIATED WITH ROW  i .
   v[j]  = DUAL VARIABLE ASSOCIATED WITH COLUMN  j .

   APC NEEDS THE FOLLOWING SUBROUTINES: incr
   init
   path

   QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO
   G. CARPANETO, S.MARTELLO AND P.TOTH
   DIPARTIMENTO DI ELETTRONICA, INFORMATICA E SISTEMISTICA
   UNIVERSITA' DI BOLOGNA, VIALE RISORGIMENTO 2
   40136 BOLOGNA (ITALY)

   THIS WORK WAS SUPPORTED BY  C.N.R. , ITALY.

*/
int apc(int n, double *a, double *z_p, int *f)
{
    int i, j, k, m, *fb, *rc, *lr, *uc;
    double *u, *v, *pi;
    double INF = 1.0E9;

    /*
       Memory allocation
       All the memory required by the method and by the subroutines is here allocated
       If not enough memory is available, the method returns -1
       */
    u=(double*)malloc(n*sizeof(double));
    if(u==NULL)
    {
        return -1;
    }
    v=(double*)malloc(n*sizeof(double));
    if(v==NULL)
    {
        free(u);
        u = NULL;
        return -1;
    }
    fb=(int*)malloc(n*sizeof(int));
    if(fb==NULL)
    {
        free(u); free(v);
        u = NULL; v = NULL;
        return -1;
    }
    rc=(int*)malloc(n*sizeof(int));
    if(rc==NULL)
    {
        free(u); free(v); free(fb);
        u = NULL; v = NULL; fb = NULL;
        return -1;
    }
    pi=(double*)malloc(n*sizeof(double));
    if(pi==NULL)
    {
        free(u); free(v); free(fb); free(rc);
        u = NULL; v = NULL; fb = NULL; rc = NULL;
        return -1;
    }
    lr=(int*)malloc(n*sizeof(int));
    if(lr==NULL)
    {
        free(u); free(v); free(fb); free(rc); free(pi);
        u = NULL; v = NULL; fb = NULL; rc = NULL; pi = NULL;
        return -1;
    }
    uc=(int*)malloc(n*sizeof(int));
    if(uc==NULL)
    {
        free(u); free(v); free(fb); free(rc); free(pi); free(lr);
        u = NULL; v = NULL; fb = NULL; rc = NULL; pi = NULL; lr = NULL;
        return -1;
    }

    //SEARCH FOR THE INITIAL DUAL AND PARTIAL PRIMAL SOLUTIONS.
    init(n,a,f,u,v,fb,rc,INF,&m);
    //SOLUTION OF THE REDUCED PROBLEM.
    if(m!=n)
    {
        for(i=0;i<n;i++)
        {
            if(f[i]<0)
            {
                //DETERMINATION OF AN AUGMENTING PATH STARTING FROM ROW  I .
                path(n,a,f,u,v,fb,rc,pi,lr,uc,INF,i,&j);
                //ASSIGNMENT OF ROW  I  AND COLUMN  J .
                incr(f,fb,rc,j);
            }
        }
    }
    //COMPUTATION OF THE SOLUTION COST  Z .
    (*z_p) = 0.0;
    for(k=0;k<n;k++)
    {
        (*z_p)+=(u[k]+v[k]);
    }

    //Free all the allocated memory and return
    free(u); free(v); free(fb); free(rc); free(pi); free(lr); free(uc);
    u = NULL; v = NULL; fb = NULL; rc = NULL; pi = NULL; lr = NULL; uc = NULL;

    return 0;

}//apc()


/*
   ASSIGNMENT OF COLUMN  j .
   */
void incr(int *f,int *fb,int *rc,int j)
{
    int i,jj;

    do
    {
        i=rc[j]; fb[j]=i; jj=f[i]; f[i]=j; j=jj;
    }
    while(j>=0);

}//incr()



/*
        SEARCH FOR THE INITIAL DUAL AND PARTIAL PRIMAL SOLUTIONS.
        p[i] = FIRST UNSCANNED COLUMN OF ROW  i .
        */
void init(int n,double *a,int *f,double *u,double *v,int *fb,int *p,double INF,int *m_p)
{

    int i,j,jmin,k,kk,r,m,skip;
    double ia, min;


    //PHASE 1 .

    m = 0;
    for(k=0;k<n;k++)
    {
        f[k]=fb[k]=-1;
    }

    //SCANNING OF THE COLUMNS ( INITIALIZATION OF  V(J) ).
    for(j=0;j<n;j++) //40
    {
        min=INF;
        for(i=0;i<n;i++) //30
        {
            ia=a[i*n+j];
            if( ia<min || (ia==min && f[i]<0) )
            {
                min=ia; r=i;
            }
        }
        v[j]=min;
        if(f[r]<0)
        {
            //ASSIGNMENT OF COLUMN  J  TO ROW  R .
            m++; fb[j]=r; f[r]=j; u[r]=0; p[r]=j+1;
        }
    }


    //PHASE 2 .

    //SCANNING OF THE UNASSIGNED ROWS ( UPDATING OF  U(I) ).
    for(i=0;i<n;i++) // DO 110
    {
        if(f[i]<0)
        {
            min=INF;
            for(k=0;k<n;k++) //60
            {
                ia=a[i*n+k]-v[k];
                if( ia<min || (ia==min && fb[k]<0 && fb[j]>=0) )
                {
                    min=ia; j=k;
                }
            }

            u[i]=min; jmin=j;
            if(fb[j]>=0)
            {
                skip=0;
                for(j=jmin;j<n && !skip;j++) //80
                {
                    if(a[i*n+j]-v[j] <= min)
                    {
                        r=fb[j]; kk=p[r];
                        if(kk<n)
                        {
                            for(k=kk;k<n && !skip;k++) //70
                            {
                                if(fb[k]<0 && a[r*n+k]-u[r]-v[k]==0)
                                {
                                    //REASSIGNMENT OF ROW  R  AND COLUMN  K .
                                    f[r]=k; fb[k]=r; p[r]=k+1;
                                    //ASSIGNMENT OF COLUMN  J  TO ROW  I .
                                    m++; f[i]=j; fb[j]=i; p[i]=j+1;
                                    skip=1; //skip from the cycles on k and on j avoiding to set p[r]=n
                                }
                            }
                            if(!skip)
                                p[r]=n;
                        }
                    }
                }
            }
            else
            {
                //100
                //ASSIGNMENT OF COLUMN  J  TO ROW  I .
                m++; f[i]=j; fb[j]=i; p[i]=j+1;
            }

        }//if(f[i]<0)

    }

    //END
    (*m_p)=m;

}//init()



/*
   DETERMINATION OF AN AUGMENTING PATH STARTING FROM
   UNASSIGNED ROW  ii  AND TERMINATING AT UNASSIGNED COLUMN
   jj , WITH UPDATING OF DUAL VARIABLES  u[i]  AND  v[j] .

   MEANING OF THE MAIN INTERNAL VARIABLES:
   lr[l] = l-TH LABELLED ROW ( l=0,nlr-1 ).
   pi[j] = MIN ( a[i][j] - u[i] - v[j] , SUCH THAT ROW  i  IS
   LABELLED AND NOT EQUAL TO  fb[j] ).
   rc[j] = ROW PRECEDING COLUMN  j  IN THE CURRENT
   ALTERNATING PATH.
   uc[l] = l-TH UNLABELLED COLUMN ( l=0,nuc-1 ).
   */
void path(int n,double *a,int *f,double *u,double *v,int *fb,int *rc,double *pi,int *lr,int *uc,double INF,int ii,int *jj_p)
{

    int k,j,jj,l,r,nuc,nlr;
    double ia, min;

    //INITIALIZATION.
    lr[0]=ii;
    for(k=0;k<n;k++)
    {
        pi[k]=a[ii*n+k]-u[ii]-v[k];
        rc[k]=ii; uc[k]=k;
    }
    nuc=n; nlr=1;
    goto L40;

    //SCANNING OF THE LABELLED ROWS.
L20:
    r=lr[nlr-1];
    for(l=0;l<nuc;l++)
    {
        j=uc[l];
        ia=a[r*n+j]-u[r]-v[j];
        if(ia<pi[j])
        {
            pi[j]=ia; rc[j]=r;
        }
    }

    //SEARCH FOR A ZERO ELEMENT IN AN UNLABELLED COLUMN.
L40:
    for(l=0;l<nuc;l++)
    {
        j=uc[l];
        if(pi[j]==0) goto L100;
    }

    //UPDATING OF THE DUAL VARIABLES  U(I)  AND  V(J) .
    min=INF;
    for(l=0;l<nuc;l++)
    {
        j=uc[l];
        if(min > pi[j]) min=pi[j];
    }
    for(l=0;l<nlr;l++)
    {
        r=lr[l];
        u[r]+=min;
    }
    for(j=0;j<n;j++)
    {
        if(pi[j]==0)
        {
            v[j]-=min;
        }
        else
        {
            pi[j]-=min;
        }
    }
    goto L40;

L100:
    if(fb[j]<0) goto L110;

    //LABELLING OF ROW  FB(J)  AND REMOVAL OF THE LABEL  OF COLUMN  J .
    nlr++; lr[nlr-1]=fb[j]; uc[l]=uc[nuc-1]; nuc--; goto L20;

    //DETERMINATION OF THE UNASSIGNED COLUMN  J .
L110:
    jj=j;


    (*jj_p)=jj;

}//path()

