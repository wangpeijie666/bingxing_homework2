#include <iostream>
#include <time.h>
#include<sys/time.h>
#include<unistd.h>
#include <pmmintrin.h>
#include<stdlib.h>
using namespace std;
void gaussserial(float** matrix,int N)
//串行高斯消去
{
    for (int k = 0; k < N; k++)
    {
        float  diagonal = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal;
        }
        for (int i = k + 1; i < N; i++)
        {
            float tmpt = matrix[i][k];
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmpt * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
void gaussSSE(float** matrix,int N)
//并行SSE，不对齐，两个循环都有优化
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(diagonal);
        int j;
        for(j=k;j<=N-4;j+=4)
        {
            t2=_mm_loadu_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1); //执行对位除法
            _mm_storeu_ps(matrix[k]+j,t3);
        }
        //如果有不能被4整除的最后串行处理
        if(j<N)
        {
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/diagonal[0];
            }
        }

        for(int i=k+1;i<N;i++)
        {
            float tmpt[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(tmpt);
            for(j=k+1;j<=N-4;j+=4)
            {
                t2=_mm_loadu_ps(matrix[i]+j);
                t3=_mm_loadu_ps(matrix[k]+j);
                t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                _mm_storeu_ps(matrix[i]+j,t4);
            }
            if(j<N)
            //如果有不能被4整除的最后串行处理
            {
                for(;j<N;j++){
                    matrix[i][j]=matrix[i][j]-tmpt[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0;

        }
    }
    
}
void gaussSSEf1(float** matrix,int N)
//并行SSE，不对齐，第一个循环有优化
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(diagonal);
        int j;
        for(j=k;j<=N-4;j+=4)
        {
            t2=_mm_loadu_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1);
            _mm_storeu_ps(matrix[k]+j,t3);
        }
        if(j<N)
        {
            for(;j<N;j++)
            {
                matrix[k][j]=matrix[k][j]/diagonal[0];
            }
        }

        for(int i=k+1;i<N;i++)
        {
            float tmpt = matrix[i][k];
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmpt * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
void gaussSSEf2(float** matrix,int N)
//并行SSE，不对齐，第二个循环有优化
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  diagonal = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal;
        }

        for(int i=k+1;i<N;i++)
        {
            float tmpt[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(tmpt);
            int j;
            for(j=k+1;j<=N-4;j+=4)
            {
                t2=_mm_loadu_ps(matrix[i]+j);
                t3=_mm_loadu_ps(matrix[k]+j);
                t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                _mm_storeu_ps(matrix[i]+j,t4);
            }
            if(j<N)
            {
                for(;j<N;j++)
                {
                    matrix[i][j]=matrix[i][j]-tmpt[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0;

        }
    }
}
void print(float** matrix,int N)
//打印矩阵
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
int main()
{
    int N[10] = {5,10,50,100,500,1000,2000,3000,4000,5000};
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
    for(int p=0;p<10;p++)
    {
        //初始化矩阵并生成一个随机数矩阵
        float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        for (int i = 0; i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }
 
        gettimeofday(&start,NULL);
        gaussserial(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"串行"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;

        gettimeofday(&start,NULL);
        gaussSSE(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"不对齐1、2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;

        gettimeofday(&start,NULL);
        gaussSSEf1(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"不对齐1"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;

        gettimeofday(&start,NULL);
        gaussSSEf2(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"不对齐2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;
        cout<<endl;
       
    }
    
    return 0;

}

