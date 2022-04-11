#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
using namespace std;
float** gaussSSEf1(float** matrix,int N)
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
    return matrix;
}
float** gaussSSEf2(float** matrix,int N)
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
    return matrix;
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
    int N[10] = {5,10,50,100,500,1000,2000,3000,4000,5000 };
    for(int p=0;p<10;p++)
    {
        long long head, tail, freq;
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
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** M = gaussSSEf1(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"只第1个"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** P= gaussSSEf2(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"只第2个"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        cout<<endl;
        //print(M,N[p]);
    }
    system("pause");
    return 0;

}

