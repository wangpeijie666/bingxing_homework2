#include <iostream>
#include <time.h>
#include <windows.h>
#include <immintrin.h>
using namespace std;
void gaussAVX(float** matrix,int N)
//并行AVX，不对齐，两个循环都有优化
{
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        int j;
        for(j=k;j<=N-8;j+=8)
        {
            t2=_mm256_loadu_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //执行对位除法
            _mm256_storeu_ps(matrix[k]+j,t3);
        }
        //如果有不能被4整除的最后串行处理
        if(j<N)
        {
            for(;j<N;j++)
            {
                matrix[k][j]=matrix[k][j]/diagonal[0];
            }
        }

        for(int i=k+1;i<N;i++)
        {
            float tmpt[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1=_mm256_set_ps(tmpt[0],tmpt[1],tmpt[2],tmpt[3],tmpt[4],tmpt[5],tmpt[6],tmpt[7]);
            for(j=k+1;j<=N-8;j+=8)
            {
                t2=_mm256_loadu_ps(matrix[i]+j);
                t3=_mm256_loadu_ps(matrix[k]+j);
                t4=_mm256_sub_ps(t2,_mm256_mul_ps(t1,t3));
                _mm256_storeu_ps(matrix[i]+j,t4);
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
void gaussAVXf1(float** matrix,int N)
//并行AVX，不对齐，第一个优化
{

    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        int j;
        for(j=k;j<=N-8;j+=8)
        {
            t2=_mm256_loadu_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //执行对位除法
            _mm256_storeu_ps(matrix[k]+j,t3);
        }
        //如果有不能被4整除的最后串行处理
        if(j<N)
        {
            for(;j<N;j++)
            {
                matrix[k][j]=matrix[k][j]/diagonal[0];
            }
        }

        for (int i = k + 1; i <N; i++)
		{
			for (int j = k + 1; j <N; j++)
				matrix[i][j] -= matrix[i][k] * matrix[k][j];

			matrix[i][k] = 0;
		}
    }

}

void gaussAVXf2(float** matrix,int N)
//并行AVX，不对齐，第二个优化
{
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal=matrix[k][k];
        for (int j = k + 1; j < N; j++)
			matrix[k][j] = matrix[k][j] / diagonal;
		matrix[k][k] = 1.0;

        for(int i=k+1;i<N;i++)
        {
            float tmpt[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1=_mm256_set_ps(tmpt[0],tmpt[1],tmpt[2],tmpt[3],tmpt[4],tmpt[5],tmpt[6],tmpt[7]);
            int j;
            for(j=k+1;j<=N-8;j+=8)
            {
                t2=_mm256_loadu_ps(matrix[i]+j);
                t3=_mm256_loadu_ps(matrix[k]+j);
                t4=_mm256_sub_ps(t2,_mm256_mul_ps(t1,t3));
                _mm256_storeu_ps(matrix[i]+j,t4);
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
        gaussAVX(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"两处AVX "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussAVXf1(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第一处 "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussAVXf1(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第二处 "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

    }
    system("pause");
    return 0;

}

