#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
void gaussSSEali(float** matrix,int N)
//两个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float __declspec(align(16)) diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        //__declspec(align(16))确保定义的数组首地址对齐16字节
        t1 = _mm_load_ps(diagonal);
        //64位系统new的时候自动对齐16字节，数组每行的首元素16字节对齐
        //手动把开头不对齐的元素处理掉
        int j=k;
        for(j=k;j%4!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
		//剩下的是对齐的部分
		int p=j;
        for (p =j; p+4<N; p += 4)
		{
			t2 = _mm_load_ps(matrix[k] + p);
			t3 = _mm_div_ps(t2, t1);
			_mm_store_ps(matrix[k] + p, t3);
		}
        //手动把结尾不对齐的元素处理掉
        for (int q=p;q<N;q++)
		{
			matrix[k][q] = matrix[k][q] / diagonal[0];
		}
        for (int i = k + 1; i < N; i++)
		{

			float __declspec(align(16)) tmpt[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
			t1 = _mm_load_ps(tmpt);
			int j=k;
            for(j=k;j%4!=0;j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            //剩下的是对齐的部分
            int p=j;
            for (p =j; p+4<N; p += 4)
            {
                t2 = _mm_load_ps(matrix[i] + p);
				t3 = _mm_load_ps(matrix[k] + p);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_store_ps(matrix[i] + p, t4);
            }
            //手动把结尾不对齐的元素处理掉
            for (int q=p;q<N;q++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
			matrix[i][k] = 0;
		}
    }
    //return matrix;
}
void gaussSSE(float** matrix, int N)
//SSE两处循环优化，不对齐（正常写）
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
    //return matrix;
}
void gaussSSEasali(float** matrix, int N)
//SSE两处循环优化，不对齐（但是和原来的代码不太一样，这里为了进一步比较不对齐、对齐间的区别，采用和对齐一样复杂的代码）
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        //__declspec(align(16))确保定义的数组首地址对齐16字节
        t1 = _mm_loadu_ps(diagonal);
        //64位系统new的时候自动对齐16字节，数组每行的首元素16字节对齐
        //手动把开头不对齐的元素处理掉
        int j=k;
        for(j=k;j%4!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
		int p=j;
        for (p =j; p+4<N; p += 4)
		{
			t2 = _mm_loadu_ps(matrix[k] + p);
			t3 = _mm_div_ps(t2, t1);
			_mm_storeu_ps(matrix[k] + p, t3);
		}
        for (int q=p;q<N;q++)
		{
			matrix[k][q] = matrix[k][q] / diagonal[0];
		}
        for (int i = k + 1; i < N; i++)
		{

			float __declspec(align(16)) tmpt[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
			t1 = _mm_loadu_ps(tmpt);
			int j=k;
            for(j=k;j%4!=0;j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            int p=j;
            for (p =j; p+4<N; p += 4)
            {
                t2 = _mm_loadu_ps(matrix[i] + p);
				t3 = _mm_loadu_ps(matrix[k] + p);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_storeu_ps(matrix[i] + p, t4);
            }
            for (int q=p;q<N;q++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
			matrix[i][k] = 0;
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
    //int N[10]={8,16,32,64,128,256,512,1024,2048,4096};
    for(int p=0;p<10;p++)
    {
        long long head, tail, freq;

        float**matrix = reinterpret_cast<float**>(_aligned_malloc(sizeof(float*)*N[p], 32));
        for(int i=0;i<N[p];i++)
            matrix[i]=reinterpret_cast<float*>(_aligned_malloc(sizeof(float)*N[p], 32));
        for (int i = 0;i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEali(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"对齐SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEasali(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"不对齐SSE(带格式)"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSE(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"不对齐SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        cout<<endl;
    }

    system("pause");
    return 0;

}

