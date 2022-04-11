#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
/*
void gaussSSEali(float** matrix,int N)
//两个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_load_ps(diagonal);
        int j=k;
        for(j=k;j%4!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
		int p=j;
        for (p =j; p+4<N; p += 4)
		{
			t2 = _mm_load_ps(matrix[k] + p);
			t3 = _mm_div_ps(t2, t1);
			_mm_store_ps(matrix[k] + p, t3);
		}
        for (int q=p;q<N;q++)
		{
			matrix[k][q] = matrix[k][q] / diagonal[0];
		}
        for (int i = k + 1; i < N; i++)
		{

			float tmpt[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
			t1 = _mm_load_ps(tmpt);
			int j=k;
            for(j=k;j%4!=0;j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            int p=j;
            for (p =j; p+4<N; p += 4)
            {
                t2 = _mm_load_ps(matrix[i] + p);
				t3 = _mm_load_ps(matrix[k] + p);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_store_ps(matrix[i] + p, t4);
            }
            for (int q=p;q<N;q++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
			matrix[i][k] = 0;
		}
    }
}*/
void gaussSSEalif1(float** matrix,int N)
//第一个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3]);
        int j;
        for(j=k;j%4!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
		int p=j;
        for (p =j; p<=N-4; p += 4)
		{
			t2 = _mm_load_ps(matrix[k] + p);
			t3 = _mm_div_ps(t2, t1);
			_mm_store_ps(matrix[k] + p, t3);
		}
        if(p<N)
        {
            for(;p<N;p++)
            {
                matrix[k][p]=matrix[k][p]/diagonal[0];
            }
        }
        for (int i = k + 1; i <N; i++)
		{
			for (int j = k + 1; j <N; j++)
				matrix[i][j] -= matrix[i][k] * matrix[k][j];

			matrix[i][k] = 0;
		}
    }
    //return matrix;
}
/*
void gaussSSEalif2(float** matrix,int N)
//第二个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal=matrix[k][k];
        for (int j = k + 1; j < N; j++)
			matrix[k][j] = matrix[k][j] / diagonal;
		matrix[k][k] = 1.0;
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
}*/
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
    //int N[10]={4,8,16,32,128,256,512,1024,2048,4096};
    for(int p=0;p<10;p++)
    {
        long long head, tail, freq;
        float**matrix= new float* [N[p]];
        for(int i=0;i<N[p];i++)
        {
            matrix[i]=new float[N[p]];
        }
        for (int i = 0;i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }
        /*QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEali(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"两个都对齐SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
*/
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEalif1(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第1个对齐SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
/*
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEalif2(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第2个对齐SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        */
        cout<<endl;

    }

    system("pause");
    return 0;

}

