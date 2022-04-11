#include <iostream>
#include <time.h>
#include<sys/time.h>
#include<unistd.h>
#include <pmmintrin.h>
#include<stdlib.h>
using namespace std;
void gaussSSEali(float** matrix,int N)
//两个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_load_ps(diagonal);
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

			float tmpt[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
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
}
void gaussSSEalif1(float** matrix,int N)
//第一个循环使用SSE，对齐
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
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
        for (int i = k + 1; i <N; i++)
		{
			for (int j = k + 1; j <N; j++)
				matrix[i][j] -= matrix[i][k] * matrix[k][j];

			matrix[i][k] = 0;
		}
    }
    //return matrix;
}
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
			float  tmpt[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
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
	float**matrix = reinterpret_cast<float**>(_mm_malloc(sizeof(float*)*N[p], 16));
         for(int i=0;i<N[p];i++)
            matrix[i]=reinterpret_cast<float*>(_mm_malloc(sizeof(float)*N[p], 16));

        for (int i = 0; i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }
        gettimeofday(&start,NULL);
        gaussSSEali(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐1、2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;

        gettimeofday(&start,NULL);
        gaussSSEalif1(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐1"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;

        gettimeofday(&start,NULL);
        gaussSSEalif2(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;
        cout<<endl;
    }
    
    return 0;

}


