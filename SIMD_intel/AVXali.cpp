#include <iostream>
#include <time.h>
#include <pmmintrin.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<unistd.h>
#include <immintrin.h>
using namespace std;
/*
void gaussAVXali(float** matrix,int N)
//两个循环使用SAVX，对齐
{
     __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        int j=k;
        for(j=k;j%8!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
        //剩下的是对齐的部分
        int p=j;
        for (p =j; p+8<N; p += 8)
        {
            t2 = _mm256_load_ps(matrix[k] + p);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(matrix[k] + p, t3);
        }
        //手动把结尾不对齐的元素处理掉
        for (int q=p;q<N;q++)
        {
            matrix[k][q] = matrix[k][q] / diagonal[0];
        }
			for (int i = k + 1; i < N; i++)
            {
                float  tmpt[8] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k] };
                t1 = _mm256_set_ps(tmpt[0],tmpt[1],tmpt[2],tmpt[3],tmpt[4],tmpt[5],tmpt[6],tmpt[7]);
                int j=k;
                for(j=k;j%8!=0;j++)
                {
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                }
                //剩下的是对齐的部分
                int p=j;
                for (p =j; p+8<N; p += 8)
                {
                    t2 = _mm256_load_ps(matrix[i] + p);
                    t3 = _mm256_load_ps(matrix[k] + p);
                    t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3));
                    _mm256_store_ps(matrix[i] + p, t4);
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
*/
void gaussAVXalif1(float** matrix,int N)
//第一个循环使用AVX，对齐
{
     __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        int j=k;
        for(j=k;j%8!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
        //剩下的是对齐的部分
        int p=j;
        for (p =j; p+8<=N; p += 8)
        {
            t2 = _mm256_load_ps(matrix[k] + p);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(matrix[k] + p, t3);
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

/*
void gaussAVXalif2(float** matrix,int N)
//第二个循环使用AVX，对齐
{
     __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal=matrix[k][k];
        for (int j = k + 1; j < N; j++)
			matrix[k][j] = matrix[k][j] / diagonal;
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
            {
                // long int address=(long long int)&matrix[k][0];
                float tmpt[8] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k] };
                t1 = _mm256_set_ps(tmpt[0],tmpt[1],tmpt[2],tmpt[3],tmpt[4],tmpt[5],tmpt[6],tmpt[7]);
                int j=k;
                for(j=k;j%8!=0;j++)
                {
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                }
                //剩下的是对齐的部分
                int p=j;
                for (p =j; p+8<N; p += 8)
                {
                    t2 = _mm256_load_ps(matrix[i] + p);
                    t3 = _mm256_load_ps(matrix[k] + p);
                    t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3));
                    _mm256_store_ps(matrix[i] + p, t4);
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
*/

int main()
{
    int N[10] = {5,10,50,100,500,1000,2000,3000,4000,5000 };
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
    for(int p=0;p<10;p++)
    {
        long long head, tail, freq;

        float**matrix = reinterpret_cast<float**>(_mm_malloc(sizeof(float*)*N[p], 32));
        for(int i=0;i<N[p];i++)
            matrix[i]=reinterpret_cast<float*>(_mm_malloc(sizeof(float)*N[p], 32));
        for (int i = 0;i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }
/*
        gettimeofday(&start,NULL);
        gaussAVXali(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐1、2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;
*/
        gettimeofday(&start,NULL);
        gaussAVXalif1(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐1"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;
/*
        gettimeofday(&start,NULL);
        gaussAVXalif2(matrix,N[p]);
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        cout<<"对齐2"<<" "<<"N="<<N[p]<<" "<<diff<<"us"<<endl;
*/
        cout<<endl;


    }
    return 0;

}


