#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
void gaussAVXali(float** matrix,int N)
//两个循环使用SAVX，对齐
{
     __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float __declspec(align(32)) diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        long long int address=(long long int)&matrix[k][0];
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
                // long int address=(long long int)&matrix[k][0];
                float __declspec(align(32)) tmpt[8] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k] };
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
void gaussAVXalif1(float** matrix,int N)
//第一个循环使用AVX，对齐
{
     __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float __declspec(align(32)) diagonal[8]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_set_ps(diagonal[0],diagonal[1],diagonal[2],diagonal[3],diagonal[4],diagonal[5],diagonal[6],diagonal[7]);
        //long long int address=(long long int)&matrix[k][0];
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
        for (int i = k + 1; i <N; i++)
		{
			for (int j = k + 1; j <N; j++)
				matrix[i][j] -= matrix[i][k] * matrix[k][j];
			matrix[i][k] = 0;
		}

    }
    //return matrix;
}


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
                float __declspec(align(32)) tmpt[8] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k] };
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

        float**matrix = reinterpret_cast<float**>(_aligned_malloc(sizeof(float*)*N[p], 32));
        for(int i=0;i<N[p];i++)
            matrix[i]=reinterpret_cast<float*>(_aligned_malloc(sizeof(float)*N[p], 32));

        /*float **matrix;
        matrix= (float**) _mm_malloc(N[p]*sizeof(float), 32);
        for(int i=0;i<N[p];i++)
        {
            matrix[i]= (float*) _mm_malloc(N[p]*sizeof(float), 32);
        }
        */

        for (int i = 0;i < N[p]; i++)
        {
            for (int j = 0; j < N[p]; j++)
            {
                matrix[i][j] = rand() % 100;
            }
        }

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussAVXali(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"两个都对齐AVX"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussAVXalif1(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第1个对齐AVX"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussAVXalif2(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"第2个对齐AVX"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        cout<<endl;

    /*for(int k=0;k<N[p];k++)
        {
            long long int address=(long long int)&matrix[k][0];
            int m=address%32;
            cout<<"N="<<N[p]<<" "<<"k="<<k<<" "<<m<<endl;
        }*/
    }


    system("pause");
    return 0;

}

