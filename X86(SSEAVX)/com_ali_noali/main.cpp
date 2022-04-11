#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
void gaussSSEali(float** matrix,int N)
//����ѭ��ʹ��SSE������
{
     __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float __declspec(align(16)) diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        //__declspec(align(16))ȷ������������׵�ַ����16�ֽ�
        t1 = _mm_load_ps(diagonal);
        //64λϵͳnew��ʱ���Զ�����16�ֽڣ�����ÿ�е���Ԫ��16�ֽڶ���
        //�ֶ��ѿ�ͷ�������Ԫ�ش����
        int j=k;
        for(j=k;j%4!=0;j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal[0];
        }
		//ʣ�µ��Ƕ���Ĳ���
		int p=j;
        for (p =j; p+4<N; p += 4)
		{
			t2 = _mm_load_ps(matrix[k] + p);
			t3 = _mm_div_ps(t2, t1);
			_mm_store_ps(matrix[k] + p, t3);
		}
        //�ֶ��ѽ�β�������Ԫ�ش����
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
            //ʣ�µ��Ƕ���Ĳ���
            int p=j;
            for (p =j; p+4<N; p += 4)
            {
                t2 = _mm_load_ps(matrix[i] + p);
				t3 = _mm_load_ps(matrix[k] + p);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_store_ps(matrix[i] + p, t4);
            }
            //�ֶ��ѽ�β�������Ԫ�ش����
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
//SSE����ѭ���Ż��������루����д��
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
            t3=_mm_div_ps(t2,t1); //ִ�ж�λ����
            _mm_storeu_ps(matrix[k]+j,t3);
        }
        //����в��ܱ�4����������д���
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
            //����в��ܱ�4����������д���
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
//SSE����ѭ���Ż��������루���Ǻ�ԭ���Ĵ��벻̫һ��������Ϊ�˽�һ���Ƚϲ����롢���������𣬲��úͶ���һ�����ӵĴ��룩
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float diagonal[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        //__declspec(align(16))ȷ������������׵�ַ����16�ֽ�
        t1 = _mm_loadu_ps(diagonal);
        //64λϵͳnew��ʱ���Զ�����16�ֽڣ�����ÿ�е���Ԫ��16�ֽڶ���
        //�ֶ��ѿ�ͷ�������Ԫ�ش����
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
//��ӡ����
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
        cout<<"����SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSEasali(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"������SSE(����ʽ)"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gaussSSE(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"������SSE"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        cout<<endl;
    }

    system("pause");
    return 0;

}

