#include <iostream>
#include <time.h>
#include <windows.h>
using namespace std;
float** gauss(float** matrix,int N)
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
    return matrix;
}
float** gausscache(float** matrix,int N)
//cache
{
    for (int k = 0; k < N; k++)
    {
        float  diagonal = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / diagonal;
        }
        for (int j = k + 1; j < N; j++)
        {
            for (int i = k + 1; i < N; i++)
            {
                float tmpt = matrix[i][k];
                matrix[i][j] = matrix[i][j] - tmpt * matrix[k][j];
                matrix[i][k] = 0;
            }

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
    //int N[10]={1,2,3,4,5,6,7,8,9,10};
    for(int p=0;p<10;p++)
    {
        long long head, tail, freq;
        //srand((int)time(0));
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
        gauss(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"普通串行"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        gausscache(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<"cache优化"<<" "<<"N="<<N[p]<<" "<<(tail-head)*1000.0 / freq<<"ms"<<endl;


    }


    system("pause");
    return 0;

}

