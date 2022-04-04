#include<arm_neon.h>
#include<iostream>
#include <sys/time.h>
using namespace std;
void gaussneon(float** matrix,int N)
{
	float32x4_t t1,t2,t3,t4;
	for(int k=0;k<N;k++)
	{
		float diagonal[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
		vld1.32 {t1}, [diagonal, :64];
		for(int j=N-4;j>=k;j-=4)
		{
			vld1.32{t2},[matrix[k]+j,:64];
			t3=vdiv1_f32(t2,t1);
			vst1.32{t3},[matrix[k],:64];
		}
		if(k%4!=(N%4))
		{
			 for(int j=k;j%4!=(N%4);j++)
			 {
				  matrix[k][j]=matrix[k][j]/diagonal[0];
			 }
		}
		for(int j=(N%4)-1;j>=0;j--)
		{
			 matrix[k][j]=matrix[k][j]/diagonal[0];
		}
		for(int i=k+1;i<N;i++)
		{
			 float tmpt[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
			  t1=vld1q_f32(tmpt);
			  for(int j=N-4;j>k;j-=4)
			  {
				  t2=vld1q_f32(matrix[i]+j);
				  t3=vld1q_f32(matrix[k]+j);
				  t4=vsubq_f32( t2,vmulq_f32(t1,t3));
				  vst1q_f32(matrix[i]+j,t4);
			  }
			  for(int j=k+1;j%4!=(N%4);j++)
			  {
				  matrix[i][j]=matrix[i][j]-matrix[i][k]*matrix[k][j];
			  }
			   matrix[i][k]=0;
		}
	}
}
void print(float** matrix,int N)
{
	  for (int i = 0; i < N; i++)
	  {
		   for (int j = 0; j < N; j++)
		   {
			   cout << matrix[i][j] << " ";
		   }
		   cout<<endl;
	  }
}
int main()
{
	//int N[10] = {5,10,50,100,500,1000,2000,3000,4000,5000 };
	int N[10]={1,2,3,4,5,6,7,8,9,10};

	for(int p=0;p<10;p++)
	{
		 srand((int)time(0));
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
		 struct  timeval start;
		 struct  timeval end;
		 unsigned  long diff
		 gettimeofday(&start, NULL);
		 gaussneon(matrix,N[p]);
		 gettimeofday(&end, NULL);
		 diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
		 cout << diff << "us" << endl;
	}
	return 0;
}
