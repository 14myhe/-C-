#include <iostream>
#include <string.h>
#include <fstream>//�Դ򿪵��ļ����ж�д���� 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <iomanip> 

#define Data 75 //��150�����ݷ�Ϊ�����֣�75������ѵ������75���������Լ� 
#define In 4 //������������ �������ĸ����ԣ����೤��--Sepal.Length,������--Speal.Width,���곤��--Petal.length��������--Petal.Width�� 
#define Out 3  //�����������
#define Neuron 10 //��Ԫ����
#define TrainC 2000 //ѵ������
#define a 0.1   //ѧϰЧ�� 
using namespace std;
//����һ��flower����4�����Ժ�����������ֵ�ͷ�������ֵ�ķ���

/*
���˼·��
1.�����е��������ݷ�Ϊ�����֣�75��Ϊѵ������
2.���������࣬�ֱ���flower,ModelData,ModelTest,��������֮��Ĺ�ϵ�Ǽ̳С�
 flower--������ 
 ��Ա���԰���
        double SepalLength=0;
		double SpealWidth=0;
		double PetalLength=0;
		double PetalWidth=0;
		int kinds=0;
 ��Ա��������
 	    void setSLength(double slen);
        void setSWidth(double swid);
        void setPWidth(double pwid );
	    void setPLength(double plen);
	    void setkinds(int kind); 
	    double getSLength();
	    double getSWidth();
	    double getPLength();
	    double getPWidth();
	    int getkind();
������  ModelData�̳�flower��
 �������Ա���ԣ�����
        double d_in[Data][In]; //��������
		double d_out[Out][Data]; //�������
		double w[Neuron][In]; //�������Ԫ��Ȩ
		double v[Out][Neuron]; //��Ԫ�������Ȩ
		double dw[Neuron][In];  //w������ 
		double dv[Out][Neuron]; //v������ 
		double o[Neuron]; //��Ԫͨ������������
		double OutputData[Out]; //BP��������
		double Maxin[In],Minin[In],Maxout[Out],Minout[Out]; 
		Flower md[Data];
		
  �������Ա����������
  	   void readData();
	   void InitBPNework();
	   void ComputO(int var);
	   void BackUpdate(int var);
	   void TrainNetwork();
   
������ ModelTest �̳� ModelData
   �������Ա����,����
        void Test();
        
3.readData()----��ȡtrain_data.txt--���������ݵ����Դ����ά����d_in[n][k] --n�������ݵı�ţ�k�Ǵ���ڼ������� 
  ���������ݵ���������ά����d[type][i] --��i�������������ĸ�type
4.InitBPNework()-----��ʼ�������磬��ͨ������ѭ���ҵ��������Ե����ֵ����Сֵ�����ڽ������Ĺ�һ����
5.�÷ֲ����ȵ�����������������ԪȨ�غ���Ԫ�������Ȩ�أ�
Ȩ�����ɵķ����ǣ�-2.4/��Ԫ�ĸ���,2.4/��Ԫ�ĸ�������Ԫ������ֵ������Ϊ0
6.void ComputO(int var)----ͨ��Ӧ������x���������y������򴫲�������  yk(p)=1/(1+exp(-Xk(p)));//ʹ��S�ͼ���� 
   ����yk(p)---��Ԫk�ڵ�p�ε��������  Xk(p)--ͬ�ε�������Ԫk�ľ�Ȩ�ص����� 
   ͨ��S�ͼ������void BackUpdate(int var);---�������Ȩ�ص�ֵ 
7.void TrainNetwork();----����ѧϰ���ʲ�����Ȩ��ѵ��ֱ������ݶȷ���Ҫ�� 
8.void  Test();���ı��ļ�������� ,����Ԥ����ȷ�ĸ��� 
 
*/ 



class Flower{
	public:
		double SepalLength=0;
		double SpealWidth=0;
		double PetalLength=0;
		double PetalWidth=0;
		int kinds=0;
    public:
     void setSLength(double slen);
     void setSWidth(double swid);
     void setPWidth(double pwid );
	 void setPLength(double plen);
	 void setkinds(int kind); 
	 double getSLength();
	 double getSWidth();
	 double getPLength();
	 double getPWidth();
	 int getkind();
}; 

void Flower::setSLength(double slen)
{
	SepalLength=slen;
}
void Flower::setSWidth(double swid)
{
	SpealWidth=swid;
}
void Flower::setPLength(double plen)
{
	PetalLength=plen;
}
void Flower::setPWidth(double pwid)
{
	PetalWidth=pwid;
}
void Flower::setkinds(int i)
{
	kinds=i;
}
double Flower::getSLength()
{
	return SepalLength;
}
double Flower::getSWidth()
{
	return SpealWidth;
}
double Flower::getPLength()
{
	return PetalLength;
}
double Flower::getPWidth()
{
	return PetalWidth;
}
int Flower::getkind()
{
	return kinds;
}
//����������
class ModelData:public Flower
{
	public:
		double d_in[Data][In]; //��������
		double d_out[Out][Data]; //�������
		double w[Neuron][In]; //�������Ԫ��Ȩ
		double v[Out][Neuron]; //��Ԫ�������Ȩ
		double dw[Neuron][In];  //w������ 
		double dv[Out][Neuron]; //v������ 
		double o[Neuron]; //��Ԫͨ������������
		double OutputData[Out]; //BP��������
		double Maxin[In],Minin[In],Maxout[Out],Minout[Out]; 
		Flower md[Data];
	public:
		void readData();
		void InitBPNework();
		void ComputO(int var);
		void BackUpdate(int var);
		void TrainNetwork();

};
class ModelTest:public ModelData{
	public:
		void Test();
};

//��ȡ�ļ�����Ϣ
void ModelData::readData()
{
	char buffer[256];
    ifstream in("train_data.txt");
    if(!in.is_open())
    {
    	cout<<"error file"<<endl;
    }
    int i=0;
    char *sign=",";
    char *p;
    int type=0;
    double exm=0;
    while(!in.eof())
    {
    	   in.getline(buffer,100);
    	   //���зָ� 
    	   //cout<<buffer<<endl;
    	   //1.���೤��
    	   p=strtok(buffer,sign);
    	   sscanf(p,"%lf",&exm);
    	   md[i].setSLength(exm);
    	   d_in[i][0]=md[i].getSLength();
    	   //2.������  
    	   p=strtok(NULL,sign);
    	   sscanf(p,"%lf",&exm);
    	  // cout<<exm<<endl;
    	   md[i].setSWidth(exm);
    	   d_in[i][1]=md[i].getSWidth();
    	   //3.���곤��
    	   p=strtok(NULL,sign);
    	   sscanf(p,"%lf",&exm);
    	   //cout<<exm<<endl;
    	   md[i].setPLength(exm);
    	   d_in[i][2]=md[i].getPLength();
    	   // 4.������ 
    	   p=strtok(NULL,sign);
    	   sscanf(p,"%lf",&exm);
    	   //cout<<exm<<endl;
    	   md[i].setPWidth(exm);
    	   d_in[i][3]=md[i].getPWidth();

           d_out[0][i]=0;
           d_out[1][i]=0;
           d_out[2][i]=0;
    	   p=strtok(NULL,sign);
    	   if(strcmp(p,"Iris-setosa")==0)
		     {
			       md[i].setkinds(0);
		     }
		     else if(strcmp(p,"Iris-versicolor")==0)
		     {
			       md[i].setkinds(1);
		     }
		     else if(strcmp(p,"Iris-virginica")==0)
		     {
			      md[i].setkinds(2);
		     }
		     
		     type=md[i].getkind();
             d_out[type][i]=1;//type��ֵ�ܹ����֣�0��1��2 
             
		     i++;
    }
    

}
//��ʼ�������� 
void ModelData::InitBPNework()
{
	
    int i,j;
    
    for (i = 0; i < In; i++)
    {
    	// Minin[i]��Maxin[i]����ʼ��Ϊd_in[0][i],���ѵ�һ���������ݵĸ��������Ծ���Ϊ��ʼֵ 
        Minin[i]=Maxin[i]=d_in[0][i];
        for (j = 0; j < Data; j++)//ѭ��ȫ����ѵ�������� 
        {
            Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];
            Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];
        }
    }
    //�������ѭ����õ������ÿ�����Զ�Ӧ�������СֵMaxin[i]��Minin[i]
    for (i = 0; i < Out; i++)
    {
        Minout[i]=Maxout[i]=d_out[0][i];
        for (j = 0; j < Data; j++)
        {
            Maxout[i]=Maxout[i]>d_out[i][j]?Maxout[i]:d_out[i][j];
            Minout[i]=Minout[i]<d_out[i][j]?Minout[i]:d_out[i][j];
        }
    }
 
 //��һ������ ���䷶Χ[0,1] 
    for (i = 0; i < In; i++)
        for(j = 0; j < Data; j++)
    	  	d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);
    
    for (i = 0; i < Out; i++)
        for(j = 0; j < Data; j++)
            d_out[i][j]=(d_out[i][j]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);
 
    for (i = 0; i < Neuron; ++i)  
        for (j = 0; j < In; ++j)
        {   
            w[i][j]=(float)((-rand()%49+24)*1.0/100);//�������Ȩ��ֵ (-2.4/Fi,2.4/Fi)
            //cout<<"w="<<w[i][j]<<endl; 
            dw[i][j]=0;
        }

    for (i = 0; i < Neuron; ++i)   
        for (j = 0; j < Out; ++j)
        {
            v[j][i]=(float)((-rand()%49+24)*1.0/100);
            dv[j][i]=0;
        }
}
//���� 
void ModelData::ComputO(int var)
{ 
    int i,j;
    double sum,y;
    //�������ʵ����� 
    for (i = 0; i < Neuron; ++i)
    {
        sum=0;
        for (j = 0; j < In; ++j)
        {
            sum+=w[i][j]*d_in[var][j]-(float)((-rand()%49+24)*1.0/100);//�Ե�i����Ԫ���㾻Ȩ������ 
            o[i]=1/(1+exp(-1*sum));//ʹ��S�ͼ���� ȷ����� 
        }
    }
     
    for (i = 0; i < Out; ++i)
    {
        sum=0;
        for (j = 0; j < Neuron; ++j)
            sum+=v[i][j]*o[j]-(float)((-rand()%49+24)*1.0/100);//����������ʵ������ 
            OutputData[i]=sum;
    }   
}
//Ȩ��ѵ�� 
void ModelData::BackUpdate(int var)
{
    int i,j;
    double t;
    //��������� 
    for (i = 0; i < Neuron; ++i)
    {
        t=0;
        for (j = 0; j < Out; ++j)
        {
        	t+=(OutputData[j]-d_out[j][var])*v[j][i];
            //���������ԪȨ�ص�У��ֵ 
            dv[j][i]=a*dv[j][i]+a*(OutputData[j]-d_out[j][var])*o[i];
            v[j][i]-=dv[j][i];
        }
        //������������Ԫ��У��ֵ 
        for (j = 0; j < In; ++j)
        {
            dw[i][j]=a*dw[i][j]+a*t*o[i]*(1-o[i])*d_in[var][j];
            w[i][j]-=dw[i][j];
        }
    }
}

//ѵ�������� ������ 
void ModelData::TrainNetwork()
{
    int c=0;
    double e=0;
    do
    {
        e=0;
        for (int i = 0; i < Out; i++)
            for (int j = 0; j < Data; j++)
            {
                ComputO(j);//���� 
                e+=fabs((OutputData[i]-d_out[i][j])/d_out[i][j]);
                BackUpdate(j);
            }
    c++;
    }while(c<TrainC && e/Data>0.01);//2000�ε��� 
}
void ModelTest::Test()
{
	char buf[256];
    int i,j,typeI=0,typeO=0,trueAns=0;
    double sum,y,data[In],ans[Out];
    ifstream file("test_data.txt");
    if(!file.is_open())
    {
    	   cout<<"error file"<<endl;
    }
    int k=0;
    char *sign1=",";
    char *p1;
    while(!file.eof())
    {
    	   file.getline(buf,100);
         //	cout<<buf<<endl;
    	   //���зָ�
		   p1=strtok(buf,sign1);
           sscanf(p1,"%lf",&data[0]);
           for(int k=1;k<=3;k++)
           {
              p1=strtok(NULL,sign1);
              sscanf(p1,"%lf",&data[k]);
         }
         p1=strtok(NULL,sign1);
       // cout<<p1<<endl;
         if(strcmp(p1,"Iris-setosa")==0)
		 {
			    typeI=0;
		 }
		 else if(strcmp(p1,"Iris-versicolor")==0)
		 {
			    typeI=1;
		 }
		 else if(strcmp(p1,"Iris-virginica")==0)
		 {
			    typeI=2;
		 }
        
	       for (i = 0; i < In; i++)
             data[i]=(data[i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1); 
             for (i = 0; i < Neuron; i++)
             {
                sum=0;
                for (j = 0; j < In; j++)
                {
                    sum+=w[i][j]*data[j];
                    o[i]=1/(1+exp(-1*sum));
                }
            }

         for (i = 0; i < Out; i++)
         {
              sum=0;
              for (j = 0; j < Neuron; j++)
              sum+=v[i][j]*o[j];
              OutputData[i]=sum;
         }

         for (i = 0; i < Out; i++)
              ans[i]=(OutputData[i]*(Maxout[i]-Minout[i]+1)+Minout[i]-1);
              typeO=(ans[0]>ans[1])?(ans[0]>ans[2]?0:2):(ans[1]>ans[2]?1:2);
              cout<<"0:"<<setprecision(6)<<std::fixed<<ans[0]<<" 1:"<<setprecision(6)<<std::fixed<<ans[1]<<"2:"<<setprecision(6)<<std::fixed<<ans[2]<<"   Ԥ�����ࣺ"<<typeO<<"  ʵ�����ࣺ"<<typeI<<endl;
              if(typeI==typeO)
                  trueAns+=1;
    }
           
   
    cout<<"Ԥ����ȷ��:"<<trueAns*1.0/Data*100<<"%"<<endl;
}

int main()
{
	srand(time(0));
	ModelTest m;
	m.readData();
	m.InitBPNework();
	m.TrainNetwork();
	m.Test();
	system("pause");
	return 0;
}



