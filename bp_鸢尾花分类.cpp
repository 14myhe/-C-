#include <iostream>
#include <string.h>
#include <fstream>//对打开的文件进行读写操作 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <iomanip> 

#define Data 75 //将150个数据分为两部分，75个用做训练集，75个用作测试集 
#define In 4 //样本参数个数 （输入四个属性：花萼长度--Sepal.Length,花萼宽度--Speal.Width,花瓣长度--Petal.length、花瓣宽度--Petal.Width） 
#define Out 3  //输出参数个数
#define Neuron 10 //神经元个数
#define TrainC 2000 //训练次数
#define a 0.1   //学习效率 
using namespace std;
//声明一个flower类有4个属性和其设置属性值和返回属性值的方法

/*
解决思路：
1.将所有的样本数据分为两部分，75个为训练数据
2.建立三个类，分别是flower,ModelData,ModelTest,他们三者之间的关系是继承。
 flower--基本类 
 成员属性包括
        double SepalLength=0;
		double SpealWidth=0;
		double PetalLength=0;
		double PetalWidth=0;
		int kinds=0;
 成员方法包括
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
派生类  ModelData继承flower类
 除基类成员属性，还有
        double d_in[Data][In]; //样本输入
		double d_out[Out][Data]; //样本输出
		double w[Neuron][In]; //输入对神经元的权
		double v[Out][Neuron]; //神经元对输出的权
		double dw[Neuron][In];  //w的修正 
		double dv[Out][Neuron]; //v的修正 
		double o[Neuron]; //神经元通过激活函数的输出
		double OutputData[Out]; //BP网络的输出
		double Maxin[In],Minin[In],Maxout[Out],Minout[Out]; 
		Flower md[Data];
		
  除基类成员方法，还有
  	   void readData();
	   void InitBPNework();
	   void ComputO(int var);
	   void BackUpdate(int var);
	   void TrainNetwork();
   
派生类 ModelTest 继承 ModelData
   除基类成员方法,还有
        void Test();
        
3.readData()----读取train_data.txt--将样本数据的属性存入二维数组d_in[n][k] --n代表数据的编号，k是代表第几个属性 
  将样本数据的输出存入二维数组d[type][i] --第i个数据组属于哪个type
4.InitBPNework()-----初始化神经网络，先通过两个循环找到各个属性的最大值和最小值，便于接下来的归一化！
5.用分布均匀的随机数设置输入对神经元权重和神经元对输出的权重，
权重生成的方法是（-2.4/神经元的个数,2.4/神经元的个数）神经元的修正值先设置为0
6.void ComputO(int var)----通过应用输入x和期望输出y来激活方向传播神经网络  yk(p)=1/(1+exp(-Xk(p)));//使用S型激活函数 
   其中yk(p)---神经元k在第p次迭代的输出  Xk(p)--同次迭代中神经元k的净权重的输入 
   通过S型激活函数，void BackUpdate(int var);---更正输出权重的值 
7.void TrainNetwork();----假设学习速率参数，权重训练直到误差梯度符合要求 
8.void  Test();从文本文件读入测试 ,计算预测正确的概率 
 
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
//样本数据类
class ModelData:public Flower
{
	public:
		double d_in[Data][In]; //样本输入
		double d_out[Out][Data]; //样本输出
		double w[Neuron][In]; //输入对神经元的权
		double v[Out][Neuron]; //神经元对输出的权
		double dw[Neuron][In];  //w的修正 
		double dv[Out][Neuron]; //v的修正 
		double o[Neuron]; //神经元通过激活函数的输出
		double OutputData[Out]; //BP网络的输出
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

//读取文件的信息
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
    	   //进行分割 
    	   //cout<<buffer<<endl;
    	   //1.花萼长度
    	   p=strtok(buffer,sign);
    	   sscanf(p,"%lf",&exm);
    	   md[i].setSLength(exm);
    	   d_in[i][0]=md[i].getSLength();
    	   //2.花萼宽度  
    	   p=strtok(NULL,sign);
    	   sscanf(p,"%lf",&exm);
    	  // cout<<exm<<endl;
    	   md[i].setSWidth(exm);
    	   d_in[i][1]=md[i].getSWidth();
    	   //3.花瓣长度
    	   p=strtok(NULL,sign);
    	   sscanf(p,"%lf",&exm);
    	   //cout<<exm<<endl;
    	   md[i].setPLength(exm);
    	   d_in[i][2]=md[i].getPLength();
    	   // 4.花瓣宽度 
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
             d_out[type][i]=1;//type的值总共三种，0、1、2 
             
		     i++;
    }
    

}
//初始化神经网络 
void ModelData::InitBPNework()
{
	
    int i,j;
    
    for (i = 0; i < In; i++)
    {
    	// Minin[i]，Maxin[i]均初始化为d_in[0][i],即把第一行样本数据的各个数属性均作为初始值 
        Minin[i]=Maxin[i]=d_in[0][i];
        for (j = 0; j < Data; j++)//循环全部的训练集数据 
        {
            Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];
            Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];
        }
    }
    //做完这个循环会得到输入的每个属性对应的最大最小值Maxin[i]，Minin[i]
    for (i = 0; i < Out; i++)
    {
        Minout[i]=Maxout[i]=d_out[0][i];
        for (j = 0; j < Data; j++)
        {
            Maxout[i]=Maxout[i]>d_out[i][j]?Maxout[i]:d_out[i][j];
            Minout[i]=Minout[i]<d_out[i][j]?Minout[i]:d_out[i][j];
        }
    }
 
 //归一化数据 区间范围[0,1] 
    for (i = 0; i < In; i++)
        for(j = 0; j < Data; j++)
    	  	d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);
    
    for (i = 0; i < Out; i++)
        for(j = 0; j < Data; j++)
            d_out[i][j]=(d_out[i][j]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);
 
    for (i = 0; i < Neuron; ++i)  
        for (j = 0; j < In; ++j)
        {   
            w[i][j]=(float)((-rand()%49+24)*1.0/100);//随机设置权重值 (-2.4/Fi,2.4/Fi)
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
//激活 
void ModelData::ComputO(int var)
{ 
    int i,j;
    double sum,y;
    //隐含层的实际输出 
    for (i = 0; i < Neuron; ++i)
    {
        sum=0;
        for (j = 0; j < In; ++j)
        {
            sum+=w[i][j]*d_in[var][j]-(float)((-rand()%49+24)*1.0/100);//对第i个神经元计算净权重输入 
            o[i]=1/(1+exp(-1*sum));//使用S型激活函数 确定输出 
        }
    }
     
    for (i = 0; i < Out; ++i)
    {
        sum=0;
        for (j = 0; j < Neuron; ++j)
            sum+=v[i][j]*o[j]-(float)((-rand()%49+24)*1.0/100);//计算输出层的实际输入 
            OutputData[i]=sum;
    }   
}
//权重训练 
void ModelData::BackUpdate(int var)
{
    int i,j;
    double t;
    //计算输出层 
    for (i = 0; i < Neuron; ++i)
    {
        t=0;
        for (j = 0; j < Out; ++j)
        {
        	t+=(OutputData[j]-d_out[j][var])*v[j][i];
            //计算输出神经元权重的校正值 
            dv[j][i]=a*dv[j][i]+a*(OutputData[j]-d_out[j][var])*o[i];
            v[j][i]-=dv[j][i];
        }
        //计算隐含层神经元的校正值 
        for (j = 0; j < In; ++j)
        {
            dw[i][j]=a*dw[i][j]+a*t*o[i]*(1-o[i])*d_in[var][j];
            w[i][j]-=dw[i][j];
        }
    }
}

//训练神经网络 ，迭代 
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
                ComputO(j);//激活 
                e+=fabs((OutputData[i]-d_out[i][j])/d_out[i][j]);
                BackUpdate(j);
            }
    c++;
    }while(c<TrainC && e/Data>0.01);//2000次迭代 
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
    	   //进行分割
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
              cout<<"0:"<<setprecision(6)<<std::fixed<<ans[0]<<" 1:"<<setprecision(6)<<std::fixed<<ans[1]<<"2:"<<setprecision(6)<<std::fixed<<ans[2]<<"   预测种类："<<typeO<<"  实际种类："<<typeI<<endl;
              if(typeI==typeO)
                  trueAns+=1;
    }
           
   
    cout<<"预测正确率:"<<trueAns*1.0/Data*100<<"%"<<endl;
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



