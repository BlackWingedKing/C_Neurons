//opencv libs
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//c++ libs
#include <iostream>

#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <math.h>
#include <cstdlib>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

//declare all the global thing here
// fstream myfile;

// myfile.open("error.txt", ios::out);// | ios::trunc);
// myfile.close();

double alpha = 0.1;
const int epoch = 5;
const int batch_size = 10;
double maxerror = -10;

const int data_size = 500;

//math and useful functions
double sigmoid(double z)
{
	//sigmoid activation function	
	double a;
	a = -1*z;
	a = exp(a);
	a = 1.0+a;
	a = 1.0/a;

	return a;
}

double ReLU(double z)
{
	//ReLU activation function
	if(z<=0.0)
	{
		return 0.0;
	}
	else
	{
		return z;
	}
}
//random double generator
double random(double min, double max) 
{
	return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;	
}

vector<double> softmax(vector<double> v)
{
	vector<double> val;
	double temp = 0.0;
	for (int i = 0; i < v.size(); ++i)
	{
		temp = temp + exp(-1*v[i]);
	}
	for (int i = 0; i < v.size(); ++i)
	{
		val.push_back((exp(-1*v[i])/temp));
	}

	return val;
}

//structures for building an nn
struct neuron
{
	//each neuron has n parameters
	//1 bias, n theta values and n inputs ai
	int n;
	double bias;
	vector<double> theta;
	vector<double> ai;
	//parameters for backprop
	double delta;

	//member functions
	//basic constructor
	neuron(int a)
	{
		bias = 0;
		n = a;
		delta =0.0;
		theta.resize(a,0.0);
		ai.resize(a,0.0);
	}

	//Resize function
	void Resize(int a, double value)
	{
		//bias = 0.0;
		delta = 0.0;
		n = a;
		theta.resize(a,value);
		ai.resize(a,value);

		//this part is for random initialisation of the neuron parameters
		bias = random(0.0,0.001);
		for (int i = 0; i < a; ++i)
		{
			double tem = random(0.0,0.001);
			theta[i] = tem;
		}
	}

	double output()
	{
		double output = 0.0;
		for(int i=0;i<n;i++)
		{
			output = output + theta[i]*ai[i];
		}
		output = output + bias;
		output = ReLU(output);

		return output;
	}
	double value()
	{
		double output = 0.0;
		for(int i=0;i<n;i++)
		{
			output = output + theta[i]*ai[i];
		}
		output = output + bias;

		return output;
	}
	//
};

struct layer
{
	//the structure layer it has n number of neurons
	int n;	 
	//nrn is a vector of neurons
	vector<neuron> nrn;
	
	//constructor creates n neurons which has only biases
	layer(int a) //a is the number of neurons
	{
		n = a;	
		neuron nrn1(0);
		nrn.resize(a,nrn1);
	}
	void Resize(int a, int t)
	{
		//Resize to a number of neurons and n number of thetas
		n = a;
		neuron nrn1(t);
		nrn1.Resize(t,0);
		nrn.resize(a,nrn1);
		nrn[0].Resize(t,0); //just added this as an extra 
		//due to some other bug in the code
	}

};

struct nn
{
	//the structure neural network it has 
	//neurons in the first layer are 
	int nlayers;
	vector<layer> lyr;

	//constructor which creates a layers of 1 neuron each
	nn(int a)
	{
		nlayers = a;
		layer lyr1(1);
		lyr.resize(a,lyr1);
	}

};

//Description of the Neural, define it as
//1024->256->64->10
//define a 4 layered  network with 2 hidden layers and the number of the layers as mentioned above
//the name of the network is hand and 4 layers were declared

//the feature input layer 
//hand(0).

//define a global neural network

nn hand(4);


//forward is the function for forward propagation
vector<double> forward(Mat img, nn &network)
{
	//resizing the image to 32*32 so that the number of features are 1024
	resize(img, img, Size(32,32));
	// threshold( img, img, 5, 1,1);

	vector<double> pred;
	vector<double> z;

	//assignining input features to the nn
	
	//inputs(a's) to the first layer
	int count = 0;
	for (int i = 0; i < img.rows; ++i)
	{
	 	for (int j = 0; j < img.cols; ++j)
	 	{
	 		//here according to the initialisation, no. of inputs to each neuron is 0
	 		//hence just use the bias as the input feature such that the output of each neuron is 
	 		//just the feature //horizontal assignment
	 		network.lyr[0].nrn[count].bias = img.at<uchar>(j,i);
	 		network.lyr[0].nrn[count].bias = network.lyr[0].nrn[count].bias/255.0;
	 		count++;
	 	}
	
	} 
	
	//all the inputs to the other layer are intialised to zero at the start!
	
	for (int k = 0; k < 3; ++k)
	{
		for (int i = 0; i < network.lyr[k+1].n ; ++i)
		{	
			//which are the same for all the neurons
			
			for (int j = 0; j < network.lyr[k].n ; ++j)
			{
				//cout<<"xyz"<<endl;
				if(k ==0 )
				{
					network.lyr[k+1].nrn[i].ai[j] = network.lyr[k].nrn[j].bias;
				}
				else
				{
					network.lyr[k+1].nrn[i].ai[j] = network.lyr[k].nrn[j].output();	
				}
			}
		}
	}
	//now since all the inputs are defined so apply the softmax to the outputs of the
	// cout<<"printing"<<endl;
	// for (int i = 0; i < network.lyr[0].n; ++i)
	// {
	// 	cout<<"image values"<<network.lyr[0].nrn[i].bias<<endl;
	// }
	
	//cout<<network.lyr[1].nrn[i].value()<<endl;
	//cout<<network.lyr[3].nrn[i].output()<<endl;
	//cout<<"bias"<<network.lyr[1].nrn[i].bias<<endl;
	// for (int j = 0; j < network.lyr[2].nrn[1].theta.size(); ++j)
	// {
	// 	cout<<"ai"<<"	"<<network.lyr[2].nrn[1].ai[j]<<"	"<<"theta"<<"	"<<j<<"	"<<network.lyr[2].nrn[1].theta[j]<<endl;
	// 	cout<<"2ai"<<"	"<<network.lyr[2].nrn[2].ai[j]<<"	"<<"theta"<<"	"<<j<<"	"<<network.lyr[2].nrn[2].theta[j]<<endl;
	// }
	
	

	for (int i = 0; i < network.lyr[3].nrn.size(); ++i)
	{
		z.push_back(network.lyr[3].nrn[i].value());
	}
	pred = softmax(z);
	return pred;
}

//functions required for backward propagation
//cost is the cost function defined for a single training example
double cost(vector<double> pred , vector<double> y)
{
	double val = 0.0;
	
	// for (int i = 0; i < pred.size(); ++i)
	// {
	// 	cout<<pred[i]<<"\t"<<y[i]<<endl;
	// }

	for (int i = 0; i < y.size(); ++i)
	{
		if(pred[i]==1.0 || pred[i]==0.0)
		{
			if(pred[i]==y[i])
			{
				val = val + 0;	
			}
			else
			{
				val = val + maxerror;
			}		
		}
		else
		{
			val = val + y[i]*log(pred[i]) + (1-y[i])*log(1-pred[i]);
		}
	}
	return val;
	//after the loop val will be summation of logistic regression cost 
}

//backpropagation is the function for the backpropagation
void backward(nn &network, vector<double> y, Mat image)
{
	int number = 0;
	for (int i = 1; i < network.nlayers; ++i)
	{
		number = number + network.lyr[i].n*network.lyr[i].nrn[i].n;
	}
	

	//calculate the gradients!!
	//for the last layer !!
	vector<double> pred;
	pred = forward(image, hand);
	
	//deltas for the last layer
	for (int i = 0; i < network.lyr[network.nlayers-1].nrn.size(); ++i)
	{
		network.lyr[network.nlayers-1].nrn[i].delta= pred[i] - y[i];

	}

	//deltas for other layer
	for (int i = network.nlayers-2; i >0 ; --i)
	{
		for (int j = 0; j < network.lyr[i].nrn.size(); ++j)
		{
			//network.lyr[i].nrn[j].delta = 		
			for (int k = 0; k < network.lyr[i+1].nrn.size(); ++k)
			{
				network.lyr[i].nrn[j].delta = 0.0;
				network.lyr[i].nrn[j].delta = network.lyr[i].nrn[j].delta + network.lyr[i+1].nrn[k].delta*network.lyr[i+1].nrn[k].theta[j];
			}
		}
	}

	for (int i = 1; i < network.nlayers; ++i)
	{
		for (int j = 0; j < network.lyr[i].nrn.size(); ++j)
		{
			for (int k = 0; k < network.lyr[i-1].nrn.size(); ++k)
			{
				//this is the line for the gradient of theta which connects layer i-1-neuron k to layer i - neuron j 
				double gradient = 0.0;
				
				gradient = gradient + network.lyr[i-1].nrn[k].output()*network.lyr[i].nrn[j].delta;
				network.lyr[i].nrn[j].theta[k] = network.lyr[i].nrn[j].theta[k] + alpha*gradient;		
				//cout << "gradients are "<<gradient <<" theta "<<network.lyr[i].nrn[j].theta[k] <<endl;
			}
			
		}
	}

}

//these are specific for my data

Mat train_data[10][data_size];

void take_data(int n)
{
	string name,temp1,temp2;

	//10 classes
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 1; j < n+1; ++j)
		{
			stringstream ss;
			stringstream ss1;
			ss<<i;
			temp1 = ss.str();
			ss1<<j;
			temp2 = ss1.str();
			name = temp1+"/"+temp2+".png";
			train_data[i][j-1] = imread(name,0);
		}
	}
}

void print_predictions(Mat img,vector<double> y)
{
	vector<double> predi = forward(img,hand);
	for (int i = 0; i < predi.size(); ++i)
	{
		cout<<predi[i]<<"\t";
	}
	cout<<"\n";
	double error = cost(predi,y);
	cout<<"The error is "<<error<<endl;
}

//for saving and loading the model
void save_model(nn &network,const char* name)
{
	fstream model;
	model.open(name,ios::out);
	
	//now the convention first line consists of number of layers
	model<<network.nlayers<<"\n";
	//next line consists of nlayers of numbers spaced by tab which are the number of neurons in each layer
	for (int i = 0; i < network.nlayers; ++i)
	{
		model<<network.lyr[i].n<<"\t";
	}
	model<<"\n";
	//next lines consist of paramters of layers from 1st neuron to last
	for (int i = 1; i < network.nlayers; ++i)
	{
		for (int j = 0; j < network.lyr[i].n; ++j)
		{
			for (int k = 0; k < network.lyr[i-1].n; ++k)
			{
				model<<network.lyr[i].nrn[j].theta[k]<<"\t";	
			}
			model<<network.lyr[i].nrn[j].bias<<"\n";
			//the last of the line will be the bias
		}
	}
	model.close();

}

void load_model(nn &network,const char* name)
{
	fstream model;
	model.open(name,ios::in);
	int n;
	model>>n;
	int number[n];
	
	for (int i = 0; i < n; ++i)
	{
		model>>number[i];
	}

	for (int i = 1; i < n; ++i)
	{
		for (int j = 0; j < network.lyr[i].n; ++j)
		{
			for (int k = 0; k < network.lyr[i-1].n; ++k)
			{
				model<<network.lyr[i].nrn[j].theta[k];
			}
			model<<network.lyr[i].nrn[j].bias;
		}
	}
	model.close();
}
int main()
{
	//initialising the NN
	hand.lyr[0].Resize(1024,0);
	hand.lyr[1].Resize(256,1024);
	hand.lyr[2].Resize(64,256);
	hand.lyr[3].Resize(10,64);

	//take or load all the data
	take_data(data_size);

	//define the annotations
	vector<vector<double> >annotation;
	for (int i = 0; i < 10; ++i)
	{
		vector<double> temp;
		for (int j = 0; j < 10; ++j)
		{
			if(j==i)
			{
				temp.push_back(1.0);	
			}
			else
			{
				temp.push_back(0.0);
			}				
		}
		annotation.push_back(temp);
		temp.clear();
	}


	// //training
	vector<double> temp;
	int count = 0;
	for (int k = 0; k < epoch; ++k)
	{
		cout<<"epoch number = "<<k<<"\n";
		for (int j = 0; j < data_size; ++j)
		{
			double error = 0.0;
			for(int i = 0;i<2;i++)		
			{
				backward(hand,annotation[i],train_data[i][j]);
				temp = forward(train_data[i][j],hand);		
				cout<<"The cost after "<<count<<" iterations is "<<cost(temp,annotation[i])<<endl; 
				count++;
				error = error + cost(temp,annotation[i]); 
			}
			cout<<"The average Cost is "<<(error/2.0)<<endl;
			
			//save the models
			if(count%100 == 0)
			{
				save_model(hand,"models/save.txt");
				cout<<"saving weights to models/save.txt"<<"\n";
			}
			
			fstream myfile;
			myfile.open("error.txt",ios::app);
			myfile<<(error/2.0)<<"\n";
			myfile.close();
		}
	}

	print_predictions(train_data[0][4],annotation[0]);
	print_predictions(train_data[1][3],annotation[1]);
	print_predictions(train_data[2][3],annotation[2]);
	print_predictions(train_data[3][3],annotation[3]);
	print_predictions(train_data[4][3],annotation[4]);
	print_predictions(train_data[5][3],annotation[5]);
	print_predictions(train_data[6][3],annotation[6]);
	print_predictions(train_data[7][3],annotation[7]);
	print_predictions(train_data[8][3],annotation[8]);
	print_predictions(train_data[9][3],annotation[9]);	
}