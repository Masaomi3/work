#include<iostream>
#include<random>
#include<cmath>
#include <iomanip>
using namespace std;

#define pi 3.14159265358979323846

int data_number;
double* training_data;
double* target_data;

int dense_num = 0;



inline double func(double x)
{
	double x1 = (double)x;
	double x2 = x1 * x1;
	double x3 = x2 * x1;
	double x4 = x3 * x1;
	return x4 - 3 * x3 + 4 * x2 - 2 * x1 + 1;// x^4 - 3x^3 + 4x^2 - 2x + 1  
}

inline double sigmod(double x)
{
	return 1 / (1 + exp(-x));
}

inline double finitlization(double x)
{
	return atan(x/100) / pi + 0.5;
}

inline double infinitlization(double x)
{
	return 100*tan((x - 0.5) * pi);
}

class node
{
public:
	int size;
	double* w;
	double b;

	node(int n)
	{
		size = n;
		w = new double[n];
		b = 0;
	}
	node()
	{
		size = 0;
		w = NULL;
		b = 0;
	}

	node& operator=(const node& other) {
		if (this == &other) return *this; 
		delete[] w;
		size = other.size;
		b = other.b;
		w = new double[size];
		for (int i = 0; i < size; i++) w[i] = other.w[i];
		return *this;
	}

	~node()
	{
		delete[] w;
	}
};

class dense
{
public:
	int node_num;
	int node_size;
	node* W;
	dense* next;
	dense* last;
	dense(int num, int size)
	{
		node_num = num;
		node_size = size;
		W = new node[num];
		for (int i = 0; i < num; i++) W[i] = node(size);
		random_device rd;//------------------------------------------------------------------------------+
		mt19937 gen(rd());                            //                                                 |
		uniform_real_distribution<> dis(-pow(6.0 / (num + size), 0.5), pow(6.0 / (num + size), 0.5));//  |
		for (int i = 0; i < num; i++)                 //                                                 |
		{                                             //                                                 |
			for (int j = 0; j < size; j++)            //                                                 |
			{                                         //                                                 |
				W[i].w[j] = dis(gen);                 //                                                 |
			}                                         //                                                 |
		}//----------------------------------------------------------------------------------------------+--> Xavier initializing

		next = NULL;
		last = NULL;
	}
	dense() 
	{
		W = NULL; node_num = 0; node_size = 0;
		next = NULL; 
		last = NULL;
	}
	~dense()
	{
		delete[] W;
	}

};

dense* neural_network = NULL;

void insert(int node_num)
{
	dense* p = neural_network;
	if (p == NULL)
	{
		neural_network = new dense(node_num, 1);
		dense_num++;
		return;
	}
	while (p->next != NULL) { p = p->next; }
	int pre_node_num = p->node_num;
	p->next = new dense(node_num, pre_node_num);
	p->next->last = p;
	dense_num++;
}

double dot(double* a, double* b, int size)
{
	double result = 0;
	for (int i = 0; i < size; i++) result += a[i] * b[i];
	return result;
}

double prediction_finitlized(double x)
{
	double* result = new double[1];
	result[0] = x;
	int size = 1;
	dense* p = neural_network;
	while(p!=NULL)
	{
		int len = p->node_num;
		double* temp = new double[len];
		for (int i = 0; i < len; i++)
		{
			temp[i] = sigmod(dot(result, p->W[i].w, size) + p->W[i].b);
		}
		size = len;

		delete[] result;
		result = temp;
		p = p->next;
	}
	double result_ = result[0];
	delete[] result;
	return result_;
}

void fit(double rate,double lambda = 0)/////////////////////
{
	
	double*** dJ_dw = new double** [dense_num];
	double** dJ_db = new double* [dense_num];
	dense* p = neural_network;
	for (int i = 0; i < dense_num; i++)
	{
		dJ_dw[i] = new double*[p->node_num];
		dJ_db[i] = new double[p->node_num];
		for (int j = 0; j < p->node_num; j++)
		{
			dJ_dw[i][j] = new double[p->W[j].size];
		}
		p = p->next;
	}


	p = neural_network;
	for (int i = 0; i < dense_num; i++)
	{
		for (int j = 0; j < p->node_num; j++)
		{
			for (int k = 0; k < p->W[j].size; k++) dJ_dw[i][j][k] = 0;
			dJ_db[i][j] = 0;
			
		}
		p = p->next;
	}

	for (int i = 0; i < data_number; i++)
	{
		double x = training_data[i];
		double** history = new double* [dense_num];
		double** dJ_dz_on_x = new double* [dense_num];
		p = neural_network;
		for (int l = 0; l < dense_num; l++)
		{
			history[l] = new double[p->node_num];
			dJ_dz_on_x[l] = new double[p->node_num];
			p = p->next;
		}
		p = neural_network;
		for (int l = 0; l < p->node_num; l++)
		{
			history[0][l] = sigmod(p->W[l].w[0] * x + p->W[l].b);
			dJ_dz_on_x[0][l] = 0;
		}
		p = p->next;
		for (int l = 1; l < dense_num; l++)
		{
			for (int m = 0; m < p->node_num; m++)
			{
				history[l][m] = sigmod(dot(history[l - 1], p->W[m].w, p->node_size) + p->W[m].b);
				dJ_dz_on_x[l][m] = 0;
			}
			p = p->next;
		}

		p = neural_network;
		while (p->next != NULL) p = p->next;////useful?
		double result = history[dense_num - 1][0];
		dJ_dz_on_x[dense_num - 1][0] = (result - finitlization(target_data[i])) * (result - result * result);
		p = p->last;
		for (int l = dense_num - 2; l >= 0; l--)
		{
			for (int m = 0; m < p->node_num; m++)
			{
				for (int k = 0; k < p->next->node_num; k++)  dJ_dz_on_x[l][m] += dJ_dz_on_x[l + 1][k] * p->next->W[k].w[m]*(history[l+1][k]-history[l+1][k]*history[l+1][k]);
			}
			p = p->last;
		}

		p = neural_network;
		for (int l = 0; l < p->node_num; l++)
		{
			for (int m = 0; m < p->node_size; m++)
			{
				dJ_dw[0][l][m] += dJ_dz_on_x[0][l] * x;
			}
			dJ_db[0][l] += dJ_dz_on_x[0][l];
		}
		p = p->next;

		for (int l = 1; l < dense_num; l++)
		{
			for (int m = 0; m < p->node_num; m++)
			{
				for (int n = 0; n < p->node_size; n++)
				{
					dJ_dw[l][m][n] += dJ_dz_on_x[l][m] * history[l - 1][n];
				}
				dJ_db[l][m] += dJ_dz_on_x[l][m];
			}
			p = p->next;
		}

		for (int l = 0; l < dense_num; l++)
		{
			delete[] history[l];
			delete[] dJ_dz_on_x[l];
		}
		delete[] history;
		delete[] dJ_dz_on_x;
	}

	p = neural_network;
	for (int i = 0; i < dense_num; i++)
	{
		for (int j = 0; j < p->node_num; j++)
		{
			for (int k = 0; k < p->node_size; k++)
			{
				dJ_dw[i][j][k] += p->W[j].w[k] * lambda;
				dJ_dw[i][j][k] /= data_number;
				p->W[j].w[k] -= rate * dJ_dw[i][j][k];
			}
			dJ_db[i][j] += p->W[j].b * lambda;
			dJ_db[i][j] /= data_number;
			p->W[j].b -= rate*dJ_db[i][j];
		}
		p = p->next;
	}

	p = neural_network;
	for (int i = 0; i < dense_num; ++i)
	{
		for (int j = 0; j < p->node_num; ++j)
		{
			delete[] dJ_dw[i][j];
		}
		delete[] dJ_dw[i];
		p = p->next;
	}
	delete[] dJ_dw;

	p = neural_network;
	for (int i = 0; i < dense_num; ++i)
	{
		delete[] dJ_db[i];
		dJ_db[i] = nullptr;
		p = p->next;
	}
	delete[] dJ_db;

	
}

int main()
{
	cout << fixed;
//////////////////////////////creating training data//////////////////////////////////////
	cout << "How many training data do you want(num > 0):" << endl;
	cin >> data_number;
	if (data_number <= 0)
	{
		cout << "num cannot less than zero!" << endl;
		exit(1);
	}
	training_data = new double[data_number];
	target_data = new double[data_number];
	double training_data_aver = 0;
	double target_data_aver = 0;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1, 1);
	for (int i = 0; i < data_number; i++)
	{
		training_data[i] = dis(gen);
		target_data[i] = func(training_data[i]);
	}

	cout << "want to show the training data?( Y:Yes | else:No ):" << endl;
	char show_training_data; cin >> show_training_data;
	if (show_training_data == 'Y')
	{
		for (int i = 0; i < data_number; i++) cout << training_data[i] << "  " << target_data[i] << endl;
	}

	
//////////////////////////////////////////////////////////////////////////////////

////////////////////////////////creating network//////////////////////////////////
	int dense_num_;
	cout << "do you want to create network manually or automatically(100 nodes/dense)?" << endl;
	cout << "( m : maually | a : automatically ) : ";
	char c; cin >> c;
	cout << "how many denses do you want?(num > 0):" << endl;
	cin >> dense_num_;
	if (c == 'm')
	{
		cout << "please assign the sizes of these " << dense_num_ << " denses:" << endl;
		for (int i = 0; i < dense_num_; i++)
		{
			cout << "size of dense " << i + 1 << " : ";
			int size; cin >> size;
			insert(size);
		}
		insert(1);
	}
	else if (c == 'a')
	{
		for (int i = 0; i < dense_num_; i++)
		{
			insert(100);
			cout << "dense " << i + 1 << " inserted" << '\r';
			cout.flush();
		}
		cout << endl;
		insert(1);
	}
	else
	{
		cout << "unvalid imput!" << endl;
		exit(-1);
	}
	
///////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////training network//////////////////////////////////

	double rate; int epoch; double lambda;
	cout << "please choose your learning rate(rate > 0):" << endl;
	cin >> rate;
	cout << "please choose your training epoch(epoch > 0):" << endl;
	cin >> epoch;
	cout << "please choose your regularization constant(constant >= 0):" << endl;
	cin >> lambda;
	cout << "-----------------------------------training process-----------------------------------------" << endl;
	for (int i = 0; i < epoch; i++)
	{
		fit(rate, lambda);
		cout << '\r' << "epoch " << i + 1 << " finished";
		cout.flush();
	}
	cout << endl;

////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////testing network//////////////////////////////////////

	cin.clear();
	cout << "how large a testing set do you want(size > 0):" << endl;
	int test_size; cin >> test_size;
	double* test = new double[test_size];
	double* target = new double[test_size];
	double* predict = new double[test_size];

	for (int i = 0; i < test_size; i++)
	{
		test[i] = dis(gen);
		target[i] = func(test[i]);
		predict[i] = infinitlization(prediction_finitlized(test[i]));
		cout << "test " << i << " :" << endl << "    test data : " << test[i] << endl << "    prediction : " << predict[i] << endl;
		cout << "    target : " << target[i] << endl << "    deviation:" << predict[i] - target[i] << endl;
	}

////////////////////////////////////////////////////////////////////////////////////
	
///////////////////////////////////ending work//////////////////////////////////////
	dense* p = neural_network;
	while (p != NULL)
	{
		dense* temp = p;
		p = p->next;
		delete temp;
	}
	delete[] training_data;
	delete[] target_data;
	delete[] test;
	delete[] target;
	delete[] predict;

	return 0;
}