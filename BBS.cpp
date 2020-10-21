#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream> 
#include <sstream>
#include <string>  
#include <cstring>
#include <streambuf> 
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>

//
//This program is inspired by an article named "Best-Buddies Similarity for Robust Template Matching£¬CVPR2015"
//After reading this passage, I started to realize this method described in it by using OPENCV and c++.
//And here is the source code I wrote.
//"BBS" is short for "Best-Buddies Similarity", which is a useful, robust, and parameter-free similarity measure between two sets of points.
//BBS is based on counting the number of Best-Buddies Pairs (BBPs)-pairs of points in source and target sets, 
//where each point is the nearest neighbor of the other. 
//BBS has several key features that make it robust against complex geometric deformations and high levels of outliers, 
//such as those arising from background clutter and occlusions. 
//And the output of this source code on the challenging real-world dataset is amazingly precise, far beyond my previous expectation.
//

using namespace cv;

using namespace std;

string openAddr = "E:\\BBS\\data\\";

//type the input source's directory here
const char *to_searchJPG = "E:\\BBS\\data\\*.jpg";
//this .txt file contains the particular part of this image to be handled.
const char *to_searchTXT = "E:\\\BBS\\data\\*.txt";

int gamma = 2, pz = 3, pairCount = 25;

//Gaussian lowpass filter
float Gaussian[]{ 0.0277, 0.1110, 0.0277, 0.1110, 0.4452, 0.1110, 0.0277, 0.1110, 0.0277 };

//convert the image's info into a matrix and store them as a 2-dim vector list.
Mat Im2col(Mat src, int Mrows, int Mcols)
{
	int col = 0;
	int rows = Mrows * Mcols;
	int cols = ceil(src.rows / Mrows) * ceil(src.cols / Mcols);
	Mat ans(rows, cols, CV_32FC3);
	for (int j = 0; pz * j < src.cols; j++)
	{
		for (int i = 0; pz * i < src.rows; i++)
		{
			for (int k = 0; k < Mcols; k++)
			{
				for (int r = 0; r < Mrows; r++)
				{
					ans.at<Vec3f>(r + k*Mrows, col)[0] = src.at<Vec3f>(i*Mcols + k, j*Mrows + r)[0];
					ans.at<Vec3f>(r + k*Mrows, col)[1] = src.at<Vec3f>(i*Mcols + k, j*Mrows + r)[1];
					ans.at<Vec3f>(r + k*Mrows, col)[2] = src.at<Vec3f>(i*Mcols + k, j*Mrows + r)[2];
				}
			}
			col++;
		}
	}
	return ans;
}

//the main code
int main()
{
	string TName = "pair0012_frm1_Coke_0114.jpg";
	string IName = "pair0012_frm2_Coke_0134.jpg";
	string TxtName = "pair0012_frm1_Coke_0114.txt";

	int pairnum = 0;
	intptr_t handle, handle2;                                             
	_finddata_t fileinfo, fileinfo2;                          
                
	while (pairnum < pairCount)              
	{
		if (pairnum == 0)
		{
			handle = _findfirst(to_searchJPG, &fileinfo);
			if (-1 == handle)return -1;
			handle2 = _findfirst(to_searchTXT, &fileinfo2);
			if (-1 == handle)return -1;
		}
		else
		{
			_findnext(handle, &fileinfo);
			_findnext(handle2, &fileinfo2);
		}
		TName = fileinfo.name;
		Mat Ts = imread(openAddr + TName);
		_findnext(handle, &fileinfo);
		IName = fileinfo.name;
		Mat Is = imread(openAddr + IName);
		TxtName = fileinfo2.name;
		ifstream input(openAddr + TxtName);
		_findnext(handle2, &fileinfo2);
		int Tcut[4];
		string temp, temp2;
		getline(input, temp);
		istringstream ss(temp);
		int i = 0;
		do
		{
			ss >> Tcut[i++];
		}
		while (getline(ss, temp2, ','));
		
		//clipping pictures
		if(Tcut[2]%pz < pz/2)
			Tcut[2] -= Tcut[2] % pz;
		else
			Tcut[2] += pz - Tcut[2] % pz;
		if (Tcut[3] % pz < pz / 2)
			Tcut[3] -= Tcut[3] % pz;
		else
			Tcut[3] += pz - Tcut[3] % pz;
		Mat T = Ts(Rect(Tcut[0], Tcut[1], Tcut[2], Tcut[3]));
		Mat I = Is(Rect(0, 0, (Is.cols - Is.cols%pz), (Is.rows - Is.rows % pz)));

		T.convertTo(T, CV_32FC3, 1.0 / 255.0);
		I.convertTo(I, CV_32FC3, 1.0 / 255.0);

		Mat TMat = Im2col(T, pz, pz);
		Mat IMat = Im2col(I, pz, pz);

		int N = TMat.cols;

		int rowT = T.rows;
		int colT = T.cols;
		int rowI = I.rows;
		int colI = I.cols;

		//pre compute spatial distance component
		vector<vector<float>> Dxy, Drgb, Drgb_prev, D, D_r, BBS;
		Dxy.resize(N);
		Drgb_prev.resize(N);
		Drgb.resize(N);
		D.resize(N);
		D_r.resize(N);
		for (int i = 0; i < N; i++)
		{
			Dxy[i].resize(N);
			Drgb[i].resize(N);
			Drgb_prev[i].resize(N);
			D[i].resize(N);
			D_r[i].resize(N);
		}

		BBS.resize(rowI);
		for (int i = 0; i < BBS.size(); i++)
			BBS[i].resize(colI);

		//Drgb's buffer
		vector<vector<vector<float>>> Drgb_buffer;
		Drgb_buffer.resize(N);
		int bufSize = rowI - rowT;
		for (int i = 0; i < Drgb_buffer.size(); i++)
		{
			Drgb_buffer[i].resize(N);
			for (int j = 0; j < Drgb_buffer[i].size(); j++)
			{
				Drgb_buffer[i][j].resize(bufSize);
			}
		}

		vector<float> xx, yy;
		for (int i = 0; pz * i < colT; i++)
		{
			float n = pz * i * 3.0039;
			for (int j = 0; pz * j < rowT; j++)
			{
				float m = pz * j * 0.0039;
				xx.push_back(n);
				yy.push_back(m);
			}
		}

		for (int j = 0; j < xx.size(); j++)
		{
			for (int i = 0; i < xx.size(); i++)
			{
				Dxy[i][j] = pow((xx[i] - xx[j]), 2) + pow((yy[i] - yy[j]), 2);
			}
		}

		vector<vector<int>> IndMat;
		IndMat.resize(I.rows / pz);
		for (int i = 0; i < IndMat.size(); i++)
			IndMat[i].resize(I.cols / pz);

		int n = 0;
		for (int j = 0; j < I.cols / pz; j++)
		{
			for (int i = 0; i < I.rows / pz; i++)
			{
				IndMat[i][j] = n++;
			}
		}

		// loop over image pixels
		for (int coli = 0; coli < (colI / pz - colT / pz + 1); coli++)
		{
			for (int rowi = 0; rowi < (rowI / pz - rowT / pz + 1); rowi++)
			{
				Mat PMat(9, N, CV_32FC3);
				vector<int> v;
				vector<float> w;
				for (int j = coli; j < (coli + colT / pz); j++)
				{
					for (int i = rowi; i < (rowi + rowT / pz); i++)
					{
						v.push_back(IndMat[i][j]);
					}
				}
				int ptv = 0;
				for (int ix = 0; ix < N; ix++)
				{
					for (int jx = 0; jx < 9; jx++)
					{
						PMat.at<Vec3f>(jx, ix)[0] = IMat.at<Vec3f>(jx, v[ptv])[0];
						PMat.at<Vec3f>(jx, ix)[1] = IMat.at<Vec3f>(jx, v[ptv])[1];
						PMat.at<Vec3f>(jx, ix)[2] = IMat.at<Vec3f>(jx, v[ptv])[2];
					}
					ptv++;
				}
				
				//compute distance matrix
				for (int idxP = 0; idxP < N; idxP++)
				{
					Mat Temp(9, N, CV_32FC3);
					for (int i = 0; i < Temp.cols; i++)
					{
						for (int j = 0; j < Temp.rows; j++)
						{
							Temp.at<Vec3f>(j, i)[0] = pow(((-TMat.at<Vec3f>(j, i)[0] + PMat.at<Vec3f>(j, idxP)[0])*Gaussian[j]), 2);
							Temp.at<Vec3f>(j, i)[1] = pow(((-TMat.at<Vec3f>(j, i)[1] + PMat.at<Vec3f>(j, idxP)[1])*Gaussian[j]), 2);
							Temp.at<Vec3f>(j, i)[2] = pow(((-TMat.at<Vec3f>(j, i)[2] + PMat.at<Vec3f>(j, idxP)[2])*Gaussian[j]), 2);
						}
					}
					for (int jx = 0; jx < N; jx++)
					{
						float res = 0;
						for (int ix = 0; ix < 9; ix++)
						{
							if (D[ix][jx] < 1e-4)
								D[ix][jx] = 0;
							res += Temp.at<Vec3f>(ix, idxP)[0];
							res += Temp.at<Vec3f>(ix, idxP)[1];
							res += Temp.at<Vec3f>(ix, idxP)[2];
						}
						Drgb[jx][idxP] = res;
					}
				}

				//make the reversed matrix of distance matrix
				for (int ix = 0; ix < N; ix++)
				{
					for (int jx = 0; jx < N; jx++)
					{
						
						//calculate distance
						D[ix][jx] = Dxy[ix][jx] * gamma + Drgb[ix][jx];
						if (D[ix][jx] < 1e-4)
							D[ix][jx] = 0;
						D_r[jx][ix] = D[ix][jx];
					}
				}

				//compute the BBS value of this point
				vector<float> minVal1, minVal2;
				vector<int> idx1, idx2, ii1, ii2;

				for (int ix = 0; ix < N; ix++)
				{
					auto min1 = min_element(begin(D[ix]), end(D[ix]));
					minVal1.push_back(*min1);
					idx1.push_back(distance(begin(D[ix]), min1));

					ii1.push_back(ix * N + idx1[ix]);
				}
				for (int ix = 0; ix < N; ix++)
				{
					auto min2 = min_element(begin(D_r[ix]), end(D_r[ix]));
					minVal2.push_back(*min2);
					idx2.push_back(distance(begin(D_r[ix]), min2));

					ii2.push_back(ix * N + idx2[ix]);
				}
				
				vector<vector<int>> IDX_MAT1, IDX_MAT2;
				IDX_MAT1.resize(N);
				IDX_MAT2.resize(N);
				for (int i = 0; i < N; i++)
				{
					IDX_MAT1[i].resize(N);
					IDX_MAT2[i].resize(N);
				}
				int sum = 0, sum2 = 0;
				int pt1 = 0, pt2 = 0;
				for (int ix = 0; ix < N; ix++)
				{
					for (int jx = 0; jx < N; jx++)
					{
						IDX_MAT1[ix][jx] = 0;
						IDX_MAT2[ix][jx] = 999;
						if (pt1 < N && ((ix*N + jx) == ii1[pt1]))
						{
							IDX_MAT1[ix][jx] = 1;
							pt1++;
						}
						if (pt2 < N && ((jx*N + ix) == ii2[pt2]))
						{
							IDX_MAT2[ix][jx] = 1;
							pt2++;
						}
						if (IDX_MAT2[ix][jx] == IDX_MAT1[ix][jx])
							sum += 1;
					}
				}	
				BBS[rowi][coli] = sum;
			}
		}
		
		float max = 0;
		int markRow, markCol;
		
		//Initialize the output iamge and .csv files
		ofstream output(format("OUTPUT%d.csv", pairnum));
		for (int i = 0; i < rowI; i++)
		{
			for (int j = 0; j < colI; j++)
			{
				BBS[i][j] /= N;
				output << BBS[i][j] << ",";
			}
			output << endl;
		}
		output.close();
		
		
		for (int i = 0; i < rowI; i++)
		{
			for (int j = 0; j < colI; j++)
			{
				float t = BBS[i][j];
				if (t >= max)
				{
					max = t;
					markRow = i;
					markCol = j;
				}
			}
		}
		Mat OUTPUT1 = imread(openAddr + IName);
		OUTPUT1.convertTo(OUTPUT1, CV_32FC3, 1.0 / 255.0);
		Mat OUTPUT2 = imread(openAddr + TName);
		OUTPUT2.convertTo(OUTPUT2, CV_32FC3, 1.0 / 255.0);
		Mat OUTPUT3;
		
		//mark rectangle
		for (int i = markRow * pz; i < markRow * pz + rowT; i++)
		{
			for (int j = markCol * pz; j < markCol*pz + colT; j++)
			{
				if (i == markRow * pz || j == markCol * pz || i == markRow * pz + rowT - 1 || j == markCol * pz + colT - 1)
				{
					OUTPUT1.at<Vec3f>(i, j)[0] = 0;
					OUTPUT1.at<Vec3f>(i, j)[1] = 0;
					OUTPUT1.at<Vec3f>(i, j)[2] = 1;

				}
			}
		}
		for (int i = Tcut[0]; i < Tcut[0] + Tcut[2]; i++)
		{
			for (int j = Tcut[1]; j < Tcut[1] + Tcut[3]; j++)
			{
				if (i == Tcut[0] || j == Tcut[1] || i == Tcut[0] + Tcut[2] - 1 || j == Tcut[1] + Tcut[3] - 1)
				{
					OUTPUT2.at<Vec3f>(j, i)[0] = 0;
					OUTPUT2.at<Vec3f>(j, i)[1] = 1;
					OUTPUT2.at<Vec3f>(j, i)[2] = 0;
				}
			}
		}
		hconcat(OUTPUT1, OUTPUT2, OUTPUT3);
		ostringstream os;
		os << "OUTPUT" << (pairnum++) << ".jpg";
		string resultName = os.str();
		OUTPUT3.convertTo(OUTPUT3, CV_32SC3, 255);
		imwrite(resultName, OUTPUT3);
	}
	_findclose(handle);                             

	
	waitKey(0);
}