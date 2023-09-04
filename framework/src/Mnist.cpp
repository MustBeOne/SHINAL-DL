#include "Mnist.h"
#include <fstream>
#include "struMat.h"
using namespace std;

StruRawDataMnist::StruRawDataMnist()
{
	emDatasetType = EmDatasetType::MNIST;
	inputTrain = nullptr;
	targetTrain = nullptr;
	inputTest = nullptr;
	targetTest = nullptr;
	lengTrain = 0, lengTest = 0;
}

StruRawDataMnist::~StruRawDataMnist()
{
	Free2DArr(inputTrain, lengTrain);
	Free2DArr(targetTrain, lengTrain);
	Free2DArr(inputTest, lengTest);
	Free2DArr(targetTest, lengTest);
}

void StruRawDataMnist::SetFileParameter(map<string, void*>* mpParam)
{
	lengTrain = *(int*)((*mpParam)["lengTrain"]);
	lengTest  = *(int*)((*mpParam)["lengTest"]);
	trainXfile = *(string*)((*mpParam)["trainXfile"]);
	trainYfile = *(string*)((*mpParam)["trainYfile"]);
	testXfile = *(string*)((*mpParam)["testXfile"]);
	testYfile = *(string*)((*mpParam)["testYfile"]);
}

void StruRawDataMnist::LoadDataset()
{
	ReadTrainFile();
	ReadTestFile();
}

MatrData*** StruRawDataMnist::GetTrainDataset()
{
	MatrData*** data = new MatrData * *[2];
	data[0] = inputTrain;
	data[1] = targetTrain;
	return data;
}

MatrData*** StruRawDataMnist::GetTestDataset()
{
	MatrData*** data = new MatrData * *[2];
	data[0] = inputTest;
	data[1] = targetTest;
	return data;
}

void StruRawDataMnist::ReadTrainFile()
{
	ifstream f;
	MatrData* sdata;
	int higPix = 28;
	int widPix = 28;
	int size = higPix * widPix;
	uint8_t* singalImg = new uint8_t[size];
	Scalar* pix = new Scalar[size];
	int i, n;
	unsigned char l;
	inputTrain = new MatrData * [lengTrain];
	targetTrain = new MatrData * [lengTrain];
	//
	f.open(trainXfile, ios::binary);
	f.seekg(16, ios::beg);
	for (n = 0; n < lengTrain; ++n) {
		sdata = new MatrData(higPix * widPix, 1);
		f.read((char*)singalImg, size * sizeof(char));
		for (i = 0; i < size; ++i)
		{
			pix[i] = static_cast<Scalar>(singalImg[i]) / 255.;
		}
		memcpy(&(*sdata)(0, 0), pix, size * sizeof(Scalar));
		inputTrain[n] = sdata;
	}
	f.close();
	delete[] singalImg;
	delete[] pix;
	//
	f.open(trainYfile, ios::binary);
	f.seekg(8, ios::beg);
	for (n = 0; n < lengTrain; ++n)
	{
		sdata = new MatrData(MatrData::Zero(10, 1));
		f.read((char*)&l, sizeof(l));
		(*sdata)((int)l, 0) = 1;
		targetTrain[n] = sdata;
	}
	f.close();
}

void StruRawDataMnist::ReadTestFile()
{
	ifstream f;
	MatrData* sdata;
	int higPix = 28;
	int widPix = 28;
	int size = higPix * widPix;
	uint8_t* singalImg = new uint8_t[size];
	Scalar* pix = new Scalar[size];
	int i, n;
	unsigned char l;
	inputTest = new MatrData * [lengTest];
	targetTest = new MatrData * [lengTest];
	//
	f.open(testXfile, ios::binary);
	f.seekg(16, ios::beg);
	for (n = 0; n < lengTest; ++n) {
		sdata = new MatrData(higPix * widPix, 1);
		f.read((char*)singalImg, size * sizeof(char));
		for (i = 0; i < size; ++i)
		{
			pix[i] = static_cast<Scalar>(singalImg[i]) / 255.;
		}
		memcpy(&(*sdata)(0, 0), pix, size * sizeof(Scalar));
		inputTest[n] = sdata;
	}
	f.close();
	delete[] singalImg;
	delete[] pix;
	//
	f.open(testYfile, ios::binary);
	f.seekg(8, ios::beg);
	for (n = 0; n < lengTest; ++n)
	{
		sdata = new MatrData(MatrData::Zero(10, 1));
		f.read((char*)&l, sizeof(l));
		(*sdata)((int)l, 0) = 1;
		targetTest[n] = sdata;
	}
	f.close();	
}
