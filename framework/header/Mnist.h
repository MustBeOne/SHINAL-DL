#pragma once
#include "CDatasetLoader.h"
#include "tools.h"

struct StruRawDataMnist:public StruRawDataBase
{
private:
	MatrData** inputTrain;
	MatrData** targetTrain;
	MatrData** inputTest;
	MatrData** targetTest;
	int lengTrain, lengTest;
	string trainXfile, trainYfile, testXfile, testYfile;
public:
	StruRawDataMnist();
	virtual ~StruRawDataMnist();
	virtual void SetFileParameter(map<string, void*>*);
	virtual void LoadDataset();

	MatrData*** GetTrainDataset();
	static MatrData*** RetTypeCallbackTraindata(void) { return nullptr; };
	MatrData*** GetTestDataset();
	static MatrData*** RetTypeCallbackTestdata(void) { return nullptr; };
private:
	void ReadTrainFile();
	void ReadTestFile();
};

