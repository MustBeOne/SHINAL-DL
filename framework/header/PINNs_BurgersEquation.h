#pragma once
#include "CDatasetLoader.h"
#include "CFileReader.h"

typedef map<string, MatrData*>* TyTrain;
typedef MatrData* TyTest;

struct StruPINNsBurgEqua :public StruRawDataBase
{
private:
	string dataMatFile;
	map<string, MatrData*>* m_mpDataTrain;
	CFilesReader<StruDataMatFile>* m_pReader;
public:
	StruPINNsBurgEqua();
	virtual ~StruPINNsBurgEqua();
	virtual void SetFileParameter(map<string, void*>*);
	virtual void LoadDataset(); 
	
	TyTrain GetTrainDataset();
	static TyTrain RetTypeCallbackTraindata(void) { return nullptr; };
	TyTest GetTestDataset();
	static TyTest RetTypeCallbackTestdata(void) { return nullptr; };
private:

};

