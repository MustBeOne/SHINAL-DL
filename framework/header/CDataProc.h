#pragma once
#include "tools.h"


/// <summary>
/// return by CDataLoader,pack the linear-data and batchded index
/// </summary>
struct struBatchedData {
public:
	struBatchedData();
	virtual ~struBatchedData();
	vector<MatrData***>* m_vcData;
	int btcSize;
	int btcN;
	int remainder;
};
class CDataLoader {
public:
	CDataLoader();
	virtual ~CDataLoader();
	////void LoadTestDataset(LoadData*);
	//PtrRawData GetTestData();
	//int GetTrainLeng();
	//int GetTestLeng();
	//int GetBatchTimes();
	//int* GetSizeSequen();

	static struBatchedData* BatchPackData(MatrData***, int, int, int, bool bshuffle = true);
private:
	static int* ShuffleIndex(int, bool ifShuf = true);
private:
};