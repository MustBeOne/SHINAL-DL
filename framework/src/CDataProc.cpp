#include "CDataProc.h"
#include "CRandom.h"
#include "CDatasetLoader.h"
using namespace std;

struBatchedData::struBatchedData()
{
	m_vcData = new vector<MatrData ***>;
	btcSize = 0;
	btcN = 0;
	remainder = 0;
}

struBatchedData::~struBatchedData()
{
	int i;
	for (auto it : (*m_vcData))
	{
		for (i = 0; i < btcN; ++i)
		{
			delete[] it[i];
		}
		delete[] it;
	}
	m_vcData->clear();
	FreeObj(m_vcData);
}

CDataLoader::CDataLoader()
{

}

CDataLoader::~CDataLoader()
{

}


struBatchedData* CDataLoader::BatchPackData(MatrData*** data ,int dataN, int leng, int batchSize, bool bshuffle)
{
	int i, n, j, k, m, ind;
	int integer = leng / batchSize;
	int remainder = leng % batchSize;
	int* index = ShuffleIndex(leng, bshuffle);
	MatrData** temp, *** res;
	struBatchedData* pdata = new struBatchedData();
	pdata->btcSize = batchSize;
	pdata->remainder = remainder;
	if (remainder == 0)
	{
		pdata->btcN = integer;
	}
	else
	{
		pdata->btcN = integer + 1;
	}
	n = integer;
	for (i = 0; i < dataN; ++i)
	{
		res = new MatrData * *[pdata->btcN];
		m = 0;
		temp = data[i];
		for (j = 0; j < n; ++j)
		{
			res[j] = new MatrData * [batchSize];
			for (k = 0; k < batchSize; ++k)
			{
				ind = index[m];
				res[j][k] = temp[ind];
				++m;
			}
		}
		if (remainder)
		{
			res[j] = new MatrData * [remainder];
			for (k = 0; k < remainder; ++k)
			{
				ind = index[m];
				res[j][k] = temp[ind];
				++m;
			}
		}
		pdata->m_vcData->push_back(res);
	}
	return pdata;
}
//
//
//PtrRawData CDataLoader::GetTestData()
//{
//	return test;
//}
//
//int CDataLoader::GetTrainLeng()
//{
//	return train->leng;
//}
//
//int CDataLoader::GetTestLeng()
//{
//	return test->leng;
//}
//
//int CDataLoader::GetBatchTimes()
//{
//	return batchTimes;
//}
//
//int* CDataLoader::GetSizeSequen()
//{
//	return btcSize;
//}
//


int* CDataLoader::ShuffleIndex(int leng, bool ifShuf)
{
	int* ind = new int[leng], i;
	for (i = 0; i < leng; ++i)
	{
		ind[i] = i;
	}
	if(ifShuf) CRandom<int*>::Shuffle(ind, ind + leng);
	return ind;
}

