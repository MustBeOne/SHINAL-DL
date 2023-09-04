#pragma once
#include "tools.h"

struct StruRawDataBase {
public:
	StruRawDataBase();
	virtual ~StruRawDataBase(); 
	virtual void SetFileParameter(map<string, void*>*) {};
	virtual void LoadDataset() {};

	EmDatasetType emDatasetType;
};

template<class DatasetType>
class CDatasetLoader {
public:
	CDatasetLoader();
	virtual ~CDatasetLoader();
	bool ReadDatasetFile();
	/*
	* These are the return-type callback function for different 
	* dataset to implement dynamic create.
	*/
	DatasetType* GetDataset() { return m_pDataset; };
	auto GetTrainData()->decltype(DatasetType::RetTypeCallbackTraindata());
	auto GetTestData()->decltype(DatasetType::RetTypeCallbackTestdata());
private:
	DatasetType* m_pDataset;
};

template<class DatasetType>
auto CDatasetLoader<DatasetType>::GetTestData() ->decltype(DatasetType::RetTypeCallbackTestdata())
{
	return m_pDataset->GetTestDataset();
}

template<class DatasetType>
auto CDatasetLoader<DatasetType>::GetTrainData() ->decltype(DatasetType::RetTypeCallbackTraindata())
{
	return m_pDataset->GetTrainDataset();
}

template<class DatasetType>
bool CDatasetLoader<DatasetType>::ReadDatasetFile()
{
	m_pDataset->LoadDataset();
	return true;
}

template<class DatasetType>
CDatasetLoader<DatasetType>::~CDatasetLoader()
{
	FreeObj(m_pDataset);
}

template<class DatasetType>
CDatasetLoader<DatasetType>::CDatasetLoader()
{
	m_pDataset = new DatasetType();
}

