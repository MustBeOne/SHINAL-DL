#pragma once
#include "tools.h"

struct StruDataFileBase;
struct StruDataMatFile;

using std::cout;
using std::endl;

template<class DataType>
class CFilesReader {
private:
	DataType* m_pReadData;
	string m_sFileDir;
public:
	CFilesReader();
	virtual ~CFilesReader();

	DataType* GetDataContainer();
	void SetFileDir(string fileDir);
	MatrData* GetMapData(string name);

	bool ReadMatrixDataByName(string* name, int n);
private:
};

template<class DataType>
MatrData* CFilesReader<DataType>::GetMapData(string name)
{
	return m_pReadData->GetMapData(name);
}

template<class DataType>
bool CFilesReader<DataType>::ReadMatrixDataByName(string* name, int n)
{
	if (m_pReadData->fileType != EmFileType::MAT_FILE)
	{
		cout << "Can not Read the specified file with this way! " << endl;
		return false;
	}
	m_pReadData->GetNameDataFromMat(m_sFileDir, name, n);
}

template<class DataType>
void CFilesReader<DataType>::SetFileDir(string fileDir)
{
	m_sFileDir = fileDir;
}

template<class DataType>
CFilesReader<DataType>::CFilesReader()
{
	m_pReadData = new DataType();
}

template<class DataType>
CFilesReader<DataType>::~CFilesReader()
{
	FreeObj(m_pReadData);
}

template<class DataType>
DataType* CFilesReader<DataType>::GetDataContainer()
{
	return m_pReadData;
}


struct StruDataFileBase
{
	StruDataFileBase();
	virtual ~StruDataFileBase() {};
	virtual MatrData* GetMapData(string name) { return nullptr; };

	EmFileType fileType;
};
struct StruDataMatFile:public StruDataFileBase
{
	StruDataMatFile();
	virtual ~StruDataMatFile();
	virtual MatrData* GetMapData(string name);

	bool GetNameDataFromMat(string fileName, string* key, int n);

	MapMatr* m_pmpData;
};