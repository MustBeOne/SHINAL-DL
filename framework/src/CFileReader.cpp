#include "CFileReader.h"
#include <mat.h>


StruDataFileBase::StruDataFileBase()
{
	fileType = EmFileType::NONE;
}

StruDataMatFile::StruDataMatFile()
{
	fileType = EmFileType::MAT_FILE;
	m_pmpData = new MapMatr;
}

StruDataMatFile::~StruDataMatFile()
{
	for (auto it : *m_pmpData)
	{
		FreeObj(it.second);
	}
	FreeObj(m_pmpData);
}

MatrData* StruDataMatFile::GetMapData(string name)
{
	auto ite = m_pmpData->end();
	auto it = m_pmpData->find(name);
	if (it == ite)
	{
		cout << "No Such Data!" << endl; 
		return nullptr;
	}
	else
	{
		return it->second;
	}
}

bool StruDataMatFile::GetNameDataFromMat(string fileName, string* key, int n)
{
	MATFile* pmat;
	mxArray* matArr;
	pmat = matOpen(fileName.c_str(), "r");
	if (pmat == NULL) {
		printf("Error creating file %s\n", fileName.c_str());
		printf("(Do you have write permission in this directory?)\n");
		return false;
	}
	string name;
	int i;
	for (i = 0; i < n; ++i)
	{
		name = key[i];
		matArr = matGetVariable(pmat, name.c_str());
		if (matArr == NULL) {
			printf("Error reading existing matrix LocalDouble\n");
			return false;
		}
		if (mxGetNumberOfDimensions(matArr) != 2) {
			printf("Error saving matrix: result does not have two dimensions\n");
			return false;
		}
		int rows, cols, i, j, n = 0;
		rows = mxGetM(matArr);
		cols = mxGetN(matArr);
		MatrData* data = new MatrData(rows, cols);
		Scalar* dataPtr = mxGetPr(matArr);
		memcpy(&(*data)(0, 0), dataPtr, rows * cols * sizeof(Scalar));
		m_pmpData->insert(make_pair(name, data));
		mxDestroyArray(matArr);
	}
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", fileName.c_str());
		return false;
	}
	return true;
}

