#pragma once
#include "tools.h"
#include <mat.h>
#include <fstream>
#include <ctime>

using std::fstream; 
using std::endl;
using std::cout;

//#############################################################################################
//##################################### The Tool Functions ####################################
//#############################################################################################
string GetCPUInfo();
string GetOsInfo();


template<class TyData = Scalar>
class CFileWriter
{
public:
	CFileWriter() {};
	virtual ~CFileWriter() {};

	static bool WriteMatrToFile_Mat(string filename, string* matnames, MatrData** mats, int N);
	static bool WriteMatrToFile_Txt(string filename, string* matnames, MatrData** mats, int N);
	static bool WriteDataToFile_Mat(string filename, string* matnames, TyData*** mats, int N);
};


//########################################################################################
//##################################### The Implement ####################################
//########################################################################################


#define If


template<class TyData>
bool CFileWriter<TyData>::WriteMatrToFile_Mat(string filename, string* matnames, MatrData** mats, int N)
{
	if (!filename.ends_with(".mat"))
	{
		filename = filename + ".mat";
	}
	int i, row, col, size;
	MATFile* pmat;
	mxArray* pa;
	string name;
	MatrData* data;
	matError status;
	pmat = matOpen(filename.c_str(), "w");
	if (pmat == NULL) {
		printf("Error creating file %s\n", filename);
		printf("(Do you have write permission in this directory?)\n");
		return false;
	}
	for (i = 0; i < N; ++i)
	{
		name = matnames[i];
		data = mats[i];
		row = data->rows();
		col = data->cols();
		size = row * col;
		pa = mxCreateDoubleMatrix(row, col, mxREAL);
		if (pa == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return false;
		}
		memcpy((void*)(mxGetPr(pa)), (void*)(&(*data)(0, 0)), size * sizeof(Scalar));
		status = matPutVariable(pmat, name.c_str(), pa);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return false;
		}
		/* clean up */
		mxDestroyArray(pa);
	}
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", filename.c_str());
		return false;
	}
	return true;
}


#pragma warning( disable : 4996 )
template<class TyData /*= Scalar*/>
bool CFileWriter<TyData>::WriteMatrToFile_Txt(string filename, string* matnames, MatrData** mats, int N)
{
	if (!filename.ends_with(".txt"))
	{
		filename = filename + ".txt";
	}
	fstream ofs;
	string name;
	MatrData* data;
	ofs.open(filename.c_str(), std::ios::out);
	if (!ofs.is_open())
	{
		cout << "Open file : " << filename << " error!" << endl;
		return false;
	}
	//process the time header
	std::time_t t = std::time(nullptr);
	std::tm* now = std::localtime(&t);

	char timebuf[128];
	string cpuInfo, osInfo;
	cpuInfo = GetCPUInfo();
	osInfo = GetOsInfo();
	strftime(timebuf, sizeof(timebuf), "%c", now);
	ofs << "Platform: " << osInfo << " " << cpuInfo << "\nCreated on: " << timebuf << "\n\n";

	int i, row, col, size;
	for (i = 0; i < N; ++i)
	{
		name = matnames[i];
		data = mats[i];
		ofs << name << ":\n";
		ofs << *data << '\n' << '\n';
	}
	ofs.close();
	return true;
}

template<class TyData /*= Scalar*/>
bool CFileWriter<TyData>::WriteDataToFile_Mat(string filename, string* matnames, TyData*** mats, int N)
{

}


