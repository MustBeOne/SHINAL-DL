#pragma once
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <type_traits>
#include <time.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <utility>
#include <math.h>
#include "CAbstractTemplate.h"
//#define  show

using std::default_random_engine;
using std::normal_distribution;

template <class T>
T* NewObj() {
	T* pData = new T();
	return pData;
}

template <class T>
T* New1DArr(int n) {
	T* pData = new T[n];
	for (int j = 0; j < n; ++j) {
		pData[j] = 0;
	}
	return pData;
}

template <class T>
T** New2DArr(int m, int n) {
	T** pData = new T * [m];
	for (int i = 0; i < m; i++) {
		pData[i] = new T[n];
		for (int j = 0; j < n; ++j) {
			pData[i][j] = 0;
		}
	}
	return pData;
}
/// <summary>
/// add a series of son-class object to a father-class vector 
/// whose size is indicated by the param num
/// </summary>
/// <typeparam name="TFa">the father class type</typeparam>
/// <typeparam name="TSo">the son class type</typeparam>
/// <param name="vc">the vector container</param>
/// <param name="num">the size of the son-class objects</param>
template<class TFa, class TSo>
void AddBatchDataVect(vector<TFa*> vc, int num) {
	TSo* temp;
	int i;
	for (i = 0; i < num; ++i) {
		temp = NewObj<TSo>();
		vc.push_back(temp);
	}
}
/// <summary>
/// clear the indicated map object
/// </summary>
/// <param name="tmap">the input map object need to be clear</param>
template<class T1, class T2>
void ClearMapData(map<T1, T2>& tmap) {
	typename map<T1, T2>::iterator it, its, ite;
	for (it = its; it != ite; ++it) {
		FreeObj(it->second);
	}
	tmap.clear();
}

template<class Ty>
vector<Ty*>* CopyVecElewise(vector<Ty*>* vec)
{
	vector<Ty*>* res = new vector<Ty*>;
	Ty* temp;
	for (auto it : *vec)
	{
		temp = new Ty(*it);
		res->push_back(temp);
	}
	return res;
}

template<class T>
T ReverseData(T n) {
	int nByte = sizeof(T), sft;
	unsigned char* b = new unsigned char[nByte];
	T newnum = 0;
	for (int i = 0; i < nByte; ++i) {
		sft = i * 8;
		b[i] = (n >> sft) & 0xff;
	}
	for (int i = nByte - 1; i >= 0; --i) {
		newnum |= (b[i] << ((nByte - 1 - i) * 8));
	}
	return newnum;
}

//////////////////////////////////////////////////////////////////////////

template<class T>
void FreeObj(T*& pData) {
	if (!pData) return;
	delete pData;
	pData = nullptr;
}

template<class T>
void FreeObj(T*&& pData) {
	if (!pData) return;
	delete pData;
	pData = nullptr;
}

template<class T>
void Free1DArr(T*& pData) {
	if (!pData) return;
	delete[] pData;
	pData = nullptr;
}
template<class T>
void Free2DArr(T*& pData, int n1) {
	if (!pData) return;
	for (int n = 0; n < n1; n++) {
		if (!pData[n])
		{
			continue;
		}
		delete[] pData[n];
		pData[n] = nullptr;
	}
	delete[] pData;
	pData = nullptr;
}

template <class T>
void FreeMapObj(map<string, T*>& mpData) {
	typedef map<string, T*>::iterator itType;
	itType it, its, ite;
	its = mpData.begin();
	ite = mpData.end();
	for (it = its; it != ite; it++) {
		FreeObj(it->second);
	}
}

template<class T>
void ClearObjVec(vector<T>* vec) {
	if (!vec) {
		return;
	}
	auto it = vec->begin();
	auto ite = vec->end();
	for (; it != ite; ++it) {
		FreeObj(*it);
	}
	vec->clear();
}

template<class T>
VectData GetRandomVect(std::default_random_engine& e, DistriType type, int leng, Scalar param1 = -1, Scalar param2 = -1) {
	VectData vec(leng);
	normal_distribution<T> n;
	switch (type) {
	case DistriType::Normal:n = normal_distribution<T>(param1, param2);
		break;
	case DistriType::Poisson:
		break;
	default:
		break;
	}
	e.seed(e());
	for (int i = 0; i < leng; ++i) {
		vec(i) = n(e);
	}
	return vec;
}
template<class T>
MatrData GetRandomMatr(std::default_random_engine& e, DistriType type, int row, int col, Scalar param1 = -1, Scalar param2 = -1) {
	MatrData mat(row, col);
	normal_distribution<T> n;
	switch (type) {
	case DistriType::Normal:n = normal_distribution<T>(param1, param2);
		break;
	case DistriType::Poisson:
		break;
	default:
		break;
	}
	e.seed(e());
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			mat(i, j) = n(e);
		}
	}
	return mat;
}
template<class T>
T* GetShuffleArr(int size, int seed) {
	T* arr = new T[size];
	for (int i = 0; i < size; ++i) {
		arr[i] = i;
	}
	srand(seed);
	random_shuffle(arr, arr + size - 1);
	return arr;
}

class CTools {
public:
	CTools();
	virtual ~CTools();

public:
	static int* AllocI(int n1);
	static Scalar* Alloc(int n);
	static Scalar** Alloc(int n1, int n2);

	static void FreeI(int*& val);
	static void Free(Scalar*& val);
	static void Free(Scalar**& val, int n1);

	static bool IsEqual(Scalar v1, Scalar v2);
	static bool IsZero(Scalar v);
	static int Round(Scalar val);

	static void SaveDataFile(string fileName, int n1, int n2, Scalar** val);
	static void Save1DataFile(string fileName, int n1, Scalar* val);
	static void ProcLineStrTwoParam(string& str, string& unit1, string& unit2);

	static void PrintProcessInWin(int, int);
	static void PrintTestProcessInWin(int, int);
};