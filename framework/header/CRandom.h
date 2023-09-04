#pragma once
#include "tools.h"


using std::default_random_engine;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::uniform_int_distribution;

template<class TenData = Scalar>
class CRandom {
public:
	CRandom() {};
	virtual ~CRandom() {};

	static TenData* RandomChoice(TenData*, int, int, bool ifReplace = false);
	static PtrTen NormalTensor(int, int, Scalar mean = 0., Scalar stddev = 1.);
	static PtrTen UniformTensor(int, int, Scalar lb = 0., Scalar ub = 1.);
	static void Shuffle(TenData, TenData);
public:
	static default_random_engine generator;
};

template<class TenData> default_random_engine CRandom<TenData>::generator = default_random_engine((unsigned int)time(0));


//-----------------------------------------------
template<class TenData>
TenData* CRandom<TenData>::RandomChoice(TenData* pData, int constSize, int sampleSize, bool ifReplace /*= false*/) {
	generator.seed(generator());
	if (constSize < sampleSize && ifReplace) {
		std::cout << "Cannot take a larger sample than population when 'replace=False'" << std::endl;
		return nullptr;
	}
	TenData* res = new TenData[sampleSize];
	if (!(constSize < sampleSize)) {
		sample(pData, pData + constSize - 1, res, sampleSize, generator);
	}
	else {
		uniform_int_distribution<int> distribution(0, constSize - 1);
		int i;
		for (i = 0; i < sampleSize; ++i) {
			res[i] = pData[distribution(generator)];
		}
	}
	return res;
}

template<class TenData>
PtrTen CRandom<TenData>::UniformTensor(int row, int col, Scalar lb /*= 0*/, Scalar ub /*= 1*/) {
	generator.seed(generator());
	uniform_real_distribution<> distribution(lb, ub);
	MatrData* data = new MatrData(row, col);
	int size = row * col;
	TenData* temp = new TenData[size];
	int i;
	for (i = 0; i < size; ++i) {
		temp[i] = distribution(generator);
	}
	memcpy(&(*data)(0, 0), temp, size * sizeof(Scalar));
	PtrTen ten = new CTensor(*data, true);
	FreeObj(data);
	delete[] temp;
	return ten;
}


//the tensor returned by this function is ifgrad=true
template<class TenData>
PtrTen CRandom<TenData>::NormalTensor(int row, int col, Scalar mean, Scalar stddev) {
	Scalar t;
	generator.seed(generator());
	normal_distribution<TenData> distribution(mean, stddev);
	MatrData* data = new MatrData(row, col);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			(*data)(i, j) = distribution(generator);
		}
	}
	PtrTen ten = new CTensor(*data, true);
	FreeObj(data);
	return ten;
}


template<class TenData>
void CRandom<TenData>::Shuffle(TenData begin, TenData end) {
	shuffle(begin, end, generator);
}