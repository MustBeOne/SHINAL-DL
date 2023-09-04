#pragma once
#include "tools.h"


struct struSeqBase {
	struSeqBase() {};
	virtual ~struSeqBase() {};
	virtual void SetParameter(PtrPara) {};
	virtual void AddModule(CLinearLayerBase*) {};
	virtual void AddModule(CActivationsFunction*) {};
	virtual void AddModule(string, CLinearLayerBase*) {};
	virtual void AddModule(string, CActivationsFunction*) {};

	virtual PtrTen SeqForward(PtrTen) { return nullptr; };
	virtual PtrMat ForwardNonGrad(PtrMat) { return nullptr; };
};
struct struSeqVec :public struSeqBase {
	struSeqVec();
	virtual ~struSeqVec();
	virtual void SetParameter(PtrPara);
	virtual void AddModule(CLinearLayerBase*);
	virtual void AddModule(CActivationsFunction*);

	virtual PtrTen SeqForward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat);
	vector<CModuleBase*> vecSeq;
};
struct struSeqMap :public struSeqBase {
	struSeqMap();
	virtual ~struSeqMap();
	virtual void SetParameter(PtrPara);
	virtual void AddModule(string, CLinearLayerBase*) {
		//
	};
	virtual void AddModule(string, CActivationsFunction*) {
		//
	};
	virtual PtrTen SeqForward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat) {
		//
		return nullptr;
	};
	map<string, CModuleBase*> mapSeq;
};

