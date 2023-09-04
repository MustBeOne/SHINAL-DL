#pragma once
#include "CModuleBase.h"
#include "PINNs_BurgersEquation.h"

struct StruPINNsParam:public NNParamBase
{
	StruPINNsParam();
	virtual ~StruPINNsParam();
	string dataMatFile;
	Scalar Nu;
	CDatasetLoader<StruPINNsBurgEqua>* m_pDataset;
	void InitDataset();
};

class C_PINNs :public CModuleBase {
private:
	int nIter;
	PtrMat XPred;
	PtrTen Xu, Tu;
	PtrTen Xf, Tf;
	PtrTen aa, aaa;
	PtrTen U;
public:
	C_PINNs();
	C_PINNs(PtrSeq, StruPINNsParam*);
	virtual ~C_PINNs();
protected:
	//neural network
private:
	PtrTen NetU();
	PtrTen NetF();
public:
	virtual void Init();
	virtual void LoadComponents(PtrLossFunc, PtrOptm);
	virtual void Train();
	virtual PtrTen Closure();
	virtual void Predict();

	ItParam Parameters();
private:
	PtrSeq m_pSeq;
	StruPINNsParam* m_pParam;
	PtrLossFunc m_pLossFn;
	PtrOptm m_pOptimizer;

	PtrPara m_pParameter;
};

