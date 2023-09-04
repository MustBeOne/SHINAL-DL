#pragma once
#include "CModuleBase.h"	
#include "Mnist.h"
#include "tools.h"

struct struDNNParam :public NNParamBase {
	struDNNParam();
	virtual ~struDNNParam();
	string trainX, trainY, testX, testY;
	bool ifEval, ifBatch;
	int epoches;
	int btcSize, trainLeng, testLeng;
	CDatasetLoader<StruRawDataMnist>* m_pDataset;
	void InitDataset();
	FuncEval eval;
};

//-------------------------BP neural network--------------------
class C_DNN :public CModuleBase {
public:
	C_DNN();
	C_DNN(PtrSeq, struDNNParam*);
	virtual ~C_DNN();
protected:
	//neural network
	virtual PtrTen ForwardNet(PtrTen);
	virtual vector<PtrTen>* ForwardNet(vector<PtrTen>*);
	virtual PtrTen Loss(PtrTen, PtrTen);
	virtual vector<PtrTen>* Loss(vector<PtrTen>*, vector<PtrTen>*);
	virtual void Backward(vector<PtrTen>*);
	virtual void Evaluate(MatrData***);
private:
	void TrainBatch(struBatchedData*);
public:
	virtual void LoadComponents(PtrLossFunc, PtrOptm);
	virtual void Train();

	ItParam Parameters();
private:
	int prednum;
	PtrSeq m_pSeq;
	struDNNParam* m_pParam;
	PtrLossFunc m_pLossFn;
	PtrOptm m_pOptimizer;

	PtrPara m_pParameter;
};
