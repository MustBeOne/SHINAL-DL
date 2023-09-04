#pragma once
#include "tools.h"

struct NNParamBase {
	NNParamBase() {};
	virtual ~NNParamBase() {};
};


class CModuleBase {
public:
	CModuleBase() {};
	virtual ~CModuleBase() {};

	//sequential layer
	virtual void Forward() {};
	virtual CTensor* Forward(CTensor*) { return  nullptr; };
	virtual struMat* ForwardNonGrad(struMat*) { return  nullptr; };
	virtual void SetParam(CParameter*) {};
	//loss function
	virtual CTensor* CmpLoss(CTensor*, CTensor*) { return  nullptr; };
	//activation function
	virtual CTensor* Activate(CTensor*) { return  nullptr; };
protected:
	//neural network
	///for singal sample
	virtual CTensor* ForwardNet(CTensor*) { return  nullptr; };
	virtual CTensor* Loss(CTensor*, CTensor*) { return  nullptr; };
	///for batch sample
	virtual vector<CTensor*>* ForwardNet(vector<CTensor*>*) { return  nullptr; };
	virtual vector<CTensor*>* Loss(vector<CTensor*>*, vector<CTensor*>*) { return  nullptr; };
	virtual void Backward(vector<CTensor*>*) {};

	virtual void Evaluate() {};
public:
	virtual void Init() {};
	virtual void Train() {};
	virtual void LoadComponents() {};
	virtual CTensor* Closure() { return nullptr; };
	virtual void Predict() {};
};

class CLayerBase :public CModuleBase {
public:
	CLayerBase() {};
	virtual ~CLayerBase() {};
	//the Linear Layer
	virtual CTensor* GetWeight() { return  nullptr; };
	virtual CTensor* GetBias() { return  nullptr; };
};

class CFunctionBase :public CModuleBase {
public:
	CFunctionBase() {};
	virtual ~CFunctionBase() {};

private:

};
