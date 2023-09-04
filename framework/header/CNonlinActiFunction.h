#pragma once
#include "CFunction.h"

///-----------------Activation Function---------------------

//-------Sigmoid Activation Function-------------
class CSigmoidAF :public CActivationsFunction {
public:
	CSigmoidAF() {};
	virtual ~CSigmoidAF() { FreeObj(tempA); };

	virtual PtrTen Forward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat);
};

//-------Tanh Activation Function-------------
class CTanhAF :public CActivationsFunction {
public:
	CTanhAF() {};
	virtual ~CTanhAF() { FreeObj(tempA); };

	virtual PtrTen Forward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat);
};

//-------ReLU Activation Function-------------

class CReLUAF :public CActivationsFunction 
{
public:
	CReLUAF(bool ifInPlace = false);
	virtual ~CReLUAF();

	virtual PtrTen Forward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat);
private:
	bool ifInPlace;
};