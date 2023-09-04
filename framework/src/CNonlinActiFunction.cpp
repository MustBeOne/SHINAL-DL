#include "CNonlinActiFunction.h"
#include "CTensor.h"
#include "Backward.h"
#include "FuncBackward.h"


///-----------------Activation Function---------------------

//-------Sigmoid Activation Function-------------
PtrTen CSigmoidAF::Forward(PtrTen z) {
	tempA = new CTensor(MatrixOp::Sigmoid(z->GetData()->GetDataPtr()), true, false);
	tempA->SetLeafSta(false);
	BackwardBase* bw = new SigmoidBackward();
	auto data = tempA->GetData()->GetDataPtr()->array();
	MatrData* dataGn = new MatrData((1. - data) * data);
	tyTensor* raw = new tyTensor(dataGn, false);
	bw->SaveNextFuns(z, raw);
	bw->SaveNode(tempA);
	tempA->SetGradFn(bw);

	tempA->SetOrigin(z->GetOrigin());
	raw = nullptr;
	return tempA;
}


PtrMat CSigmoidAF::ForwardNonGrad(PtrMat z) {
	PtrMat result = new struMat(MatrixOp::Sigmoid(z->GetDataPtr()), false);
	FreeObj(z);
	return result;
}


//-------Tanh Activation Function-------------

PtrTen CTanhAF::Forward(PtrTen z) {
	tempA = new CTensor(MatrixOp::Tanh(z->GetData()->GetDataPtr()), true, false);
	tempA->SetLeafSta(false);
	BackwardBase* bw = new TanhBackward();
	auto data = tempA->GetData()->GetDataPtr()->array();
	MatrData* dataGn = new MatrData(1. - data.pow(2));
	tyTensor* raw = new tyTensor(dataGn, false);
	bw->SaveNextFuns(z, raw);
	bw->SaveNode(tempA);
	tempA->SetGradFn(bw);

	tempA->SetOrigin(z->GetOrigin());
	raw = nullptr;
	return tempA;
}

PtrMat CTanhAF::ForwardNonGrad(PtrMat z)
{
	PtrMat result = new struMat(MatrixOp::Tanh(z->GetDataPtr()), false);
	FreeObj(z);
	return result;
}

//-------ReLU Activation Function-------------

CReLUAF::CReLUAF(bool ifInPlace)
{
	this->ifInPlace = ifInPlace;
}

CReLUAF::~CReLUAF()
{
	if (!ifInPlace)
	{
		FreeObj(tempA);
	}
}

PtrTen CReLUAF::Forward(PtrTen z)
{
	if (ifInPlace)
	{
		MatrData* t = z->GetData()->GetDataPtr();
		MatrixOp::ReLU(t);
	}
	else
	{

	}
	tempA = new CTensor(MatrixOp::Tanh(z->GetData()->GetDataPtr()), true, false);
	tempA->SetLeafSta(false);
	BackwardBase* bw = new TanhBackward();
	auto data = tempA->GetData()->GetDataPtr()->array();
	MatrData* dataGn = new MatrData(1. - data.pow(2));
	tyTensor* raw = new tyTensor(dataGn, false);
	bw->SaveNextFuns(z, raw);
	bw->SaveNode(tempA);
	tempA->SetGradFn(bw);

	tempA->SetOrigin(z->GetOrigin());
	raw = nullptr;
	return tempA;
}

PtrMat CReLUAF::ForwardNonGrad(PtrMat z)
{
	return nullptr;
}
