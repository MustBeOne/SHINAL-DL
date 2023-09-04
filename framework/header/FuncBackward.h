#pragma once
#include "CAbstractTemplate.h"
#include "Backward.h"

class FuncBackwardBase:public BackwardBase
{
public:
	FuncBackwardBase() {};
	virtual ~FuncBackwardBase() {};
	virtual void Backward(tyTensor*) {};
	virtual void Backward(Scalar) {};
	virtual void ShowThis() {};
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool) {};
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*) {};
};


class QuadraPolyBackward :public FuncBackwardBase
{
public:
	QuadraPolyBackward(Scalar a = 0., Scalar b = 1.);
	virtual ~QuadraPolyBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
protected:
	virtual CTensor* CmpPostPG(VecTsr*);
private:
	Scalar a, b;
};


//-----------------------the Activation Functions Backward--------------------------
class SigmoidBackward :public FuncBackwardBase {
public:
	SigmoidBackward();
	virtual ~SigmoidBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
protected:
	virtual CTensor* CmpPostPG(VecTsr*);
private:
};

class TanhBackward :public FuncBackwardBase {
public:
	TanhBackward();
	virtual ~TanhBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
protected:
	virtual CTensor* CmpPostPG(VecTsr*);
private:
};