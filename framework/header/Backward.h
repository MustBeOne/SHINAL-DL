#if 1
#pragma once
#include "tools.h"

template<class TyBw>
using FuncPG = CTensor * (TyBw::*)(VecTsr*);
class BackwardBase {
public:
	BackwardBase();
	virtual ~BackwardBase();
	virtual void Backward(tyTensor*) {};
	virtual void Backward(Scalar) {};
	virtual void ShowThis() {};
	virtual void SetFactors(Scalar c1 = 0, Scalar = 0) {};

	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool) {};
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*) {};

	void SaveNode(CTensor*);
	void SaveNextFuns(CTensor*, tyTensor*);
	void DeleteInleafNode(bool ifRetainSelf = false);
	void ReleaseGrad();
	void ReleaseFwGraph();
	void ProcVrtScal(Scalar);
	EmBackwardType GetBwType();
	MpFunDepend* GetDependNode();

protected:
	void FreeRawGrad();
	virtual CTensor* CmpPostPG(VecTsr*) { return nullptr; };
	bool IfAutogradEnd(CTensor*&, CTensor*, VecTsr*, VecTsr*, bool&);
	bool IfAutogradEnd(Scalar&, CTensor*, VecTsr*);
	bool ProcScalar(Scalar, CTensor*&, VecTsr*);
protected:
	EmBackwardType emBwType;
	MpFunDepend* m_mpNextFuns;
	CTensor* m_pTNode;

	Scalar vrtScaler;
	CTensor* postPG;
};


class NoneBackward :public BackwardBase {
public:
	NoneBackward();
	virtual ~NoneBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool) {};
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*) {};
private:
};

class AccumulateGrad :public BackwardBase {
public:
	AccumulateGrad();
	virtual ~AccumulateGrad();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class NegBackward :public BackwardBase {
public:
	NegBackward();
	virtual ~NegBackward() {};
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class CopyBackward :public BackwardBase {
public:
	CopyBackward();
	virtual ~CopyBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class AddBackward :public BackwardBase {
public:
	AddBackward();
	virtual ~AddBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

/*
* The first of the m_mpNextFuns must be the left-node of the SUB
* Second must be the right-node of the SUB.
*/
class SubBackward :public BackwardBase {
public:
	SubBackward();
	virtual ~SubBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class MulBackward :public BackwardBase {
public:
	MulBackward();
	virtual ~MulBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void SetFactors(Scalar scaler = 1, Scalar placeholder = 0);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	Scalar scaler;
};

class DivBackward :public BackwardBase {
public:
	DivBackward();
	virtual ~DivBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void SetFactors(Scalar fac = 1, Scalar pos = 0);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	/*
	* 0: the left is scaler
	* 1: the right is scaler
	*/
	Scalar fac, pos;
};

class SumBackward :public BackwardBase {
public:
	SumBackward();
	virtual ~SumBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
private:
};

class MmBackward :public BackwardBase {
public:
	MmBackward();
	virtual ~MmBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
	void SetTransSta(bool lT = false, bool rT = false);
	void SetCoeff(int rows, int cols, Scalar scaler, int pos = 0);
private:
	/*
	* 0: the left is scaler
	* 1: the right is scaler
	*/
	Scalar scaler;
	int rows, cols, pos;
	bool lT, rT;
};

class AddmmBackward :public BackwardBase {
public:
	AddmmBackward();
	virtual ~AddmmBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void SetFactors(Scalar beta = 1, Scalar alpha = 1);
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	Scalar beta, alpha;
};

class PowBackward :public BackwardBase {
public:
	PowBackward();
	virtual ~PowBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void SetFactors(Scalar powCoe = 1, Scalar sign = 1);
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	Scalar powCoe;
	Scalar sign;
};

class ExpBackward :public BackwardBase {
public:
	ExpBackward();
	virtual ~ExpBackward();
	virtual void SetFactors(Scalar coeExp = 1, Scalar sign = 1);
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	Scalar coeExp;
	Scalar sign;
};

class AccBackward :public BackwardBase {
public:
	AccBackward();
	virtual ~AccBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class MeanBackward :public BackwardBase {
public:
	MeanBackward();
	virtual ~MeanBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
};

class CatBackward :public BackwardBase {
public:
	CatBackward();
	virtual ~CatBackward();
	virtual void SetFactors(Scalar dim = 0, Scalar mtc1 = 0);
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	int dim;
	int mtc1;
};
class SliceBackward :public BackwardBase {
public:
	SliceBackward(int, int, int);
	virtual ~SliceBackward();
	virtual void Backward(tyTensor*);
	virtual void Backward(Scalar);
	virtual void ShowThis();
	virtual void BackwardInAuto(CTensor*, CTensor*, VecTsr*, VecTsr*, bool);
	virtual void BackwardInAuto(Scalar, CTensor*, VecTsr*, VecTsr*);
private:
	void ProcOnesMat();
	MatrData* dataPG;
	int dim;
	int leng;
	int pos;
};

//-----------------------the Loss Functions Backward--------------------------
class MseLossBackward :public BackwardBase {
public:
	MseLossBackward();
	virtual ~MseLossBackward();
	virtual void Backward(tyTensor*);
	virtual void ShowThis();
private:
};
#endif