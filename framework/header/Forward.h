#pragma once
#include "tools.h"

class ForwardBase {
public:
	ForwardBase() {};
	virtual ~ForwardBase() {};
	virtual void Forward(CTensor*, VecTsr*, VecTsr*) {};
	virtual void IsLeft() {};
	virtual void SetAnotherNode(CTensor*) {};
	virtual void SetFactor(Scalar, Scalar) {};

	CTensor* GetCurNode();
	CTensor* GetNxtNode();
	void FreeForward();
protected:
	CTensor* pNode;
	CTensor* nextNode;
};

class NoneForward :public ForwardBase {
public:
	NoneForward();
	virtual ~NoneForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
};

class AddForward :public ForwardBase {
public:
	AddForward(CTensor*, CTensor*);
	virtual ~AddForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
};

class SubForward :public ForwardBase {
public:
	SubForward(CTensor*, CTensor*);
	virtual ~SubForward();
	virtual void IfLeft();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
private:
	bool ifLeft;
};

class MulForward :public ForwardBase {
public:
	MulForward(CTensor*, CTensor*);
	virtual ~MulForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
	virtual void SetAnotherNode(CTensor*);
private:
	CTensor* anotherNode;
};

class DivForward :public ForwardBase {
public:
	DivForward(CTensor*, CTensor*);
	virtual ~DivForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
	virtual void SetAnotherNode(CTensor*);
	virtual void IsLeft();
private:
	CTensor* anotherNode;
	bool isLeft;
	CTensor* denominator;
};

class CpyForward :public ForwardBase {
public:
	CpyForward(CTensor*, CTensor*);
	virtual ~CpyForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
};

class PowForward :public ForwardBase {
public:
	PowForward(CTensor*, CTensor*);
	virtual ~PowForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
	virtual void SetFactor(Scalar, Scalar);
private:
	Scalar sign;
	Scalar powCoe;
};

class MmForward :public ForwardBase {
public:
	MmForward(CTensor*, CTensor*);
	virtual ~MmForward();
	virtual void Forward(CTensor*, VecTsr*, VecTsr*);
	virtual void SetAnotherNode(CTensor*);
	virtual void IsLeft();
private:
	bool isLeft;
	CTensor* anotherNode;
};

class ExpForward :public ForwardBase {
public:
	ExpForward(CTensor*, CTensor*);
	virtual ~ExpForward() {};
	virtual void Forward(CTensor*, VecTsr*, VecTsr*) {};
	virtual void SetAnotherNode(CTensor*) {};
	virtual void IsLeft() {};
private:
};