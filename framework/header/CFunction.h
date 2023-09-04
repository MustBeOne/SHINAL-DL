#pragma once
#include "CModuleBase.h"
#include "tools.h"


class CActivationsFunction :public CFunctionBase {
public:
	CActivationsFunction() { tempA = nullptr; };
	virtual ~CActivationsFunction() {};
protected:
	PtrTen tempA;
};

class CLossFunction :public CFunctionBase {
public:
	CLossFunction() { tempLoss = nullptr; };
	virtual ~CLossFunction() {};
protected:
	PtrTen tempLoss;
};
