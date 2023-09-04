#pragma once
#include "tools.h"

struct struParameter {
	struParameter(int, int);
	virtual ~struParameter();
	PtrTen data;
};

class CParameter {
public:
	CParameter() { vecMods = nullptr; };
	virtual ~CParameter() {};

	void SetParam(PtrTen);
	void SetMods(VecModule*);
	void SetMods(CModuleBase*);

	VecParam* GetParamsPtr() { return &vecParas; };

	void MeanLrParamGrad(int base);
#ifdef show
	void Show();
#endif
	ItParam operator()();
private:
	VecParam vecParas;
	VecModule* vecMods;
};