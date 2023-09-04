#pragma once
#include "tools.h"

class CModuleBase;

class COptimizerBase {
public:
	COptimizerBase() {};
	virtual ~COptimizerBase() {};
	virtual void ZeroGrad() {};
	virtual void Step(/*FuncClosure closure = nullptr, */CModuleBase* pMod = nullptr) {};
};

//-----------------this is SGD optimizer-------------------
class COptimizerSGD :public COptimizerBase {
public:
	COptimizerSGD(
		ItParam params,
		Scalar   lr,
		Scalar   momentum = 0,
		Scalar   WeightDecay = 0,
		Scalar   dampening = 0,
		bool     nesterov = false,
		bool	  maximize = false,
		bool     foreach = false
	);
	virtual ~COptimizerSGD();
	virtual void ZeroGrad();
	//virtual void Step(FuncClosure closure = nullptr, CModuleBase* pMod = nullptr);
	virtual void Step(CModuleBase* pMod = nullptr);
private:
	ItParam   itParams;
	Scalar    lr;
	Scalar    momentum;
	Scalar    WeightDecay;
	Scalar    dampening;
	bool      nesterov;
	bool	  maximize;
	bool      foreach;
private:
	bool ifFirst = true;
};

//-----------------this is L-BFGS optimizer-------------------
typedef int (*FuncLineSearch)();

enum class emLineSearFn :unsigned char {
	StrongWolfe
};

enum class emBackTracking : unsigned char {
	LineSearchArmijo,
	LineSearchWeakWolfe,
	LineSearchStrongWolfe
};

struct struLBFGSCtrlParam {
	int maxLineSrch;
	Scalar minStep;
	Scalar maxStep;
	Scalar wolfeC1;
	Scalar wolfeC2;
	Scalar ToleranceChange;
	emBackTracking backtrack;
};

static struLBFGSCtrlParam defParam =
{
	25,1.0e-20,1.0e+20,1.0e-4,0.9,1.0e-9,
	emBackTracking::LineSearchStrongWolfe
};


class COptimizerLBFGS :public COptimizerBase {
public:
	COptimizerLBFGS(
		ItParam params,
		Scalar lr = 1,
		int maxIter = 20,
		Scalar maxEval = 0,
		Scalar toleranceGrad = 1e-7,
		Scalar toleranceChange = 1e-9,
		int histSize = 100,
		emLineSearFn lineSrchFn = emLineSearFn::StrongWolfe,
		struLBFGSCtrlParam* ctrlPtr = nullptr
	);
	virtual~COptimizerLBFGS();
	virtual void ZeroGrad();
	virtual void Step(/*FuncClosure closure = nullptr,*/ CModuleBase* pMod = nullptr);
private:
	void FlatGrad(VectData*& flatedGrad);
	//Scalar GetLossValue(FuncClosure, CModuleBase*);
	Scalar GetLossValue(CModuleBase*);
	void LineSearchByStrongWolfe(
		Scalar& step,
		VectData*,
		Scalar&,
		VectData* flatGrad,
		Scalar,
		int&,
		/*FuncClosure,*/
		CModuleBase*);
	//Scalar EvalInDesDir(FuncClosure, CModuleBase*, Scalar, VectData*, VectData*&);
	Scalar EvalInDesDir(CModuleBase*, Scalar, VectData*, VectData*&);
	void AddParams(Scalar, VectData*);
	void ResetParams(VecParamLBFGS*);
	void CallbackConverExit();
	VecParamLBFGS* SavePreParam();
	void ReleasePreParamMemo(VecParamLBFGS*);
	Scalar CubicInterpolate(Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, pair<Scalar, Scalar>* bound = nullptr);
private:
	ItParam		   itParams;
	Scalar		   lr;
	int			   maxIter;
	Scalar		   maxEval;
	Scalar		   toleranceGrad;
	Scalar		   toleranceChange;
	int			   histSize;
	emLineSearFn lineSrchFn;
	struLBFGSCtrlParam* ctrlPtr;
private:
	int nParamDim;
	vector<void*>* state;
};

