#pragma once
#include "CModuleBase.h"
#include "tools.h"

/// <summary>
/// the Linear Layers Class,including Identity,Linear(all-connect)
/// Bilinear,LazyLinear
/// </summary>

class CLinearLayerBase :public CLayerBase {
public:
	CLinearLayerBase() {};
	virtual ~CLinearLayerBase() {};
};

/// <summary>
/// the all-connect layer
/// </summary>
class CLinear :public CLinearLayerBase {
public:
	CLinear(int inFeatures, int outFeatures, bool ifBias = true);
	virtual ~CLinear();

	virtual PtrTen Forward(PtrTen);
	virtual PtrMat ForwardNonGrad(PtrMat);
	virtual void SetParam(PtrPara);
private:
	PtrTen m_pWeight;
	PtrTen m_pBias;
	PtrTen tempZ;
};

