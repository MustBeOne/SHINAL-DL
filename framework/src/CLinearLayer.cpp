#include "CLinearLayer.h"
#include "tools.h"
#include "CParameter.h"
#include "CTensor.h"
#include "Backward.h"
#include "CRandom.h"
class CLinearLayer;
CLinear::CLinear(int inFeatures, int outFeatures, bool ifBias) {
	Scalar bound;
	bound = 1. / sqrt(inFeatures);
	m_pWeight = CRandom<Scalar>::UniformTensor(outFeatures, inFeatures, -bound, bound);
	ifBias ? m_pBias = CRandom<Scalar>::UniformTensor(outFeatures, 1, -bound, bound) : m_pBias = nullptr;
	tempZ = nullptr;
}

CLinear::~CLinear() {
	FreeObj(m_pWeight);
	FreeObj(m_pBias);
	FreeObj(tempZ);
}

PtrTen CLinear::Forward(PtrTen a) {
	tempZ = CTensor::AddMM(1., m_pBias, 1., m_pWeight, a);
	return tempZ;
}

PtrMat CLinear::ForwardNonGrad(PtrMat input) {
	PtrMat temp = input;
	input = m_pWeight->GetData()->Dot(temp);
	FreeObj(temp);
	*input += *m_pBias->GetData();
	return input;
}

void CLinear::SetParam(PtrPara para) {
	para->SetParam(m_pWeight);
	para->SetParam(m_pBias);
}
