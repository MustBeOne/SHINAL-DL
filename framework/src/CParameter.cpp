#include "CParameter.h"
#include "CTensor.h"

struParameter::struParameter(int row, int col) {
	data = new CTensor(row, col, true);
}

struParameter::~struParameter() {
	FreeObj(data);
}

void CParameter::SetParam(PtrTen para) {
	ITER(VecParam) it = find(vecParas.begin(), vecParas.end(), para);
	if (it == vecParas.end()) {
		vecParas.push_back(para);
	}
	else {
		return;
	}
}

void CParameter::SetMods(VecModule* modes) {
	this->vecMods = modes;
}

void CParameter::SetMods(CModuleBase* mod) {
	this->vecMods->push_back(mod);
}

void CParameter::MeanLrParamGrad(int base) {
	auto it = vecParas.begin();
	auto ite = vecParas.end();
	tyTensor* temp;
	for (; it != ite; ++it) {
		temp = (*it)->GetGrad();
		*temp = *temp / Scalar(base);
	}
	temp = nullptr;
}

#ifdef show
void CParameter::Show() {
	for (auto it = vecParas.begin(); it != vecParas.end(); ++it) {
		cout << (*it)->GetData()->GetDataPtr()->array().pow(2).sum() / (*it)->GetData()->GetDataPtr()->size();
	}
}
#endif

ItParam CParameter::operator()() {
	return make_pair(&vecParas, vecMods);
}
