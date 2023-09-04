#include "CSequential.h"
#include "CLinearLayer.h"
#include "CModuleBase.h"
#include "CParameter.h"
#include "CFunction.h"
#include "struMat.h"
#include "CTensor.h"

struSeqVec::struSeqVec() {

}

struSeqVec::~struSeqVec() {
	auto it = vecSeq.begin();
	auto ite = vecSeq.end();
	for (; it != ite; ++it) {
		FreeObj(*it);
	}
	vecSeq.clear();
}

void struSeqVec::SetParameter(PtrPara para) {
	para->SetMods(&vecSeq);
	auto it = vecSeq.begin();
	auto ite = vecSeq.end();
	for (; it != ite; ++it) {
		CLayerBase* lay = dynamic_cast<CLayerBase*>(*it);
		if (!lay) {
			continue;
		}
		else {
			lay->SetParam(para);
		}
	}
}

void struSeqVec::AddModule(CLinearLayerBase* mod) {
	vecSeq.push_back(mod);
}

void struSeqVec::AddModule(CActivationsFunction* mod) {
	vecSeq.push_back(mod);
}

PtrTen struSeqVec::SeqForward(PtrTen input) {
	auto it = vecSeq.begin();
	auto ite = vecSeq.end();
	PtrTen temp = input;
	for (; it != ite; ++it) {
		temp = (*it)->Forward(temp);
	}
	return temp;
}

PtrMat struSeqVec::ForwardNonGrad(PtrMat input) {
	auto it = vecSeq.begin();
	auto ite = vecSeq.end();
	PtrMat temp = new struMat(input->GetDataPtr());
	for (; it != ite; ++it) {
		temp = (*it)->ForwardNonGrad(temp);
	}
	return temp;
}

struSeqMap::struSeqMap() {

}

struSeqMap::~struSeqMap() {
	auto its = mapSeq.begin();
	auto ite = mapSeq.end();
	for (auto it = its; it != ite; ++it) {
		FreeObj(it->second);
	}
	mapSeq.clear();
}

void struSeqMap::SetParameter(PtrPara para) {

	auto its = mapSeq.begin();
	auto ite = mapSeq.end();
	for (auto it = its; it != ite; ++it) {
		para->SetMods(it->second);
		CLayerBase* lay = dynamic_cast<CLayerBase*>(it->second);
		if (!lay) {
			continue;
		}
		else {
			lay->SetParam(para);
		}
	}
}

PtrTen struSeqMap::SeqForward(PtrTen input) {
	auto its = mapSeq.begin();
	auto ite = mapSeq.end();
	PtrTen temp = input;
	for (auto it = its; it != ite; ++it) {
		temp = it->second->Forward(temp);
	}
	return temp;
}
