#include "C_PINNs.h"
#include "CSequential.h"
#include "CTensor.h"
#include "COptimizerBase.h"
#include "CDataProc.h"
#include "CParameter.h"
#include "CFileWriter.h"
#include "CDatasetLoader.h"
using namespace std;

//-------------------------Deep neural network--------------------


StruPINNsParam::StruPINNsParam() {
	m_pDataset = new CDatasetLoader<StruPINNsBurgEqua>;
}

StruPINNsParam::~StruPINNsParam() {
	FreeObj(m_pDataset);
}

void StruPINNsParam::InitDataset() {
	map<string, void*>* pInit = new map<string, void*>;
	pInit->insert(make_pair("dataMatFile", &dataMatFile));
	m_pDataset->GetDataset()->SetFileParameter(pInit);
	m_pDataset->ReadDatasetFile();
	pInit->clear();
	FreeObj(pInit);
}

C_PINNs::C_PINNs() {
	nIter = 0;
	m_pSeq = nullptr;
	m_pParam = nullptr;
	m_pLossFn = nullptr;
	m_pOptimizer = nullptr;
	m_pParameter = nullptr;
	XPred = nullptr, Xu = nullptr, Tu = nullptr, Xf = nullptr, Tf = nullptr, U = nullptr;
}

C_PINNs::C_PINNs(PtrSeq seq, StruPINNsParam* param) {
	nIter = 0;
	m_pSeq = seq;
	m_pParam = param;
	m_pParameter = new CParameter;
	m_pSeq->SetParameter(m_pParameter);
	m_pLossFn = nullptr;
	m_pOptimizer = nullptr;
	XPred = nullptr, Xu = nullptr, Tu = nullptr, Xf = nullptr, Tf = nullptr, U = nullptr;
}

C_PINNs::~C_PINNs() {
	FreeObj(m_pSeq);
	FreeObj(m_pParam);
	FreeObj(m_pLossFn);
	FreeObj(m_pOptimizer);
	FreeObj(m_pParameter);
	FreeObj(XPred); FreeObj(Xu); FreeObj(Tu); FreeObj(Xf); FreeObj(Tf); FreeObj(U);
}


PtrTen C_PINNs::NetU() {
	PtrTen input = CTensor::Concatenation(Xu, Tu, 0);
	//PtrTen input = CTensor::Concatenation(aa, aaa, 0);
	//input->TransposeIP();
	PtrTen output;
	output = m_pSeq->SeqForward(input);
	return output;
}


PtrTen C_PINNs::NetF() {
	PtrTen input = CTensor::Concatenation(Xf, Tf, 0);
	//PtrTen input = CTensor::Concatenation(aa, aaa, 0);
	//input->TransposeIP();
	PtrTen u, output;
	u = m_pSeq->SeqForward(input);
	PtrTen Ut, Ux, Uxx;
	Ut = u->AutoGradBwMode(Tf, true);
	Ux = u->AutoGradBwMode(Xf, true);
	Uxx = Ux->AutoGradBwMode(Xf, true);
	/*Ut = u->AutoGradBwMode(input, true);
	Ux = Ut->AutoGradBwMode(input, true);
	Ut = u->AutoGradBwMode(aa, true);
	Ux = u->AutoGradBwMode(aaa, true);
	Uxx = Ux->AutoGradBwMode(aaa, true);*/
	output = *Ut + *(*u * *Ux);
	output = *output - *(m_pParam->Nu * *Uxx);
	return output;
}

PtrTen C_PINNs::Closure() {
	PtrTen predU, predF;
	m_pOptimizer->ZeroGrad();
	predF = NetF();
	predU = NetU();
	PtrTen lossU, lossF, loss;
	lossU = (*U - *predU)->Pow(2);
	//lossU = (*U - 2)->Pow(2);
	lossF = predF->Pow(2);
	lossU = CTensor::Mean(lossU);
	lossF = CTensor::Mean(lossF);
	loss = *lossU + *lossF;
	loss->Backward();
	/*for (auto it : *(m_pParameter->GetParamsPtr()))
	{
		cout << *it->GetGrad()<<endl;
	}*/
	if (nIter % 10 == 0) {
		cout << "Iter " << nIter << " Loss: " << loss->GetData()->GetData() << " Loss_u : " << lossU->GetData()->GetData() << " Loss_f : " << lossF->GetData()->GetData() << endl;
	}
	loss->DeleteInleafNode(true);
	++nIter;
	return loss;
}

void C_PINNs::Predict()
{
	PtrMat output;
	output = m_pSeq->ForwardNonGrad(XPred);
	/*
	* Save predict results to *.txt
	*/
	string filename = "BurgersPINNsPredictResult.mat";
	string* matnames = new string[1];
	matnames[0] = "result1";
	MatrData** mats = new MatrData * [1];
	mats[0] =new MatrData(output->GetDataPtr()->reshaped<RowMajor>(256, 100));
	CFileWriter<>::WriteMatrToFile_Mat(filename, matnames, mats, 1);
	FreeObj(output);
}

void C_PINNs::Init() {
	decltype(m_pParam->m_pDataset->GetTrainData()) rawData;
	rawData = m_pParam->m_pDataset->GetTrainData();
	Xu = new CTensor(rawData->at("Xu")->col(0), true);
	Tu = new CTensor(rawData->at("Xu")->col(1), true);
	Xf = new CTensor(rawData->at("Xf")->col(0), true);
	Tf = new CTensor(rawData->at("Xf")->col(1), true);
	U = new CTensor(rawData->at("U")->col(0), true);
	Xu->TransposeIP();
	Tu->TransposeIP();
	Xf->TransposeIP();
	Tf->TransposeIP();
	U->TransposeIP();
	auto temp = m_pParam->m_pDataset->GetTestData();
	XPred = new struMat(temp);
	XPred->TransposeIP();
	FreeObj(temp);


	/*aa = new CTensor(-1.e-2, true);
	aaa = new CTensor(2.e-2, true);*/
	/*for (auto it : *(m_pParameter->GetParamsPtr()))
	{
		cout << *it << endl;
	}*/
	/*
	Xu->SetLeafSta(false);
	Tu->SetLeafSta(false);
	Xf->SetLeafSta(false);
	Tf->SetLeafSta(false);
	U->SetLeafSta(false);*/
}

void C_PINNs::LoadComponents(PtrLossFunc lf, PtrOptm op) {
	m_pLossFn = lf;
	m_pOptimizer = op;
}

void C_PINNs::Train() {
	m_pOptimizer->Step(/*&C_PINNs::Closure, */this);
}

ItParam C_PINNs::Parameters() {
	return (*this->m_pParameter)();
}

