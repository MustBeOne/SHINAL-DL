#include "C_DNN.h"
#include "CSequential.h"
#include "CTensor.h"
#include "COptimizerBase.h"
#include "CDataProc.h"
#include "CParameter.h"
#include "CDatasetLoader.h"
using namespace std;
//-------------------------Deep neural network--------------------


struDNNParam::struDNNParam() {
	m_pDataset = new CDatasetLoader<StruRawDataMnist>;
}

struDNNParam::~struDNNParam() {
	FreeObj(m_pDataset);
}

void struDNNParam::InitDataset()
{
	map<string, void*>* pInit = new map<string, void*>;
	pInit->insert(make_pair("lengTrain", &trainLeng));
	pInit->insert(make_pair("lengTest", &testLeng));
	pInit->insert(make_pair("trainXfile",&trainX));
	pInit->insert(make_pair("trainYfile",&trainY));
	pInit->insert(make_pair("testXfile",&testX));
	pInit->insert(make_pair("testYfile",&testY));		
	m_pDataset->GetDataset()->SetFileParameter(pInit);
	m_pDataset->ReadDatasetFile();
	pInit->clear();
	FreeObj(pInit);
}

C_DNN::C_DNN() {
	prednum = 0;
	m_pSeq = nullptr;
	m_pParam = nullptr;
	m_pLossFn = nullptr;
	m_pOptimizer = nullptr;
	m_pParameter = nullptr;
}

C_DNN::C_DNN(PtrSeq seq, struDNNParam* param) {
	prednum = 0;
	m_pSeq = seq;
	m_pParam = param;
	m_pParameter = new CParameter;
	m_pSeq->SetParameter(m_pParameter);
	m_pLossFn = nullptr;
	m_pOptimizer = nullptr;
}

C_DNN::~C_DNN() {
	FreeObj(m_pSeq);
	FreeObj(m_pParam);
	FreeObj(m_pLossFn);
	FreeObj(m_pOptimizer);
	FreeObj(m_pParameter);
}

//InterData* C_DNN::ForwardNet(TrainData* input)
//{
//	InterData* output = new InterData;
//	PtrTen in;
//	auto its = input->begin();
//	auto ite = input->end();
//	for (auto it = its; it != ite; ++it)
//	{
//		in = new CTensor(**it);
//		output->push_back(m_pSeq->SeqForward(in));
//	}
//	return output;
//}

PtrTen C_DNN::ForwardNet(PtrTen input) {
	PtrTen output;
	output = m_pSeq->SeqForward(input);
	return output;
}

vector<PtrTen>* C_DNN::ForwardNet(vector<PtrTen>* btcInput) {
	vector<PtrTen>* btcOutput = new vector<PtrTen>;
	PtrTen output;
	auto it = btcInput->begin();
	auto ite = btcInput->end();
	for (; it != ite; ++it) {
		output = m_pSeq->SeqForward(*it);
		btcOutput->push_back(output);
	}
	output = nullptr;
	return btcOutput;
}

//InterData* C_DNN::Loss(InterData* output, TrainData* target)
//{
//	InterData* loss = new InterData;
//	PtrTen tar;
//	auto its = output->begin();
//	auto ite = output->end();
//	auto it1 = target->begin();
//	for (auto it = its; it != ite; ++it, ++it1)
//	{
//		tar = new CTensor(**it1);
//		loss->push_back(m_pLossFn->CmpLoss(*it, tar));
//	}
//	return loss;
//}

PtrTen C_DNN::Loss(PtrTen output, PtrTen target) {
	PtrTen loss;
	loss = m_pLossFn->CmpLoss(output, target);
	return loss;
}


vector<PtrTen>* C_DNN::Loss(vector<PtrTen>* btcOutput, vector<PtrTen>* btcTarget) {
	vector<PtrTen>* btcLoss = new vector<PtrTen>;
	PtrTen loss;
	auto ito = btcOutput->begin();
	auto itoe = btcOutput->end();
	auto itt = btcTarget->begin();
	for (; ito != itoe; ++ito, ++itt) {
		loss = m_pLossFn->CmpLoss(*ito, *itt);
		btcLoss->push_back(loss);
	}
	loss = nullptr;
	return btcLoss;
}

void C_DNN::Backward(vector<CTensor*>* btcLoss) {
	auto it = btcLoss->begin();
	auto ite = btcLoss->end();
	for (; it != ite; ++it) {
		(*it)->Backward();
		(*it)->DeleteInleafNode();
	}
}

void C_DNN::Evaluate(MatrData*** test) {
	MatrData** input;
	MatrData** target;
	input = test[0];
	target = test[1];
	int leng = m_pParam->testLeng;
	int i;
	this->prednum = 0;
	bool isRight;
	PtrMat in, out;
	for (i = 0; i < leng; ++i) {
		in = new tyTensor(input[i]);
		out = m_pSeq->ForwardNonGrad(in);
		isRight = m_pParam->eval(out->GetDataPtr(), target[i]);
		if (isRight) {
			++this->prednum;
		}
		FreeObj(in);
		FreeObj(out);
	}
	cout << '\n' << this->prednum << endl;;
}

void C_DNN::TrainBatch(struBatchedData* dataset) {
	PtrTen input, target;
	MatrData*** inputSet, *** targetSet;
	vector<PtrTen>* btcinput, * btctarget, * btcoutput, * btcloss;
	btcinput = new vector<PtrTen>;
	btcoutput = nullptr;
	btctarget = new vector<PtrTen>;
	btcloss = nullptr;
	int j, k, btcSize, btcN;
	int n = 0;
	//
	btcN = dataset->btcN;
	btcSize = dataset->btcSize;
	inputSet = (*dataset->m_vcData)[0];
	targetSet = (*dataset->m_vcData)[1];
	//
	for (j = 0; j < btcN - 1; ++j) {
		m_pOptimizer->ZeroGrad();
		for (k = 0; k < btcSize; ++k) {
			//input and target have no grad and won't be backward
			input = new CTensor(inputSet[j][k]);
			target = new CTensor(targetSet[j][k]);
			btcinput->push_back(input);
			btctarget->push_back(target);
		}
		btcoutput = ForwardNet(btcinput);
		btcloss = Loss(btcoutput, btctarget);
		Backward(btcloss);
		m_pParameter->MeanLrParamGrad(btcSize);
		ClearObjVec<PtrTen>(btcinput);
		ClearObjVec<PtrTen>(btctarget);
		btcoutput->clear(); FreeObj(btcoutput);
		btcloss->clear(); FreeObj(btcloss);
		++n;
		if (n % 100 == 0) cout << n * btcSize << " ";
		m_pOptimizer->Step();
	}
	if (dataset->remainder)
	{
		m_pOptimizer->ZeroGrad();
		for (k = 0; k < dataset->remainder; ++k) {
			//input and target have no grad and won't be backward
			input = new CTensor(inputSet[j][k]);
			target = new CTensor(targetSet[j][k]);
			btcinput->push_back(input);
			btctarget->push_back(target);
		}
		btcoutput = ForwardNet(btcinput);
		btcloss = Loss(btcoutput, btctarget);
		Backward(btcloss);
		m_pParameter->MeanLrParamGrad(btcSize);
		ClearObjVec<PtrTen>(btcinput);
		ClearObjVec<PtrTen>(btctarget);
		btcoutput->clear(); FreeObj(btcoutput);
		btcloss->clear(); FreeObj(btcloss);
		++n;
		if (n % 100 == 0) cout << n * btcSize << " ";
		m_pOptimizer->Step();
	}
}

void C_DNN::LoadComponents(PtrLossFunc lf, PtrOptm op) {
	m_pLossFn = lf;
	m_pOptimizer = op;
}

void C_DNN::Train() {
	MatrData*** rawData;
	struBatchedData* dataset;
	PtrRawData test = nullptr;
	int epoches = m_pParam->epoches, i;
	for (i = 0; i < epoches; ++i) {
		cout << i << endl;
		rawData = m_pParam->m_pDataset->GetTrainData();
		dataset = CDataLoader::BatchPackData(rawData, 2, m_pParam->trainLeng, m_pParam->btcSize);
		//
		TrainBatch(dataset);
		if (m_pParam->ifEval) {
			Evaluate(m_pParam->m_pDataset->GetTestData());
		}
		FreeObj(dataset);
	}
}

ItParam C_DNN::Parameters() {
	return (*this->m_pParameter)();
}

