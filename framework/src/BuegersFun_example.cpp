#include <iostream>
#include "CTensor.h"
#include "C_DNN.h"
#include "struMat.h"
#include "C_PINNs.h"
#include "CRandom.h"
#include "CDataProc.h"
#include "Mnist.h"
#include "CDatasetLoader.h"
#include "NNHead.h"
#include "CFileWriter.h"
#include "PINNs_BurgersEquation.h"

#include "CFileReader.h"
using namespace std;

int main() {
	string path = ".\\";
	CDatasetLoader<StruPINNsBurgEqua>* loader = new CDatasetLoader<StruPINNsBurgEqua>;
	StruPINNsParam* param = new StruPINNsParam;
	param->dataMatFile = path + "burgers_shock.mat";
	param->Nu = 0.01 / PI;
	param->m_pDataset = loader;
	param->InitDataset();
	//
	CLinearLayerBase* lay;
	CActivationsFunction* act;
	PtrSeq seq = new struSeqVec();
	//the 1 layer
	lay = new CLinear(2, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 2 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 3 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 4 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 5 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 6 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 7 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 8 layer
	lay = new CLinear(20, 20);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	//the 9 layer
	lay = new CLinear(20, 1);
	act = new CTanhAF();
	seq->AddModule(lay);
	seq->AddModule(act);
	C_PINNs* nn;
	nn = new C_PINNs(seq, param);
	//
	COptimizerBase* opti;
	opti = new COptimizerLBFGS(nn->Parameters(), 1., 50000, 50000, 1e-5, 2.22e-16, 50);
	nn->LoadComponents(nullptr, opti);
	//
	nn->Init();
	nn->Train();
	nn->Predict();

	return 0;
}




