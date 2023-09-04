#include "PINNs_BurgersEquation.h"
#include "CTensor.h"
#include "CRandom.h"
using namespace std;

StruPINNsBurgEqua::StruPINNsBurgEqua() {
	m_mpDataTrain = new map<string, MatrData*>;
	m_pReader = new CFilesReader<StruDataMatFile>;
	emDatasetType = EmDatasetType::PINNS_BURGERSEQUA;
}

StruPINNsBurgEqua::~StruPINNsBurgEqua() {
	for (auto it : *m_mpDataTrain) {
		FreeObj(it.second);
	}
	m_mpDataTrain->clear();
	FreeObj(m_mpDataTrain);
	FreeObj(m_pReader);
}

void StruPINNsBurgEqua::SetFileParameter(map<string, void*>* mpParam) {
	dataMatFile = *(string*)((*mpParam)["dataMatFile"]);
}

void StruPINNsBurgEqua::LoadDataset() {
	int n = 3;
	string* keys = new string[3];
	keys[0] = "x";
	keys[1] = "t";
	keys[2] = "usol";
	m_pReader->SetFileDir(dataMatFile);
	m_pReader->ReadMatrixDataByName(keys, n);
}

std::map<std::string, MatrData*>* StruPINNsBurgEqua::GetTrainDataset() {
	int Nu = 20, Nf = 50, i;
	MatrData* tXu, * tU, * Xf, * Xu, * U;
	MatrData* x, * t, * usol;
	MatrData* tempIC, * tempLB, * tempUB, * temp1, * temp2;
	x = m_pReader->GetMapData("x");
	t = m_pReader->GetMapData("t");
	usol = m_pReader->GetMapData("usol");
	int sizeIC = x->rows();
	int sizeBC = t->rows();
	int sizeXu = sizeIC + 2 * sizeBC;
	tXu = new MatrData(sizeXu, 2);
	tU = new MatrData(sizeXu, 1);
	Xu = new MatrData(Nu, 2);
	U = new MatrData(Nu, 1);
	//the initial condition
	tempIC = new MatrData(sizeIC, 2);
	tempIC->col(0) << *x;
	tempIC->col(1).fill((*t)(0, 0));
	//the boundary condition
	//the low boundary
	tempLB = new MatrData(sizeBC, 2);
	tempLB->col(0).fill((*x)(0, 0));
	tempLB->col(1) << *t;
	//the upper boundary
	tempUB = new MatrData(sizeBC, 2);
	tempUB->col(0).fill(x->col(0).tail(1)(0));
	tempUB->col(1) << *t;
	*tXu << *tempIC, * tempLB, * tempUB;
	*tU << usol->col(0), usol->row(0).transpose(), usol->row(static_cast<uint64_t>(sizeIC) - 1).transpose();
	//the random sample points
	Xf = new MatrData(sizeXu + Nf, 2);
	Xf->block(0, 0, sizeXu, 2) << *tXu;
	temp1 = MatrixOp::UniformDistMat(Nf, 1, -1, 1);
	temp2 = MatrixOp::UniformDistMat(Nf, 1, 0, 1);
	Xf->block(sizeXu, 0, Nf, 1) << *temp1;
	Xf->block(sizeXu, 1, Nf, 1) << *temp2;
	//sample the IC&BC
	int* index = new int[sizeXu];
	for (i = 0; i < sizeXu; ++i) {
		index[i] = i;
	}
	int* idx, n;
	idx = CRandom<int>::RandomChoice(index, sizeXu, Nu);
	for (i = 0; i < Nu; ++i) {
		n = idx[i];
		Xu->row(i) = tXu->row(n);
		U->row(i) = tU->row(n);
	}
	//free memory
	delete[] index;
	delete[] idx;
	FreeObj(tempIC);
	FreeObj(tempLB);
	FreeObj(tempUB);
	FreeObj(temp1);
	FreeObj(temp2);
	FreeObj(tXu);
	FreeObj(tU);
	//return
	m_mpDataTrain->insert(make_pair("Xu", Xu));
	m_mpDataTrain->insert(make_pair("Xf", Xf));
	m_mpDataTrain->insert(make_pair("U", U));
	return m_mpDataTrain;
}

TyTest StruPINNsBurgEqua::GetTestDataset()
{
	MatrData* Xu;
	MatrData* x, * t, * usol;
	MatrData* tempIC, * tempLB, * tempUB;
	int row = 256, col = 100, i, j, p, q, k;
	x = m_pReader->GetMapData("x");
	t = m_pReader->GetMapData("t");
	Xu = new MatrData(row * col, 2);
	for (i = 0; i < row; ++i)
	{
		k = i * col;
		Xu->block(k, 0, col, 1).fill((*x)(i, 0));
		Xu->block(k, 1, col, 1) << *t;
	}
	return Xu;
}
