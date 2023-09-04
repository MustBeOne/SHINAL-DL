#if 1
#include "CTensor.h"
#include "struMat.h"
#include "Backward.h"
#include "FuncBackward.h"
//#include "Forward.h"
using namespace std;

CTensor::CTensor() {
	//
	m_pData = new tyTensor(1, 1, 0);
	m_pGrad = new tyTensor(1, 1, 0);
	//
	m_pGradFn = new NoneBackward();
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(int row, int col, Scalar data, bool if_grad) {
	//
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(row, col, data);
	this->m_pGrad = new tyTensor(row, col, 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(Scalar data, bool if_grad) {
	//
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(1, 1, data);
	this->m_pGrad = new tyTensor(1, 1, 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(tyTensor* data, bool if_grad, bool ifCopy) {
	this->if_grad = if_grad;
	if (ifCopy)
	{
		this->m_pData = new tyTensor(*data);
	}
	else
	{
		this->m_pData = data;
	}
	m_pGrad = new tyTensor(data->GetShape(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(tyTensor& data, bool if_grad) {
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(data);
	m_pGrad = new tyTensor(data.GetShape(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(tyTensor&& data, bool if_grad) {
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(data);
	m_pGrad = new tyTensor(data.GetShape(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(MatrData* data, bool if_grad, bool ifCopy) {
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(data, ifCopy);
	m_pGrad = new tyTensor(data->rows(), data->cols(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(MatrData& data, bool if_grad) {
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(data);
	m_pGrad = new tyTensor(data.rows(), data.cols(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(MatrData&& data, bool if_grad) {
	this->if_grad = if_grad;
	this->m_pData = new tyTensor(data);
	m_pGrad = new tyTensor(data.rows(), data.cols(), 0);
	//
	if (this->if_grad) {
		m_pGradFn = new AccumulateGrad();
	}
	else {
		m_pGradFn = new NoneBackward();
	}
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd();
	//
	m_stOrigin = new setTsr();
	m_stOrigin->insert(this);
	/*
	* the default member
	*/
	if_grad = false;
	is_leaf = true;
	m_vcRetainGraph = nullptr;
	m_stFwPath = nullptr;
	if_gradGet = false;
}

CTensor::CTensor(CTensor& tensor) {
	FreeObj(m_pData);
	m_pData = new tyTensor(*tensor.GetData());
	FreeObj(m_pGrad);
	m_pGrad = new tyTensor(*tensor.GetGrad());
	//
	if_grad = tensor.IfGrad();
	is_leaf = false;
	//
	m_pGradFn = new CopyBackward();
	m_pGradFn->SaveNextFuns(&tensor, nullptr);
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd;
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* cpy = new CpyForward(&tensor, this);
	tensor.GetForward()->push_back(cpy);
#endif
	//
	m_stOrigin = new setTsr(*tensor.GetOrigin());
	m_vcRetainGraph = nullptr;
}
CTensor::CTensor(CTensor&& tensor) {
	FreeObj(m_pData);
	m_pData = new tyTensor(*tensor.GetData());
	FreeObj(m_pGrad);
	m_pGrad = new tyTensor(*tensor.GetGrad());
	//
	if_grad = tensor.IfGrad();
	is_leaf = false;
	//
	m_pGradFn = new CopyBackward();
	m_pGradFn->SaveNextFuns(&tensor, nullptr);
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd;
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* cpy = new CpyForward(&tensor, this);
	tensor.GetForward()->push_back(cpy);
#endif
	//
	m_stOrigin = new setTsr(*tensor.GetOrigin());
	m_vcRetainGraph = nullptr;
}


CTensor::CTensor(CTensor* tensor)
{
	FreeObj(m_pData);
	m_pData = new tyTensor(*(tensor->GetData()));
	FreeObj(m_pGrad);
	m_pGrad = new tyTensor(*(tensor->GetGrad()));
	//
	if_grad = tensor->IfGrad();
	is_leaf = false;
	//
	m_pGradFn = new CopyBackward();
	m_pGradFn->SaveNextFuns(tensor, nullptr);
	m_pGradFn->SaveNode(this);
	//
	m_vcForward = new VecFowd;
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* cpy = new CpyForward(tensor, this);
	tensor->GetForward()->push_back(cpy);
#endif
	//
	m_stOrigin = new setTsr(*(tensor->GetOrigin()));
	m_vcRetainGraph = nullptr;
}

CTensor::~CTensor() {
	//
	FreeObj(m_pData);
	FreeObj(m_pGrad);
	//
	FreeObj(m_pGradFn);
#ifdef FORWARD_AUTOGRAD_MODLE
	ClearForwawrdGraph();
#endif
	FreeObj(m_vcForward);
	//
	m_stOrigin->clear();
	FreeObj(m_stOrigin);
	if (if_gradGet) {
		ClearObjVec(m_vcRetainGraph);
		FreeObj(m_vcRetainGraph);
	}
}

//--------------------------------------
tyTensor* CTensor::GetData() {
	return this->m_pData;
}

void CTensor::SetData(tyTensor* data) {
	FreeObj(this->m_pData);
	this->m_pData = new tyTensor(*data);
}

void CTensor::SetData(MatrData* data) {
	FreeObj(this->m_pData);
	this->m_pData = new tyTensor(data);
}

tyTensor* CTensor::GetGrad() {
	if (!this->m_pGrad) {
		cout << "Grad None." << endl;
		return nullptr;
	}
	return this->m_pGrad;
}

void CTensor::ShowGrad() {
	if (!this->m_pGrad) {
		cout << "Grad None." << endl;
	}
	else {
		cout << "grad: " << endl;
		cout << *m_pGrad << endl;
	}
}

void CTensor::SetGrad(tyTensor* grad) {
	FreeObj(this->m_pGrad);
	this->m_pGrad = new tyTensor(*grad);
}

void CTensor::AddGrad(tyTensor* grad) {
	if (!m_pGrad) return;
	if (is_leaf)
	{
		*this->m_pGrad += *grad;
	}
}

void CTensor::AddGrad(Scalar grad)
{
	if (!m_pGrad) return;
	if (is_leaf)
	{
		this->m_pGrad->GetDataPtr()->array() += grad;
	}
}

void CTensor::ClearGrad() {
	FreeObj(this->m_pGrad);
}

void CTensor::ClearData() {
	FreeObj(this->m_pData);
}

#ifdef FORWARD_AUTOGRAD_MODLE
VecFowd* CTensor::GetForward() {
	return m_vcForward;
}

void CTensor::SetForward(ForwardBase* fw) {
	this->m_vcForward->push_back(fw);
}
void CTensor::ClearForwawrdGraph() {
	ClearObjVec(m_vcForward);
}
#endif

void CTensor::ZeroGrad() {
	this->m_pGrad->GetDataPtr()->setConstant(0.);
}

void CTensor::DeleteInleafNode(bool ifRetainSelf) {
	this->m_pGradFn->DeleteInleafNode(ifRetainSelf);
}

void CTensor::SetRetainedGraph(VecTsr* graph) {
	m_vcRetainGraph = graph;
}

void CTensor::CleanGraph() {
	m_pGradFn->ReleaseFwGraph();
}

BackwardBase* CTensor::GetGradFn() {
	return this->m_pGradFn;
}

void CTensor::SetGradFn(BackwardBase* bw) {
	FreeObj(this->m_pGradFn);
	this->m_pGradFn = bw;
}

bool CTensor::IfGrad(void) {
	return this->if_grad;
}

void CTensor::SetGradSta(bool sta) {
	this->if_grad = sta;
	if (!sta) {
		FreeObj(m_pGradFn);
		m_pGradFn = new NoneBackward();
	}
}

bool CTensor::IsLeaf(void) {
	return this->is_leaf;
}

void CTensor::SetLeafSta(bool sta) {
	this->is_leaf = sta;
	if (!sta) {
		FreeObj(m_pGrad);
	}
	else
	{
		if (!m_pGrad)
		{
			m_pGrad = new tyTensor(m_pData->GetShape(), 0.);
		}
	}
}

bool CTensor::IsGradGet(void) {
	return this->if_gradGet;
}

void CTensor::SetGradGetSta(bool ifgradGet) {
	if_gradGet = ifgradGet;
}

setTsr* CTensor::GetOrigin() {
	return this->m_stOrigin;
}

void CTensor::SetOrigin(setTsr* ori) {
	setTsr::iterator it = ori->begin();
	setTsr::iterator ite = ori->end();
	for (; it != ite; ++it) {
		m_stOrigin->insert(*it);
	}
}

tyTensor* CTensor::GetSameSizeData(Scalar tdata) {
	tyTensor* temp = new tyTensor(m_pData->GetShape(), tdata);
	return temp;
}

//-------------------------AUTOGRAD UNITS------------------------------
/*
* These are the private APIs performed in the Autograd functions by Backward Mode
*/


/*
* Followings are public APIs,including
* the solutions to performing autograd based on FORWARD or BACKWARD,
* the function to set the control properties to autograd
*/
void CTensor::RequireGrad(bool set) {
	this->if_grad = set;
}

void CTensor::RetainGrad() {
	this->is_leaf = true; 
	if (!m_pGrad)
	{
		m_pGrad = new tyTensor(m_pData->GetShape(), 0.);
	}
}

//these are Assistant-Functions
bool CTensor::IfSearchDir(CTensor* origin) {
	setTsr::iterator ite = m_stOrigin->end();
	if (m_stOrigin->find(origin) == ite) {
		return false;
	}
	return true;
}


void CTensor::Backward(bool retainGrad, bool retainFwGraph) {
	m_pGradFn->Backward(1.);
	if (!retainGrad) m_pGradFn->ReleaseGrad();
	if (!retainFwGraph) m_pGradFn->ReleaseFwGraph();
}

CTensor* CTensor::AutoGradBwMode(CTensor* input, bool ifRetainGraph /*= false*/, bool ifCreateGraph /*= false*/) {
	CTensor* result;
	VecTsr* vecResult = new VecTsr;
	VecTsr* vecGraph = new VecTsr;
	m_pGradFn->BackwardInAuto(1., input, vecGraph, vecResult);
	if (!vecResult->size()) {
		cout << "The AutoGrad Cannot Perform!" << endl;
		return nullptr;
	}
	if (vecResult->size() == 1) {
		result = *vecResult->begin();
	}
	else {
		result = Accumulate(vecResult);
		for (CTensor* t : *vecResult) {
			vecGraph->push_back(t);
		}
	}
	vecResult->clear();
	FreeObj(vecResult);
	if (ifRetainGraph) {
		result->SetRetainedGraph(vecGraph);
		result->SetGradGetSta(true);
	}
	else {
		ClearObjVec(vecGraph);
		FreeObj(vecGraph);
	}
	return result;
}

/*
* These are the private APIs performed in the Autograd functions by Forward Mode
*/

#ifdef FORWARD_AUTOGRAD_MODLE
void CTensor::Forward(CTensor* prenode, VecTsr* result, VecTsr* graph) {
	if (!m_stFwPath) {
		result->push_back(prenode);
		return;
	}
	setFwd::iterator it = m_stFwPath->begin();
	setFwd::iterator ite = m_stFwPath->end();
	for (; it != ite; ++it) {
		(*it)->Forward(prenode, result, graph);
	}
}
bool CTensor::SearchPath(CTensor* end) {
	if (!m_vcForward->size()) {
		if (this == end) {
			return true;
		}
		else return false;
	}
	bool sta = false;
	auto it = m_vcForward->begin();
	auto ite = m_vcForward->end();
	CTensor* cur, * next;
	for (; it != ite; ++it) {
		cur = (*it)->GetCurNode();
		next = (*it)->GetNxtNode();
		if (cur == end) {
			return true;
		}
		if (next->SearchPath(end)) {
			if (!m_stFwPath) {
				m_stFwPath = new setFwd;
			}
			m_stFwPath->insert(*it);
			sta = true;
		}
	}
	return sta;
}

void CTensor::ClearPath() {
	if (!m_stFwPath) {
		return;
	}
	setFwd::iterator it = m_stFwPath->begin();
	setFwd::iterator ite = m_stFwPath->end();
	for (; it != ite; ++it) {
		(*it)->GetNxtNode()->ClearPath();
	}
	m_stFwPath->clear();
	FreeObj(m_stFwPath);
}


CTensor* CTensor::AutoGradFwMode(CTensor* input, bool ifRetainGraph, bool ifCreateGraph) {
	CTensor* result;
	VecTsr* vecResult = new VecTsr;
	VecTsr* vecGraph = new VecTsr;
	input->SearchPath(this);
	input->Forward(nullptr, vecResult, vecGraph);
	result = input->Accumulate(vecResult);
	input->ClearPath();
	vecResult->clear();
	FreeObj(vecResult);
	if (ifRetainGraph) {
		result->SetRetainedGraph(vecGraph);
	}
	else {
		ClearObjVec(vecGraph);
	}
	return result;
}
#endif

CTensor* CTensor::Transpose() {
	CTensor* temp = new CTensor(*this);
	temp->TransposeIP();
	return temp;
}

void CTensor::TransposeIP() {
	this->m_pData->TransposeIP();
	this->m_pGrad->TransposeIP();
}

void CTensor::NegData() {
	*m_pData->GetDataPtr() = -*m_pData->GetDataPtr();
}

CTensor* CTensor::Sum() {
	CTensor* result = new CTensor(this->m_pData->Sum(), if_grad);
	BackwardBase* bw;
	if (!if_grad) {
		bw = new NoneBackward();
	}
	else {
		bw = new SumBackward();
		//the raw grad
		tyTensor* raw = new tyTensor(m_pData->GetShape(), 1);
		bw->SaveNextFuns(this, raw);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	result->SetLeafSta(false);
	return result;
}

CTensor* CTensor::Pow(Scalar powCoe, Scalar sign, Scalar alpha) {
	CTensor* result = new CTensor(this->m_pData->GetDataPtr()->array().pow(powCoe) * sign, if_grad);
	result->AddScalarIP(alpha);
	result->SetLeafSta(false);
	result->SetGradSta(if_grad);
	// process the backward graph
	BackwardBase* bw;
	if (!if_grad) {
		bw = new NoneBackward();
	}
	else {
		bw = new PowBackward();
		//the raw grad
		MatrData* rawGrad = new MatrData(m_pData->GetDataPtr()->array().pow(powCoe - 1));
		*rawGrad = (rawGrad->array() * powCoe * sign).matrix();
		tyTensor* raw = new tyTensor(rawGrad);
		FreeObj(rawGrad);
		bw->SaveNextFuns(this, raw);
		bw->SetFactors(powCoe, sign);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* pForward = new PowForward(this, result);
	pForward->SetFactor(sign, powCoe);
	SetForward(pForward);
#endif
	// process the origin node
	result->SetOrigin(m_stOrigin);
	return result;
}

CTensor* CTensor::Dot(CTensor* rTensor, bool lT, bool rT) {
	bool ifGrad = if_grad | rTensor->IfGrad();
	tyTensor lpart, rpart;
	lT ? lpart = m_pData->Transpose() : lpart = m_pData->GetData();
	rT ? rpart = rTensor->GetData()->Transpose() : rpart = rTensor->GetData()->GetData();
	tyTensor* data = lpart.Dot(&rpart);
	CTensor* result = new CTensor(data, ifGrad, false);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		auto tbw = new MmBackward();
		//the raw grad
		tyTensor* raw;
		raw = !lT && !rT ? rTensor->GetData()->TransposePtr() : new tyTensor(rTensor->GetData());
		tbw->SaveNextFuns(this, raw);
		tyTensor* raw1;
		raw1 = !lT && !rT ? m_pData->TransposePtr() : new tyTensor(m_pData);
		tbw->SaveNextFuns(rTensor, raw1);
		tbw->SetTransSta(lT, rT);
		bw = tbw;

	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new MmForward(this, result);
	m_pForward->IsLeft();
	SetForward(m_pForward);
	m_pForward = new MmForward(rTensor, result);
	rTensor->SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(m_stOrigin);
	result->SetOrigin(rTensor->GetOrigin());
	return result;
}

CTensor* CTensor::Dot(CTensor* lTensor, CTensor* rTensor, bool lT, bool rT)
{
	return lTensor->Dot(rTensor, lT, rT);
}

CTensor* CTensor::Dot(Scalar lScaler, CTensor* rTensor, int dim /*= 1*/, bool rT /*= false*/)
{
	bool ifGrad = rTensor->IfGrad();
	tyTensor rpart;
	rT ? rpart = rTensor->GetData()->Transpose() : rpart = rTensor->GetData()->GetData();
	int rows = dim, cols = rpart.GetShape().first;
	tyTensor* data = tyTensor(rows, cols, lScaler).Dot(&rpart);
	CTensor* result = new CTensor(data, ifGrad, false);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		auto tbw = new MmBackward();
		//the raw grad
		tbw->SaveNextFuns(rTensor, nullptr);
		tbw->SetTransSta(false, rT);
		tbw->SetCoeff(rows, cols, lScaler, 0);
		bw = tbw;
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(rTensor->GetOrigin());
	return result;
}

CTensor* CTensor::Dot(CTensor* lTensor, Scalar rScaler, int dim /*= 1*/, bool lT /*= false*/)
{
	bool ifGrad = lTensor->IfGrad();
	tyTensor lpart, rpart;
	lT ? lpart = lTensor->GetData()->Transpose() : lpart = lTensor->GetData()->GetData();
	int cols = dim, rows = lpart.GetShape().second;
	rpart = tyTensor(rows, cols, rScaler);
	tyTensor* data = lpart.Dot(&rpart);
	CTensor* result = new CTensor(data, ifGrad, false);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		auto tbw = new MmBackward();
		//the raw grad
		tbw->SaveNextFuns(lTensor, nullptr);
		tbw->SetTransSta(lT, false);
		tbw->SetCoeff(rows, cols, rScaler, 1);
		bw = tbw;
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(lTensor->GetOrigin());
	return result;
}

CTensor* CTensor::AddMM(Scalar beta, CTensor* mat, Scalar alpha, CTensor* mat1, CTensor* mat2)
{
	bool ifGrad = mat->IfGrad() | mat1->IfGrad() | mat2->IfGrad();
	tyTensor* data1, * data2, * data3, * data;
	data1 = mat->GetData();
	data2 = mat1->GetData();
	data3 = mat2->GetData();
	if ((data1->GetShape().first != data2->GetShape().first && data1->GetShape().second != data3->GetShape().second)
		|| data2->GetShape().second != data3->GetShape().first)
	{
		cout << "Can't perform the 'AddMM' operation with the matrix'size mismatch!" << endl;
		return nullptr;
	}
	data = tyTensor::AddMM(beta, data1, alpha, data2, data3);
	CTensor* result = new CTensor(data, ifGrad, false);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new AddmmBackward();
		bw->SetFactors(beta, alpha);
		//the raw grad
		bw->SaveNextFuns(mat, nullptr);
		bw->SaveNextFuns(mat1, data3->TransposePtr());
		bw->SaveNextFuns(mat2, data2->TransposePtr());
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(mat->GetOrigin());
	result->SetOrigin(mat1->GetOrigin());
	result->SetOrigin(mat2->GetOrigin());
	return result;
}

CTensor* CTensor::Copy() {
	CTensor* cpy = new CTensor(*this);
	return cpy;
}

CTensor* CTensor::ExpEach(Scalar coeExp, Scalar sign) {
	tyTensor* exp = this->m_pData->ExpEach(coeExp, sign);
	CTensor* result = new CTensor(exp, if_grad);
	FreeObj(exp);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!if_grad) {
		bw = new NoneBackward();
	}
	else {
		bw = new ExpBackward();
		//the raw grad
		tyTensor* raw = m_pData->ExpEach(coeExp, coeExp * sign);
		bw->SaveNextFuns(this, raw);
		bw->SetFactors(coeExp, sign);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* pForward = new ExpForward(this, result);
	SetForward(pForward);
#endif
	// process the origin node
	result->SetOrigin(m_stOrigin);
	return result;
}


CTensor* CTensor::Reciprocal(Scalar scaler) {
	CTensor* result;
	result = this->Pow(-1, scaler);
	return result;
}

CTensor* CTensor::QuadraPoly(Scalar a /*= 0.*/, Scalar b /*= 1.*/, Scalar c /*= 0.*/)
{
	CTensor* result;
	if (a == 0. && b == 0.)
	{
		result = new CTensor(c, false);
		result->SetLeafSta(false);
		return result;
	}
	tyTensor* data = m_pData->Pow(2., a);
	data->AddScalarIP(c);
	tyTensor temp = b * *m_pData;
	*data += temp;
	result = new CTensor(data, if_grad, false);
	BackwardBase* bw;
	if (!if_grad) {
		bw = new NoneBackward();
	}
	else
	{
		bw = new QuadraPolyBackward(a, b);
		tyTensor* raw = new tyTensor((2 * a) * *m_pData);
		raw->AddScalarIP(b);
		bw->SaveNextFuns(this, raw);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(m_stOrigin);
	return result;
}

CTensor* CTensor::Concatenation(CTensor* left, CTensor* right, int dim /*= 0*/) {
	int lMtcCoe, rMtcCoe;
	bool ifGrad = left->IfGrad() | right->IfGrad();
	CTensor* result;
	tyTensor* res = tyTensor::Cat(left->GetData(), right->GetData(), dim);
	if (!res) {
		cout << "The Concatenation Operation is error! Please check the input parameters!" << endl;
		return nullptr;
	}
	result = new CTensor(res, ifGrad);
	result->SetLeafSta(false);
	FreeObj(res);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new CatBackward();
		//the raw grad
		bw->SaveNextFuns(left, nullptr);
		bw->SaveNextFuns(right, nullptr);
		int mtcSize = dim == 0 ?
			left->GetData()->GetDataPtr()->rows() :
			left->GetData()->GetDataPtr()->cols();
		bw->SetFactors(dim, mtcSize);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(left->GetOrigin());
	result->SetOrigin(right->GetOrigin());
	return result;
}

CTensor* CTensor::Mean(CTensor* pTen) {
	bool ifGrad = pTen->IfGrad();
	CTensor* result;
	result = new CTensor(pTen->GetData()->GetDataPtr()->mean(), ifGrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifGrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new MeanBackward();
		//the raw grad

		bw->SaveNextFuns(pTen, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(pTen->GetOrigin());
	return result;
}

CTensor* CTensor::FillWith0(int dim, int leng, int pos) {
	CTensor* result;
	int rows, cols, mtcx, mtcy, mtc1, i = 0, j = 0;
	MatrData* pdata = m_pData->GetDataPtr();
	rows = pdata->rows();
	cols = pdata->cols();
	if (dim == 0) {
		mtcx = leng - rows;
		mtcy = cols;
		pos ? mtc1 = mtcx : mtc1 = rows;
		rows = leng;
	}
	else {
		mtcy = leng - cols;
		mtcx = rows;
		pos ? mtc1 = mtcy : mtc1 = cols;
		cols = leng;
	}
	MatrData zero = MatrData::Zero(mtcx, mtcy);
	MatrData* data = new MatrData(rows, cols);
	BackwardBase* bw = new CatBackward();
	if (pos == 0) {
		data->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
		dim == 0 ? i = pdata->rows() : j = pdata->cols();
		data->block(i, j, mtcx, mtcy) << zero;
		result = new CTensor(data, if_grad);
		FreeObj(data);
		bw->SaveNextFuns(this, nullptr);
		bw->SaveNextFuns(nullptr, nullptr);
	}
	else {
		data->block(i, j, mtcx, mtcy) << zero;
		dim == 0 ? i = mtcx : j = mtcy;
		data->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
		result = new CTensor(data, if_grad);
		FreeObj(data);
		bw->SaveNextFuns(nullptr, nullptr);
		bw->SaveNextFuns(this, nullptr);
	}
	result->SetLeafSta(false);
	bw->SetFactors(dim, mtc1);
	bw->SaveNode(result);
	result->SetGradFn(bw);
	result->SetOrigin(m_stOrigin);
	return result;
}

void CTensor::ScaleIP(Scalar scaler)
{
	m_pData->ScaledIP(scaler);
	m_pGradFn->ProcVrtScal(scaler);
}

void CTensor::AddScalarIP(Scalar scaler)
{
	m_pData->AddScalarIP(scaler);
}

CTensor* CTensor::Accumulate(VecTsr* tsArr) {
	bool bGrad = false;
	CTensor* result = new CTensor();
	result->SetLeafSta(false);
	BackwardBase* bw = new AccBackward();
	ForwardBase* m_pForward;
	auto it = tsArr->begin();
	auto ite = tsArr->end();
	//the first result data
	tyTensor res = *(*it)->GetData();
	//save the first node in the backward of the result
	bw->SaveNextFuns(*it, nullptr);
	//
	result->SetOrigin((*it)->GetOrigin());
	++it;
	//process the rest of the result nodes
	for (; it != ite; ++it) {
		// process the backward graph
		res = res + *(*it)->GetData();
		bw->SaveNextFuns(*it, nullptr);
		//
		result->SetOrigin((*it)->GetOrigin());
		bGrad = bGrad | (*it)->IfGrad();
	}
	result->SetGradSta(bGrad);
	result->SetData(&res);
	bw->SaveNode(result);
	result->SetGradFn(bw);
	return result;
}

/// -------------------the ADD OPERATION PART-----------------------
///both are tensor object
CTensor* operator+(CTensor& lTensor, CTensor& rTensor) {
	bool ifgrad = lTensor.IfGrad() | rTensor.IfGrad();
	tyTensor data = *rTensor.GetData() + *lTensor.GetData();
	CTensor* result = new CTensor(data, ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new AddBackward();
		//the raw grad
		bw->SaveNextFuns(&lTensor, nullptr);
		bw->SaveNextFuns(&rTensor, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new AddForward(&lTensor, result);
	lTensor.SetForward(m_pForward);
	m_pForward = new AddForward(&rTensor, result);
	rTensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}
///the left is a scaler and implement the broadcast operation
CTensor* operator+(Scalar scaler, CTensor& tensor) {
	bool ifgrad = tensor.IfGrad();
	tyTensor data = *tensor.GetData() + scaler;
	CTensor* result = new CTensor(data);
	result->SetGradSta(ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new AddBackward();
		//the raw grad
		bw->SaveNextFuns(&tensor, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new AddForward(&tensor, result);
	tensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(tensor.GetOrigin());
	return result;
}

CTensor* operator+(CTensor& tensor, Scalar scaler) {
	return scaler + tensor;
}

/// -------------------the SUBTRACTION OPERATION PART-----------------------
/// 
/// 

CTensor* CTensor::operator-()
{
	bool ifgrad = this->if_grad;
	MatrData* data = new MatrData(-*m_pData->GetDataPtr());
	CTensor* result = new CTensor(data, ifgrad, false);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new NegBackward();
		//the raw grad
		bw->SaveNextFuns(this, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new SubForward(this, result);
	m_pForward->IsLeft();
	SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(GetOrigin());
	return result;
}
///both are tensor object
CTensor* operator-(CTensor& lTensor, CTensor& rTensor) {
	bool ifgrad = lTensor.IfGrad() | rTensor.IfGrad();
	tyTensor data = *lTensor.GetData() - *rTensor.GetData();
	CTensor* result = new CTensor(data);
	result->SetGradSta(lTensor.IfGrad() | rTensor.IfGrad());
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new SubBackward();
		//the raw grad
		bw->SaveNextFuns(&lTensor, nullptr);
		bw->SaveNextFuns(&rTensor, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new SubForward(&lTensor, result);
	m_pForward->IsLeft();
	lTensor.SetForward(m_pForward);
	m_pForward = new SubForward(&rTensor, result);
	rTensor.SetForward(m_pForward)
#endif;
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}
///the left is a scaler and implement the broadcast operation
CTensor* operator-(Scalar scaler, CTensor& tensor) {
	bool ifgrad = tensor.IfGrad();
	tyTensor data = scaler - *tensor.GetData();
	CTensor* result = new CTensor(data);
	result->SetGradSta(ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new SubBackward();
		//the raw grad
		bw->SaveNextFuns(nullptr, nullptr);
		bw->SaveNextFuns(&tensor, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new SubForward(&tensor, result);
	m_pForward->IsLeft();
	tensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(tensor.GetOrigin());
	return result;
}


CTensor* operator-(CTensor& tensor, Scalar scaler) {
	bool ifgrad = tensor.IfGrad();
	tyTensor data = *tensor.GetData() - scaler;
	CTensor* result = new CTensor(data);
	result->SetGradSta(ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new SubBackward();
		//the raw grad
		bw->SaveNextFuns(&tensor, nullptr);
		bw->SaveNextFuns(nullptr, nullptr);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new SubForward(&tensor, result);
	m_pForward->IsLeft();
	tensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(tensor.GetOrigin());
	return result;
}

/// <summary>
/// the mul-op gradfn for right side of the "*"
/// </summary>
/// <param name="grad">the last node grad</param>
/// <param name="data"></param>
/// <returns></returns>
CTensor* operator*(CTensor& lTensor, CTensor& rTensor) {
	bool ifgrad = lTensor.IfGrad() | rTensor.IfGrad();
	tyTensor data = *rTensor.GetData() * *lTensor.GetData();
	CTensor* result = new CTensor(data);
	result->SetGradSta(lTensor.IfGrad() | rTensor.IfGrad());
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new MulBackward();
		//the raw grad
		tyTensor* raw;
		raw = new tyTensor(*rTensor.GetData());
		bw->SaveNextFuns(&lTensor, raw);
		raw = new tyTensor(*lTensor.GetData());
		bw->SaveNextFuns(&rTensor, raw);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new MulForward(&lTensor, result);
	m_pForward->SetAnotherNode(&rTensor);
	lTensor.SetForward(m_pForward);
	m_pForward = new MulForward(&rTensor, result);
	m_pForward->SetAnotherNode(&lTensor);
	rTensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}

CTensor* operator*(CTensor& lTensor, Scalar rTensor) {
	bool ifgrad = lTensor.IfGrad();
	tyTensor data = *lTensor.GetData() * rTensor;
	CTensor* result = new CTensor(data);
	result->SetGradSta(ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new MulBackward();
		//the raw grad
		bw->SaveNextFuns(&lTensor, nullptr);
		bw->SetFactors(rTensor);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new MulForward(&lTensor, result);
	lTensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	return result;
}

CTensor* operator*(Scalar lTensor, CTensor& rTensor) {
	bool ifgrad = rTensor.IfGrad();
	tyTensor data = *rTensor.GetData() * lTensor;
	CTensor* result = new CTensor(data);
	result->SetGradSta(ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new MulBackward();
		//the raw grad
		bw->SaveNextFuns(&rTensor, nullptr);
		bw->SetFactors(lTensor);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new MulForward(&rTensor, result);
	rTensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}

CTensor* operator/(CTensor& lTensor, CTensor& rTensor) {
	bool ifgrad = lTensor.IfGrad() | rTensor.IfGrad();
	tyTensor data = *lTensor.GetData() / *rTensor.GetData();
	CTensor* result = new CTensor(data, ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new DivBackward();
		//the raw grad
		tyTensor* raw;
		raw = new tyTensor(1. / *rTensor.GetData()); 
		bw->SaveNextFuns(&lTensor, raw);
		tyTensor* raw1, * raw2;
		raw1 = lTensor.GetData();
		raw2 = rTensor.GetData()->Pow(-2, -1);
		raw = new tyTensor(*raw1 * *raw2);
		FreeObj(raw2);
		bw->SaveNextFuns(&rTensor, raw);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
#ifdef FORWARD_AUTOGRAD_MODLE
	ForwardBase* m_pForward = new DivForward(&lTensor, result);
	m_pForward->IsLeft();
	m_pForward->SetAnotherNode(&rTensor);
	lTensor.SetForward(m_pForward);
	m_pForward = new DivForward(&rTensor, result);
	m_pForward->SetAnotherNode(&lTensor);
	rTensor.SetForward(m_pForward);
#endif
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}

CTensor* operator/(CTensor& lTensor, Scalar rTensor)
{
	bool ifgrad = lTensor.IfGrad();
	tyTensor data = *lTensor.GetData() / rTensor;
	CTensor* result = new CTensor(data, ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new DivBackward();
		//the raw grad
		bw->SaveNextFuns(&lTensor, nullptr);
		bw->SetFactors(rTensor, 1.);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(lTensor.GetOrigin());
	return result;
}

CTensor* operator/(Scalar lTensor, CTensor& rTensor)
{
	bool ifgrad = rTensor.IfGrad();
	tyTensor data = lTensor / *rTensor.GetData();
	CTensor* result = new CTensor(data, ifgrad);
	result->SetLeafSta(false);
	// process the backward graph
	BackwardBase* bw;
	if (!ifgrad) {
		bw = new NoneBackward();
	}
	else {
		bw = new DivBackward();
		//the raw grad
		bw->SaveNextFuns(&rTensor, nullptr);
		bw->SetFactors(lTensor, 0.);
	}
	bw->SaveNode(result);
	result->SetGradFn(bw);
	// process the origin node
	result->SetOrigin(rTensor.GetOrigin());
	return result;
}

CTensor& CTensor::operator=(CTensor& tensor) {
	CTensor cpy = CTensor(tensor);
	return cpy;
}

CTensor& CTensor::operator=(CTensor&& tensor) {
	CTensor cpy = CTensor(tensor);
	return cpy;
}

ostream& operator<<(ostream& os, CTensor& t) {
	cout << "tensor{\n" << *t.GetData() << "\n}  ";
	t.GetGradFn()->ShowThis();
	return os;
}
ostream& operator<<(ostream& os, CTensor&& t) {
	cout << "tensor{\n" << *t.GetData() << "\n}  ";
	t.GetGradFn()->ShowThis();
	return os;
}


bool CTensor::operator<(CTensor& t) {
	if (GetData()->GetDataPtr()->sum() < t.GetData()->GetDataPtr()->sum()) {
		return true;
	}
	return false;
}

CTensor& CTensor::operator<<(Scalar d) {
	*m_pData << d;
	return *this;
}


#endif