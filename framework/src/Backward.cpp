#include "Backward.h"
#include "CTensor.h"
#include "struMat.h"
#include <algorithm>
using namespace std;

BackwardBase::BackwardBase() {
	emBwType = EmBackwardType::NONE;
	m_pTNode = nullptr;
	m_mpNextFuns = new MpFunDepend;

	vrtScaler = 1.;
	postPG = nullptr;
}

BackwardBase::~BackwardBase() {
	FreeRawGrad();
	FreeObj(m_mpNextFuns);
	m_pTNode = nullptr;
	postPG = nullptr;
}

void BackwardBase::FreeRawGrad() {
	auto its = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (auto it = its; it != ite; ++it) {
		it->first = nullptr;
		FreeObj(it->second);
	}
	m_mpNextFuns->clear();
}

void BackwardBase::ProcVrtScal(Scalar sca)
{
	vrtScaler *= sca;
}

void BackwardBase::SaveNode(CTensor* node) {
	this->m_pTNode = node;
}

void BackwardBase::SaveNextFuns(CTensor* node, tyTensor* rawGrad) {
	m_mpNextFuns->push_back(make_pair(node, rawGrad));
}

void BackwardBase::DeleteInleafNode(bool ifRetainSelf) {
	if (!ifRetainSelf)
	{
		if (m_pTNode->IsGradGet()) {
			if (!m_pTNode->IsLeaf()) {
				delete m_pTNode;
				return;
			}
		}
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (!it->first->IsLeaf()) {
			if (it->first->IsGradGet()) {
				FreeObj(it->first);
			}
			else it->first->GetGradFn()->DeleteInleafNode();
		}
	}
	if (!m_pTNode->IsLeaf() && !ifRetainSelf) delete m_pTNode;
}

void BackwardBase::ReleaseGrad() {
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (!it->first->IsLeaf()) {
			it->first->GetGradFn()->ReleaseGrad();
		}
	}
	m_pTNode->ClearGrad();
}

void BackwardBase::ReleaseFwGraph() {
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		it->first->GetGradFn()->ReleaseFwGraph();
	}
}

bool BackwardBase::IfAutogradEnd(CTensor*& preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool& ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return true;
	}
	if (ifModifiable)
		preNode->ScaleIP(vrtScaler);
	else
	{
		if (vrtScaler!=1.)
		{
			preNode = *preNode * vrtScaler;
			ifModifiable = true;
		}
	}
	return false;
}

bool BackwardBase::IfAutogradEnd(Scalar& preScaler, CTensor* root, VecTsr* vcResult) {
	if (root == m_pTNode) {
		tyTensor* data = new tyTensor(m_pTNode->GetData()->GetShape(), preScaler);
		vcResult->push_back(new CTensor(data, false, false));
		return true;
	}
	return false;
}

bool BackwardBase::ProcScalar(Scalar preScaler, CTensor*& preNode, VecTsr* vcGraph)
{
	if (preScaler == 1.)
	{
		return false;
	}
	preNode = preScaler * *preNode;
	vcGraph->push_back(preNode);
	return true;
}

EmBackwardType BackwardBase::GetBwType()
{
	return emBwType;
}

MpFunDepend* BackwardBase::GetDependNode() {
	return m_mpNextFuns;
}
//---------------------------------------------------

NoneBackward::NoneBackward()
{
	emBwType = EmBackwardType::NONE;
}

NoneBackward::~NoneBackward() {
}

void NoneBackward::Backward(tyTensor* grad) {
	return;
}

void NoneBackward::Backward(Scalar grad) {
	return;
}

void NoneBackward::ShowThis() {
	cout << "None" << endl;
}

//---------------------------------------------------

AccumulateGrad::AccumulateGrad()
{
	emBwType = EmBackwardType::ACCUMULATE;
}

AccumulateGrad::~AccumulateGrad() {
}

void AccumulateGrad::Backward(tyTensor* grad) {
	if (!m_pTNode->IsLeaf())
	{
		return;
	}
	tyTensor* temp = new tyTensor(*grad);
	temp->ReshapeIn(m_pTNode->GetGrad()->GetShape());
	m_pTNode->AddGrad(temp);
	FreeObj(temp);
}

void AccumulateGrad::Backward(Scalar grad) {
	if (!m_pTNode->IsLeaf())
	{
		return;
	}
	m_pTNode->AddGrad(grad);
}

void AccumulateGrad::ShowThis() {
	cout << "grad_fn=<AccumulateGrad>" << endl;
}


void AccumulateGrad::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		if (vcGraph->size())
		{
			vcGraph->pop_back();
		}
	}
}

void AccumulateGrad::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (root == m_pTNode) {
		CTensor* res;
		tyTensor* data = new tyTensor(m_pTNode->GetData()->GetShape(), preScaler);
		res = new CTensor(data, false, false);
		res->SetLeafSta(false);
		vcResult->push_back(res);
	}
}

//---------------------------------------------------

NegBackward::NegBackward()
{
	emBwType = EmBackwardType::NEG;
}

void NegBackward::Backward(tyTensor* grad)
{
	m_pTNode->AddGrad(grad);
	tyTensor* temp =new tyTensor(*grad * -vrtScaler);
	auto its = m_mpNextFuns->begin();
	its->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void NegBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= -vrtScaler;
	auto its = m_mpNextFuns->begin();
	its->first->GetGradFn()->Backward(grad);
}

void NegBackward::ShowThis()
{
	cout << "grad_fn=<NegBackward>" << endl;
}

void NegBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable)
{
	if (root == m_pTNode) 
	{
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next;
	if (ifModifiable)
	{
		preNode->ScaleIP(-vrtScaler);
		next = preNode;
	}
	else
	{
		next = -vrtScaler * *preNode;
		vcGraph->push_back(next);
		ifModifiable = true;
	}
	(*m_mpNextFuns)[0].first->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
}

void NegBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	(*m_mpNextFuns)[0].first->GetGradFn()->BackwardInAuto(-preScaler, root, vcGraph, vcResult);
}

//---------------------------------------------------

CopyBackward::CopyBackward() {
	emBwType = EmBackwardType::COPY;
}

CopyBackward::~CopyBackward() {
}

void CopyBackward::Backward(tyTensor* grad)
{
	m_pTNode->AddGrad(grad);
	auto its = m_mpNextFuns->begin();
	tyTensor* temp;
	if (vrtScaler != 1.)
	{
		temp = new tyTensor(vrtScaler * *grad);
		its->first->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
	else
	{
		its->first->GetGradFn()->Backward(grad);
	}
}

void CopyBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto its = m_mpNextFuns->begin();
	its->first->GetGradFn()->Backward(grad);
}

void CopyBackward::ShowThis() {
	cout << "grad_fn=<CopyBackward0>" << endl;
}

void CopyBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	(*m_mpNextFuns)[0].first->GetGradFn()->BackwardInAuto(preNode, root, vcGraph, vcResult, ifModifiable);
}

void CopyBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult) {
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	(*m_mpNextFuns)[0].first->GetGradFn()->BackwardInAuto(preScaler, root, vcGraph, vcResult);
}

//---------------------------------------------------
AddBackward::AddBackward() {
	emBwType = EmBackwardType::ADD;
}

AddBackward::~AddBackward() {
}

void AddBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	tyTensor* temp;
	CTensor* node;
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (node = it->first)
		{
			temp = new tyTensor(grad);
			temp->ReshapeIn(m_pTNode->GetData()->GetShape());
			temp->ReshapeIn(node->GetData()->GetShape());
			temp->ScaledIP(vrtScaler);
			node->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
	}
}

void AddBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (auto node = it->first)
		{
			node->GetGradFn()->Backward(grad);
		}
	}
}

void AddBackward::ShowThis() {
	cout << "grad_fn=<AddBackward0>" << endl;
}

void AddBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* node;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first)
		{
			if (node->IfSearchDir(root)) {
				node->GetGradFn()->BackwardInAuto(preNode, root, vcGraph, vcResult, ifModifiable);
			}
		}
	}
}

void AddBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult) {
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* node;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first)
		{
			if (node->IfSearchDir(root)) {
				node->GetGradFn()->BackwardInAuto(preScaler, root, vcGraph, vcResult);
			}
		}
	}
}
//---------------------------------------------------

SubBackward::SubBackward() {
	emBwType = EmBackwardType::SUB;
}

SubBackward::~SubBackward() {
}

void SubBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	CTensor* node;
	tyTensor* temp;
	temp = new tyTensor(grad);
	auto left = (*m_mpNextFuns)[0];
	auto right = (*m_mpNextFuns)[1];
	temp->ReshapeIn(m_pTNode->GetData()->GetShape());
	if (node = left.first)
	{
		temp->ScaledIP(vrtScaler);
		temp->ReshapeIn(node->GetData()->GetShape());
		node->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
	if (node = right.first)
	{
		temp = new tyTensor(*grad * -1);
		temp->ReshapeIn(node->GetData()->GetShape());
		node->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
	//
}

void SubBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	CTensor* node;
	auto left = (*m_mpNextFuns)[0];
	auto right = (*m_mpNextFuns)[1];
	if (node = left.first)
	{
		node->GetGradFn()->Backward(grad);
	}
	if (node = right.first)
	{
		grad *= -1;
		node->GetGradFn()->Backward(grad);
	}
}

void SubBackward::ShowThis() {
	cout << "grad_fn=<SubBackward0>" << endl;
}


void SubBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* node;
	auto left = (*m_mpNextFuns)[0];
	auto right = (*m_mpNextFuns)[1];
	if (node = left.first)
	{
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(preNode, root, vcGraph, vcResult, ifModifiable);
		}
	}
	if (node = right.first)
	{
		if (node->IfSearchDir(root)) {
			CTensor* next;
			if (ifModifiable)
			{
				preNode->ScaleIP(-1.);
				next = preNode;
			}
			else
			{
				next = -*preNode;
				ifModifiable = true;
				vcGraph->push_back(next);
			}
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
		}
	}
}

void SubBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* node;
	auto left = (*m_mpNextFuns)[0];
	auto right = (*m_mpNextFuns)[1];
	if (node = left.first)
	{
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(preScaler, root, vcGraph, vcResult);
		}
	}
	if (node = right.first)
	{
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(-preScaler, root, vcGraph, vcResult);
		}
	}
}

//---------------------------------------------------

MulBackward::MulBackward() {
	scaler = 1;
	emBwType = EmBackwardType::MUL;
}

MulBackward::~MulBackward() {
}

void MulBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	tyTensor* temp;
	CTensor* node;
	if (m_mpNextFuns->size() != 2)
	{
		node = (*m_mpNextFuns)[0].first;
		temp = new tyTensor(*grad * (scaler * vrtScaler));
		temp->ReshapeIn(node->GetData()->GetShape());
		node->GetGradFn()->Backward(temp);
		FreeObj(temp);
		return;
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (node = it->first)
		{
			temp = new tyTensor(*it->second * *grad);
			temp->ReshapeIn(node->GetData()->GetShape());
			temp->ScaledIP(vrtScaler);
			node->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
	}
}

void MulBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	tyTensor* temp;
	CTensor* node;
	if (m_mpNextFuns->size() != 2)
	{
		node = (*m_mpNextFuns)[0].first;
		grad *= scaler;
		node->GetGradFn()->Backward(grad);
		return;
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		if (node = it->first)
		{
			temp = new tyTensor(*it->second * grad);
			temp->ReshapeIn(node->GetData()->GetShape());
			node->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
	}
}

void MulBackward::SetFactors(Scalar scaler, Scalar placeholder) {
	this->scaler = scaler;
}

void MulBackward::ShowThis() {
	cout << "grad_fn=<MulBackward0>" << endl;
}

void MulBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* node;
	CTensor* next;
	if (m_mpNextFuns->size() != 2)
	{
		node = (*m_mpNextFuns)[0].first;
		if (node->IfSearchDir(root)) {
			if (ifModifiable)
			{
				preNode->ScaleIP(scaler);
				next = preNode;
			}
			else
			{
				next = *preNode * scaler;
				vcGraph->push_back(next);
				ifModifiable = true;
			}
			next->GetData()->ReshapeIn(node->GetData()->GetShape());
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
			return;
		}
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	int n = 1;
	for (; it != ite; ++it) {
		if (node = it->first)
		{
			if (node->IfSearchDir(root)) {
				next = *preNode * *(*m_mpNextFuns)[n].first;
				next->GetData()->ReshapeIn(node->GetData()->GetShape());
				vcGraph->push_back(next);
				node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
			}
		}
		--n;
	}
}

void MulBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* node;
	CTensor* next;
	if (m_mpNextFuns->size() != 2)
	{
		node = (*m_mpNextFuns)[0].first;
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(preScaler * scaler, root, vcGraph, vcResult);
			return;
		}
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	int n = 1;
	for (; it != ite; ++it) {
		if (node = it->first)
		{
			if (node->IfSearchDir(root)) {
				if (preScaler == 1.)
				{
					node->GetGradFn()->BackwardInAuto((*m_mpNextFuns)[n].first, root, vcGraph, vcResult, false);
					--n;
					continue;
				}
				next = preScaler * *(*m_mpNextFuns)[n].first;
				next->GetData()->ReshapeIn(node->GetData()->GetShape());
				vcGraph->push_back(next);
				node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
			}
		}
		--n;
	}
}

//---------------------------------------------------


DivBackward::DivBackward()
{
	emBwType = EmBackwardType::DIV;
	fac = 1;
	pos = 0;
}

DivBackward::~DivBackward() {
}

void DivBackward::SetFactors(Scalar fac, Scalar pos) {
	this->fac = fac;
	this->pos = pos;
}

void DivBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	tyTensor* temp;
	if (m_mpNextFuns->size() != 2)
	{
		auto it = m_mpNextFuns->begin()->first;
		if (pos)
		{
			temp = new tyTensor(*grad / fac);
			temp->ReshapeIn(it->GetData()->GetShape());
			temp->ScaledIP(vrtScaler);
			it->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
		else
		{
			temp = it->GetData()->Pow(-2, -fac);
			*temp *= *grad;
			temp->ReshapeIn(it->GetData()->GetShape());
			temp->ScaledIP(vrtScaler);
			it->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
		return;
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		temp = new tyTensor(*it->second * *grad);
		temp->ReshapeIn(it->first->GetData()->GetShape());
		temp->ScaledIP(vrtScaler);
		it->first->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
}

void DivBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	tyTensor* temp;
	if (m_mpNextFuns->size() != 2)
	{
		auto it = m_mpNextFuns->begin()->first;
		if (pos)
		{
			it->GetGradFn()->Backward(grad / fac);
		}
		else
		{
			temp = it->GetData()->Pow(-2, -fac * grad);
			temp->ReshapeIn(it->GetData()->GetShape());
			it->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
		return;
	}
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		temp = new tyTensor(*it->second * grad);
		temp->ReshapeIn(it->first->GetData()->GetShape());
		it->first->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
}

void DivBackward::ShowThis() {
	cout << "grad_fn=<DivBackward0>" << endl;
}

void DivBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* next;
	if (m_mpNextFuns->size() != 2)
	{
		auto node = m_mpNextFuns->begin()->first;
		if (pos)
		{
			if (ifModifiable)
			{
				preNode->ScaleIP(1. / fac);
				next = preNode;
			}
			else
			{
				next = *preNode / fac;
				vcGraph->push_back(next);
				ifModifiable = true;
			}
			next->GetData()->ReshapeIn(node->GetData()->GetShape());
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
		}
		else
		{
			next = node->Pow(-2, -fac);
			vcGraph->push_back(next);
			next = *next * *preNode;
			vcGraph->push_back(next);
			next->GetData()->ReshapeIn(node->GetData()->GetShape());
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		}
		return;
	}

	CTensor* left, * right;
	left = (*m_mpNextFuns)[0].first;
	right = (*m_mpNextFuns)[1].first;
	bool ifLeft = left->IfSearchDir(root);
	bool ifRight = right->IfSearchDir(root);
	if (ifLeft) {
		next = right->Pow(-1);
		vcGraph->push_back(next);
		next = *next * *preNode;
		vcGraph->push_back(next);
		next->GetData()->ReshapeIn(left->GetData()->GetShape());
		left->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (ifRight) {
		next = right->Pow(-2, -1);
		vcGraph->push_back(next);
		next = *left * *next;
		vcGraph->push_back(next);
		next = *next * *preNode;
		vcGraph->push_back(next);
		next->GetData()->ReshapeIn(right->GetData()->GetShape());
		right->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

void DivBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* next;
	if (m_mpNextFuns->size() != 2)
	{
		auto node = m_mpNextFuns->begin()->first;
		if (pos)
		{
			node->GetGradFn()->BackwardInAuto(preScaler / fac, root, vcGraph, vcResult);
		}
		else
		{
			next = node->Pow(-2, -preScaler * fac);
			vcGraph->push_back(next);
			next->GetData()->ReshapeIn(node->GetData()->GetShape());
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		}
		return;
	}

	CTensor* left, * right;
	left = (*m_mpNextFuns)[0].first;
	right = (*m_mpNextFuns)[1].first;
	bool ifLeft = left->IfSearchDir(root);
	bool ifRight = right->IfSearchDir(root);
	if (ifLeft) {
		next = right->Pow(-1, preScaler);
		vcGraph->push_back(next);
		next->GetData()->ReshapeIn(left->GetData()->GetShape());
		left->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (ifRight) {
		next = right->Pow(-2, -preScaler);
		vcGraph->push_back(next);
		next = *left * *next;
		vcGraph->push_back(next);
		next->GetData()->ReshapeIn(right->GetData()->GetShape());
		right->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

//---------------------------------------------------
SumBackward::SumBackward() {
	emBwType = EmBackwardType::SUM;
}

SumBackward::~SumBackward() {
}

void SumBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto it = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*it->second * *grad);
	temp->ReshapeIn(it->first->GetData()->GetShape());
	temp->ScaledIP(vrtScaler);
	it->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void SumBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto it = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*it->second * grad);
	temp->ReshapeIn(it->first->GetData()->GetShape());
	it->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void SumBackward::ShowThis() {
	cout << "grad_fn=<SumBackward0>" << endl;
}

void SumBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;

}

//---------------------------------------------------
MmBackward::MmBackward() {
	lT = false;
	rT = false;
	pos = 0;
	scaler = 1.;
	emBwType = EmBackwardType::MM;
}

MmBackward::~MmBackward() {
}

void MmBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	if (m_mpNextFuns->size() != 2)
	{
		CTensor* node = (*m_mpNextFuns)[0].first; 
		tyTensor* temp;
		if (pos)
		{
			temp = new tyTensor(*grad->GetDataPtr()*MatrData::Constant(cols, rows,scaler));
			if (lT)
			{
				temp->TransposeIP();
			}
			temp->ScaledIP(vrtScaler);
			node->GetGradFn()->Backward(temp);
		}
		else
		{
			temp = new tyTensor(MatrData::Constant(cols, rows, scaler) * grad->GetDataPtr()->transpose());
			if (rT)
			{
				temp->TransposeIP();
			}
			temp->ScaledIP(vrtScaler);
			node->GetGradFn()->Backward(temp);
		}
		FreeObj(temp);
		return;
	}
	tyTensor* tgrad = new tyTensor(grad);
	auto itl = (*m_mpNextFuns)[0];
	auto itr = (*m_mpNextFuns)[1];
	tyTensor* temp;
	if (lT && !rT)
	{
		auto d = tgrad->Transpose();
		temp = itl.second->Dot(&d);
	}
	else
	{
		temp = tgrad->Dot(itl.second);
	}
	temp->ScaledIP(vrtScaler);
	itl.first->GetGradFn()->Backward(temp);
	FreeObj(temp);

	if (!lT && rT)
	{
		auto d = tgrad->Transpose();
		temp = d.Dot(itr.second);
	}
	else
	{
		temp = itr.second->Dot(tgrad);
	}
	temp->ScaledIP(vrtScaler);
	FreeObj(tgrad);
	itr.first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void MmBackward::Backward(Scalar grad) {
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	if (m_mpNextFuns->size() != 2)
	{
		CTensor* node = (*m_mpNextFuns)[0].first;
		tyTensor* temp;
		if (pos)
		{
			temp = new tyTensor(MatrData::Ones(m_pTNode->GetData()->GetDataPtr()->rows(), cols)*MatrData::Constant(cols, rows, scaler));
			temp->ScaledIP(grad);
			if (lT)
			{
				temp->TransposeIP();
			}
			node->GetGradFn()->Backward(temp);
		}
		else
		{
			temp = new tyTensor(MatrData::Constant(cols, rows, scaler)*MatrData::Ones(rows, m_pTNode->GetData()->GetDataPtr()->cols()));
			temp->ScaledIP(grad);
			if (rT)
			{
				temp->TransposeIP();
			}
			node->GetGradFn()->Backward(temp);
		}
		FreeObj(temp);
		return;
	}
	tyTensor* tgrad = new tyTensor(m_pTNode->GetData()->GetShape(), 1.);
	auto itl = (*m_mpNextFuns)[0];
	auto itr = (*m_mpNextFuns)[1];
	tyTensor* temp;
	if (lT && !rT)
	{
		auto d = tgrad->Transpose();
		temp = itl.second->Dot(&d);
	}
	else
	{
		temp = tgrad->Dot(itl.second);
	}
	temp->ScaledIP(grad);
	itl.first->GetGradFn()->Backward(temp);
	FreeObj(temp);

	if (!lT && rT)
	{
		auto d = tgrad->Transpose();
		temp = d.Dot(itr.second);
	}
	else
	{
		temp = itr.second->Dot(tgrad);
	}
	temp->ScaledIP(grad);
	itr.first->GetGradFn()->Backward(temp);
	FreeObj(temp);
	FreeObj(tgrad);
}

void MmBackward::ShowThis() {
	cout << "grad_fn=<MmBackward0>" << endl;
}

void MmBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	//
	if (m_mpNextFuns->size() != 2)
	{
		CTensor* node = (*m_mpNextFuns)[0].first;
		CTensor* next;
		if (pos)
		{
			if (!lT)
			{
				next = CTensor::Dot(preNode, scaler, rows, false);
			}
			else
			{
				next = CTensor::Dot(scaler, preNode, rows, true);
			}
		}
		else
		{
			if (!rT)
			{
				next = CTensor::Dot(scaler, preNode, cols, false);
			}
			else
			{
				next = CTensor::Dot(preNode, scaler, cols, true);
			}

		}
		vcGraph->push_back(next);
		node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		return;
	}
	//
	CTensor* left, * right, * next;
	left = (*m_mpNextFuns)[0].first;
	right = (*m_mpNextFuns)[1].first;
	if (left->IfSearchDir(root)) {
		if (lT && !rT)
		{
			next = CTensor::Dot(right, preNode, false, true);
		}
		else
		{
			if (rT)
			{
				next = CTensor::Dot(preNode, right);
			}
			else
			{
				next = CTensor::Dot(preNode, right, false, true);
			}
		}
		vcGraph->push_back(next);
		left->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (right->IfSearchDir(root)) {
		if (!lT && rT)
		{
			next = CTensor::Dot(preNode, left, true, false);
		}
		else
		{
			if (lT)
			{
				next = CTensor::Dot(left, preNode);
			}
			else
			{
				next = CTensor::Dot(left, preNode, true, false);
			}
		}
		vcGraph->push_back(next);
		right->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

void MmBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	//
	int c1, c2;
	c1 = m_pTNode->GetData()->GetDataPtr()->rows();
	c2 = m_pTNode->GetData()->GetDataPtr()->cols();
	if (m_mpNextFuns->size() != 2)
	{
		CTensor* node = (*m_mpNextFuns)[0].first;
		CTensor* next;
		if (pos)
		{
			if (!lT)
			{
				MatrData* data = new MatrData(MatrData::Ones(c1, c2)*MatrData::Ones(cols, rows));
				next = new CTensor(data, true, false);
			}
			else
			{
				MatrData* data = new MatrData(MatrData::Ones(rows, cols)*MatrData::Ones(c2, c1));
				next = new CTensor(data, true, false);
			}
			next->ScaleIP(preScaler);
			vcGraph->push_back(next);
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		}
		else
		{
			if (!rT)
			{
				MatrData* data = new MatrData(MatrData::Ones(cols, rows)*MatrData::Ones(c1, c2));
				next = new CTensor(data, true, false);
			}
			else
			{
				MatrData* data = new MatrData(MatrData::Ones(c2, c1)*MatrData::Ones(rows, cols));
				next = new CTensor(data, true, false);
			}
			vcGraph->push_back(next);
			next->ScaleIP(preScaler);
			node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		}
		return;
	}
	//
	CTensor* left, * right, * next;
	left = (*m_mpNextFuns)[0].first;
	right = (*m_mpNextFuns)[1].first;
	if (left->IfSearchDir(root)) {
		if (lT && !rT)
		{
			next = CTensor::Dot(right, 1., c1, false);
		}
		else
		{
			if (rT)
			{
				next = CTensor::Dot(1., right, c1, false);
			}
			else
			{
				next = CTensor::Dot(1., right, c1, true);
			}
		}
		next->ScaleIP(preScaler);
		vcGraph->push_back(next);
		left->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (right->IfSearchDir(root)) {
		if (!lT && rT)
		{
			next = CTensor::Dot(1., left, c2, false);
		}
		else
		{
			if (lT)
			{
				next = CTensor::Dot(left, 1., c2, false);
			}
			else
			{
				next = CTensor::Dot(left, 1., c2, true);
			}
		}
		next->ScaleIP(preScaler);
		vcGraph->push_back(next);
		right->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

void MmBackward::SetTransSta(bool lT, bool rT)
{
	this->lT = lT;
	this->rT = rT;
}

void MmBackward::SetCoeff(int rows, int cols, Scalar scaler, int pos /*= 0*/)
{
	this->rows = rows;
	this->cols = cols;
	this->scaler = scaler;
	this->pos = pos;
}

//---------------------------------------------------

AddmmBackward::AddmmBackward()
{
	emBwType = EmBackwardType::ADDMM;
	beta = 1.;
	alpha = 1.;
}

AddmmBackward::~AddmmBackward() {
}

void AddmmBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	tyTensor* temp = grad->Scaled(beta);
	(*m_mpNextFuns)[0].first->GetGradFn()->Backward(temp);
	FreeObj(temp); 
	temp = grad->Dot((*m_mpNextFuns)[1].second);
	temp->ScaledIP(alpha);
	(*m_mpNextFuns)[1].first->GetGradFn()->Backward(temp);
	FreeObj(temp);
	temp = (*m_mpNextFuns)[2].second->Dot(grad);
	temp->ScaledIP(alpha);
	(*m_mpNextFuns)[2].first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void AddmmBackward::Backward(Scalar grad)
{
	int rows = m_pTNode->GetData()->GetShape().first;
	int cols = m_pTNode->GetData()->GetShape().second;
	tyTensor ones(rows, cols);
	m_pTNode->AddGrad(grad);
	(*m_mpNextFuns)[0].first->GetGradFn()->Backward(grad * beta);
	tyTensor* temp = ones.Dot((*m_mpNextFuns)[1].second);
	temp->ScaledIP(alpha);
	(*m_mpNextFuns)[1].first->GetGradFn()->Backward(temp);
	FreeObj(temp);
	temp = (*m_mpNextFuns)[2].second->Dot(&ones);
	temp->ScaledIP(alpha);
	(*m_mpNextFuns)[2].first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void AddmmBackward::ShowThis() {
	cout << "grad_fn=<AddmmBackward>" << endl;
}

void AddmmBackward::SetFactors(Scalar beta /*= 1*/, Scalar alpha /*= 1*/)
{
	this->beta = beta;
	this->alpha = alpha;
}

void AddmmBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	Scalar fac;
	CTensor* mat1, * mat2, * mat3;
	CTensor* next;
	mat1 = (*m_mpNextFuns)[0].first;
	mat2 = (*m_mpNextFuns)[1].first;
	mat3 = (*m_mpNextFuns)[2].first;
	bool ifmat1 = mat1->IfSearchDir(root);
	bool ifmat2 = mat2->IfSearchDir(root);
	bool ifmat3 = mat3->IfSearchDir(root);
	if (ifmat1) {
		if (fac = alpha * vrtScaler != 1.)
		{
			next = *preNode * fac; 
			ifModifiable = true;
			vcGraph->push_back(next);
		}
		else
			next = preNode;
		mat1->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
	}
	if (ifmat2) {
		next = preNode->Dot(mat3, false, true);
		vcGraph->push_back(next);
		next->ScaleIP(alpha * vrtScaler);
		mat2->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (ifmat3) {
		next = mat2->Dot(preNode, true);
		vcGraph->push_back(next);
		next->ScaleIP(alpha * vrtScaler);
		mat3->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

void AddmmBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	Scalar fac;
	CTensor* mat1, * mat2, * mat3;
	CTensor* next;
	mat1 = (*m_mpNextFuns)[0].first;
	mat2 = (*m_mpNextFuns)[1].first;
	mat3 = (*m_mpNextFuns)[2].first;
	bool ifmat1 = mat1->IfSearchDir(root);
	bool ifmat2 = mat2->IfSearchDir(root);
	bool ifmat3 = mat3->IfSearchDir(root);
	if (ifmat1) {
		mat1->GetGradFn()->BackwardInAuto(preScaler * vrtScaler * beta, root, vcGraph, vcResult);
	}
	if (ifmat2) {
		next = CTensor::Dot(1., mat3, m_pTNode->GetData()->GetShape().first, true);
		next->ScaleIP(alpha * vrtScaler * preScaler);
		vcGraph->push_back(next);
		mat2->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
	if (ifmat3) {
		next = CTensor::Dot(mat2, 1., m_pTNode->GetData()->GetShape().second, true);
		next->ScaleIP(alpha * vrtScaler * preScaler);
		vcGraph->push_back(next);
		mat3->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	}
}

//---------------------------------------------------


PowBackward::PowBackward() {
	powCoe = 1;
	sign = 1;
	emBwType = EmBackwardType::POW;
}

PowBackward::~PowBackward() {
}

void PowBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto it = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*it->second * *grad);
	temp->ReshapeIn(it->first->GetData()->GetShape());
	temp->ScaledIP(vrtScaler);
	it->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void PowBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto it = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*it->second * grad);
	temp->ReshapeIn(it->first->GetData()->GetShape());
	it->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void PowBackward::ShowThis() {
	cout << "grad_fn=<PowBackward0>" << endl;
}

void PowBackward::SetFactors(Scalar powCoe, Scalar sign) {
	this->powCoe = powCoe;
	this->sign = sign;
}

void PowBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	//if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* next, * node;
	node = (*m_mpNextFuns)[0].first;
	next = node->Pow(powCoe - 1, powCoe * sign * vrtScaler);
	vcGraph->push_back(next);
	next = *next * *preNode;
	vcGraph->push_back(next);
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
}

void PowBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult) {
	if (root == m_pTNode) {
		tyTensor* data = new tyTensor(m_pTNode->GetData()->GetShape(), preScaler);
		vcResult->push_back(new CTensor(data, false, false));
		return;
	}
	//if (IfAutogradEnd(preScaler, root, vcGraph, vcResult)) return;
	CTensor* next, * node;
	node = (*m_mpNextFuns)[0].first;
	next = node->Pow(powCoe - 1, powCoe * sign * preScaler * vrtScaler);
	vcGraph->push_back(next);
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

//---------------------------------------------------


ExpBackward::ExpBackward() {
	coeExp = 1;
	sign = 1;
	emBwType = EmBackwardType::EXP;
}

ExpBackward::~ExpBackward() {
}

void ExpBackward::SetFactors(Scalar coeExp, Scalar sign) {
	this->coeExp = coeExp;
	this->sign = sign;
}

void ExpBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto node = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*node->second * *grad);
	temp->ReshapeIn(node->first->GetData()->GetShape());
	temp->ScaledIP(vrtScaler);
	node->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void ExpBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto node = m_mpNextFuns->begin();
	tyTensor* temp = new tyTensor(*node->second * grad);
	temp->ReshapeIn(node->first->GetData()->GetShape());
	node->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void ExpBackward::ShowThis() {
	cout << "grad_fn=<ExpBackward0>" << endl;
}

void ExpBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	next = dep->ExpEach(coeExp, coeExp * sign * vrtScaler);
	vcGraph->push_back(next);
	next = *next * *preNode;
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}


void ExpBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult) {
	if (root == m_pTNode) {
		tyTensor* data = new tyTensor(m_pTNode->GetData()->GetShape(), preScaler);
		vcResult->push_back(new CTensor(data, false, false));
		return;
	}
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	next = dep->ExpEach(coeExp, coeExp * sign * preScaler * vrtScaler);
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}
//--------------------------------------

AccBackward::AccBackward()
{
	emBwType = EmBackwardType::ACC;
}

AccBackward::~AccBackward() {
}

void AccBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	tyTensor* temp, * ones;
	for (; it != ite; ++it) {
		temp = new tyTensor(grad);
		temp->ReshapeIn(m_pTNode->GetData()->GetShape());
		temp->ReshapeIn(it->first->GetData()->GetShape());
		temp->ScaledIP(vrtScaler);
		it->first->GetGradFn()->Backward(temp);
		FreeObj(temp);
	}
}

void AccBackward::Backward(Scalar grad) {
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto it = m_mpNextFuns->begin();
	auto ite = m_mpNextFuns->end();
	for (; it != ite; ++it) {
		it->first->GetGradFn()->Backward(grad);
	}
}

void AccBackward::ShowThis() {
	cout << "grad_fn=<AccBackward0>" << endl;
}

void AccBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* node;
	for (auto it : (*m_mpNextFuns)) {
		node = it.first;
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(preNode, root, vcGraph, vcResult, ifModifiable);
		}
	}
}

void AccBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult) {
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* node;
	for (auto it : (*m_mpNextFuns)) {
		node = it.first;
		if (node->IfSearchDir(root)) {
			node->GetGradFn()->BackwardInAuto(preScaler, root, vcGraph, vcResult);
		}
	}
}

//--------------------------------------


MeanBackward::MeanBackward()
{
	emBwType = EmBackwardType::MEAN;
}

MeanBackward::~MeanBackward() {

}

void MeanBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto it = m_mpNextFuns->begin();
	Scalar mean = 1. / it->first->GetData()->GetSize();
	tyTensor* temp = new tyTensor(*grad * mean);
	temp->ReshapeIn(it->first->GetData()->GetShape());
	temp->ScaledIP(vrtScaler);
	it->first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void MeanBackward::Backward(Scalar grad) {
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto it = m_mpNextFuns->begin();
	Scalar mean = 1. / it->first->GetData()->GetSize();
	it->first->GetGradFn()->Backward(grad * mean);
}

void MeanBackward::ShowThis() {
	cout << "grad_fn=<MeanBackward>" << endl;
}

void MeanBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (preNode->GetData()->GetSize() != 1)
	{
		preNode = preNode->Sum();
	}
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	Scalar mean = dep->GetData()->GetDataPtr()->size();
	mean = vrtScaler / mean;
	next = new CTensor(
		dep->GetData()->GetShape().first, 
		dep->GetData()->GetShape().second,
		mean, false);
	next->SetLeafSta(false);
	vcGraph->push_back(next);
	next = *next * *preNode;
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void MeanBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	Scalar mean = dep->GetData()->GetDataPtr()->size();
	mean = 1. / mean;
	next = new CTensor(
		dep->GetData()->GetShape().first,
		dep->GetData()->GetShape().second,
		mean * preScaler, false);
	next->SetLeafSta(false);
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

//--------------------------------------


CatBackward::CatBackward() {
	dim = 0;
	mtc1 = 0;
	emBwType = EmBackwardType::CAT;
}


CatBackward::~CatBackward() {

}

void CatBackward::SetFactors(Scalar dim /*= 0*/, Scalar mtc1 /*= 0*/) {
	this->dim = dim;
	this->mtc1 = mtc1;
}


void CatBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	tyTensor* temp;
	MatrData* matData;
	CTensor* node;
	int i = 0, j = 0, p = 0, q = 0, rows, cols;
	rows = grad->GetDataPtr()->rows();
	cols = grad->GetDataPtr()->cols();
	dim ? p = rows : q = cols;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first) {
			matData = node->GetData()->GetDataPtr();
			dim ?  j += q, q = matData->cols():
				i += p, p = matData->rows();
			temp = new tyTensor(grad->GetDataPtr()->block(i, j, p, q));
			temp->ScaledIP(vrtScaler);
			node->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
		else {
			dim ? j += mtc1 :i += mtc1;
		}
	}
}

void CatBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	tyTensor* temp;
	MatrData* matData;
	CTensor* node;
	int p = 0, q = 0, rows, cols;
	rows = m_pTNode->GetData()->GetDataPtr()->rows();
	cols = m_pTNode->GetData()->GetDataPtr()->cols();
	dim ? p = rows : q = cols;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first) {
			matData = node->GetData()->GetDataPtr();
			dim ? q = matData->cols() :
				p = matData->rows();
			temp = new tyTensor(p, q, grad);
			node->GetGradFn()->Backward(temp);
			FreeObj(temp);
		}
	}
}

void CatBackward::ShowThis() {
	cout << "grad_fn=<CatBackward>" << endl;
}

void CatBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* next, * res, *node;
	int i = 0, j = 0, pos = 0, p = 0, q = 0, rows, cols, leng;
	rows = preNode->GetData()->GetDataPtr()->rows();
	cols = preNode->GetData()->GetDataPtr()->cols();
	dim ? p = rows, leng = cols : q = cols, leng = rows;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first) {
			dim ? q = node->GetData()->GetDataPtr()->cols() :
				p = node->GetData()->GetDataPtr()->rows();
			if (node->IfSearchDir(root)) {
				next = new CTensor(new tyTensor(preNode->GetData()->GetDataPtr()->block(i, j, p, q)), true, false);
				BackwardBase* bw = new SliceBackward(dim, leng, pos);
				bw->SaveNextFuns(preNode, nullptr);
				bw->SaveNode(next);
				next->SetLeafSta(false);
				next->SetGradFn(bw);
				next->SetOrigin(preNode->GetOrigin());
				vcGraph->push_back(next);
				node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
				//}
			}
		}
		dim ? j += mtc1 : i += mtc1;
		++pos;
	}
}


void CatBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* next, * node;
	int p = 0, q = 0, rows, cols;
	rows = m_pTNode->GetData()->GetDataPtr()->rows();
	cols = m_pTNode->GetData()->GetDataPtr()->cols();
	dim ? p = cols: q = rows;
	for (auto it : (*m_mpNextFuns)) {
		if (node = it.first) {
			dim ? q = node->GetData()->GetDataPtr()->cols() :
				p = node->GetData()->GetDataPtr()->rows();
			if (node->IfSearchDir(root)) {
				next = new CTensor(p, q, preScaler, false);
				vcGraph->push_back(next);
				node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
				//}
			}
		}
	}
}

//--------------------------------------

SliceBackward::SliceBackward(int dim, int leng, int pos) {
	this->dim = dim;
	this->leng = leng;
	this->pos = pos;
	emBwType = EmBackwardType::SLICE;
	dataPG = nullptr;
}

SliceBackward::~SliceBackward() {
	FreeObj(dataPG);
}

void SliceBackward::Backward(tyTensor* grad) 
{
	m_pTNode->AddGrad(grad);
	tyTensor* next;
	int rows, cols, mtcx, mtcy, mtc1, i = 0, j = 0;
	MatrData* pdata = grad->GetDataPtr();
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
	auto datanew = new MatrData(rows, cols);
	if (pos == 0) {
		datanew->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
		dim == 0 ? i = pdata->rows() : j = pdata->cols();
		datanew->block(i, j, mtcx, mtcy) << zero;
	}
	else {
		datanew->block(i, j, mtcx, mtcy) << zero;
		dim == 0 ? i = mtcx : j = mtcy;
		datanew->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
	}
	next = new tyTensor(datanew, false);
	next->ScaledIP(vrtScaler);
	(*m_mpNextFuns)[0].first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void SliceBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	ProcOnesMat();
	tyTensor* temp = new tyTensor(grad * *dataPG);
	(*m_mpNextFuns)[0].first->GetGradFn()->Backward(temp);
	FreeObj(temp);
}

void SliceBackward::ShowThis() {
	cout << "grad_fn=<SliceBackward>" << endl;
}

void SliceBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (IfAutogradEnd(preNode, root, vcGraph, vcResult, ifModifiable)) return;
	CTensor* next, * node;
	node = (*m_mpNextFuns)[0].first;
	next = preNode->FillWith0(dim, leng, pos);
	vcGraph->push_back(next);
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void SliceBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	ProcOnesMat();
	CTensor* node, * next;
	auto data = preScaler * dataPG->array();
	next = new CTensor(data, false);
	vcGraph->push_back(next);
	node = (*m_mpNextFuns)[0].first;
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void SliceBackward::ProcOnesMat()
{
	if (!dataPG)
	{
		int rows, cols, mtcx, mtcy, mtc1, i = 0, j = 0;
		MatrData* pdata = m_pTNode->GetData()->GetDataPtr();
		rows = pdata->rows();
		cols = pdata->cols();
		pdata = new MatrData(MatrData::Ones(rows, cols));
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
		dataPG = new MatrData(rows, cols);
		if (pos == 0) {
			dataPG->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
			dim == 0 ? i = pdata->rows() : j = pdata->cols();
			dataPG->block(i, j, mtcx, mtcy) << zero;
		}
		else {
			dataPG->block(i, j, mtcx, mtcy) << zero;
			dim == 0 ? i = mtcx : j = mtcy;
			dataPG->block(i, j, pdata->rows(), pdata->cols()) << *pdata;
		}
		FreeObj(pdata);
	}
}

//-----------------------the Loss Functions Backward--------------------------

MseLossBackward::MseLossBackward()
{
	emBwType = EmBackwardType::MSE;
}

MseLossBackward::~MseLossBackward() {
}

void MseLossBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto its = m_mpNextFuns->begin();
	*grad = *its->second * *grad;
	grad->ReshapeIn(its->first->GetData()->GetShape());
	its->first->GetGradFn()->Backward(grad);
}

void MseLossBackward::ShowThis() {
	cout << "grad_fn=<MseLossBackward>" << endl;
}
