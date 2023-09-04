#include "FuncBackward.h"
#include "CTensor.h"
using namespace std;

QuadraPolyBackward::QuadraPolyBackward(Scalar a, Scalar b)
{
	this->a = a;
	this->b = b;
}

QuadraPolyBackward::~QuadraPolyBackward()
{

}

void QuadraPolyBackward::Backward(tyTensor* grad)
{
	m_pTNode->AddGrad(grad);
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * *grad);
	next->ScaledIP(vrtScaler);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void QuadraPolyBackward::Backward(Scalar grad)
{
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * grad);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void QuadraPolyBackward::ShowThis()
{
	cout << "grad_fn=<QuadraPolyBackward>" << endl;
}

void QuadraPolyBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable)
{
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next, * node;
	node = (*m_mpNextFuns)[0].first;
	if (postPG)
	{
		next = *postPG * *preNode;
		vcGraph->push_back(next);
		node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
		return;
	}
	next = CmpPostPG(vcGraph);
	next = *preNode * *next;
	vcGraph->push_back(next);
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void QuadraPolyBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	bool ifModifiable;
	CTensor* next, * node;
	node = (*m_mpNextFuns)[0].first;
	if (postPG)
	{
		if (preScaler != 1.)
		{
			next = preScaler * *postPG;
			vcGraph->push_back(next); 
			ifModifiable = true;
		}
		else
		{
			next = postPG;
			ifModifiable = false;
		}
		node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
		return;
	}
	next = CmpPostPG(vcGraph);
	ifModifiable = ProcScalar(preScaler, next, vcGraph);
	node->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
}

CTensor* QuadraPolyBackward::CmpPostPG(VecTsr* vcGraph)
{
	CTensor* node;
	node = (*m_mpNextFuns)[0].first;
	postPG = node->QuadraPoly(0, 2. * a, b);
	vcGraph->push_back(postPG);
	postPG->ScaleIP(vrtScaler);
	return postPG;
}


//-----------------------the Activation Functions Backward--------------------------

SigmoidBackward::SigmoidBackward()
{
	emBwType = EmBackwardType::SIGMOID;
}

SigmoidBackward::~SigmoidBackward() {
}

void SigmoidBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * *grad);
	next->ScaledIP(vrtScaler);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void SigmoidBackward::Backward(Scalar grad) {
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * grad);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void SigmoidBackward::ShowThis() {
	cout << "grad_fn=<SigmoidBackward>" << endl;
}

void SigmoidBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	if (postPG)
	{
		next = *postPG * *preNode;
		vcGraph->push_back(next);
		dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		return;
	}
	next = CmpPostPG(vcGraph);
	next = *preNode * *next;
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void SigmoidBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	CTensor* next, * dep;
	bool ifModifiable;
	dep = (*m_mpNextFuns)[0].first;
	if (postPG)
	{
		if (preScaler != 1.)
		{
			next = preScaler * *postPG;
			vcGraph->push_back(next);
			ifModifiable = true;
		}
		else
		{
			next = postPG;
			ifModifiable = false;
		}
		dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		return;
	}
	next = CmpPostPG(vcGraph);
	ifModifiable = ProcScalar(preScaler, next, vcGraph); 
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
}

CTensor* SigmoidBackward::CmpPostPG(VecTsr* vcGraph)
{
	postPG = m_pTNode->QuadraPoly(-1., 1);
	vcGraph->push_back(postPG);
	return postPG;
}

//--------------------------------------

TanhBackward::TanhBackward()
{
	emBwType = EmBackwardType::TANH;
}

TanhBackward::~TanhBackward()
{
}

void TanhBackward::Backward(tyTensor* grad) {
	m_pTNode->AddGrad(grad);
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * *grad);
	next->ScaledIP(vrtScaler);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void TanhBackward::Backward(Scalar grad) {
	m_pTNode->AddGrad(grad);
	grad *= vrtScaler;
	auto its = m_mpNextFuns->begin();
	tyTensor* next = new tyTensor(*its->second * grad);
	its->first->GetGradFn()->Backward(next);
	FreeObj(next);
}

void TanhBackward::ShowThis() {
	cout << "grad_fn=<TanhBackward>" << endl;
}

void TanhBackward::BackwardInAuto(CTensor* preNode, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult, bool ifModifiable) {
	if (root == m_pTNode) {
		vcResult->push_back(preNode);
		vcGraph->pop_back();
		return;
	}
	CTensor* next, * dep;
	dep = (*m_mpNextFuns)[0].first;
	if (postPG)
	{
		next = *postPG * *preNode;
		vcGraph->push_back(next);
		dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
		return;
	}
	next = CmpPostPG(vcGraph);
	next = *postPG * *preNode;
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
}

void TanhBackward::BackwardInAuto(Scalar preScaler, CTensor* root, VecTsr* vcGraph, VecTsr* vcResult)
{
	if (IfAutogradEnd(preScaler, root, vcResult)) return;
	preScaler *= vrtScaler;
	CTensor* next, * dep;
	bool ifModifiable;
	dep = (*m_mpNextFuns)[0].first;
	next = m_pTNode->Pow(2., -preScaler, preScaler);
	vcGraph->push_back(next);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, true);
	return;
	if (postPG)
	{
		if (preScaler != 1.)
		{
			next = *postPG * preScaler;
			vcGraph->push_back(next);
			ifModifiable = true;
		}
		else
		{
			next = postPG;
			ifModifiable = false;
		}
		dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
		return;
	}
	next = CmpPostPG(vcGraph);
	ifModifiable = ProcScalar(preScaler, next, vcGraph);
	dep->GetGradFn()->BackwardInAuto(next, root, vcGraph, vcResult, ifModifiable);
}

CTensor* TanhBackward::CmpPostPG(VecTsr* vcGraph)
{
	postPG = m_pTNode->Pow(2., -1., 1.);
	postPG->ScaleIP(vrtScaler);
	vcGraph->push_back(postPG);
	return postPG;
}
