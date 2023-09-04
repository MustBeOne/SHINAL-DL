#include "Forward.h"
#include "CTensor.h"

static CTensor* Const1 = new CTensor(1);

CTensor* ForwardBase::GetCurNode() {
	return pNode;
}

CTensor* ForwardBase::GetNxtNode() {
	return nextNode;
}


void ForwardBase::FreeForward() {
	pNode = nullptr;
	nextNode = nullptr;
}


NoneForward::NoneForward() {
	pNode = nullptr;
}

NoneForward::~NoneForward() {
	pNode = nullptr;
}

void NoneForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	res->push_back(node);
}

//--------------------------------------------------------------------------

AddForward::AddForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}

AddForward::~AddForward() {
	FreeForward();
}

void AddForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	if (!node) {
		node = Const1;
	}
	//nextNode->Forward(node, res, graph);
}

//--------------------------------------------------------------------------

SubForward::SubForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	ifLeft = false;
	nextNode = nnode;
}

SubForward::~SubForward() {
	FreeForward();
}

void SubForward::IfLeft() {
	this->ifLeft = true;
}

void SubForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	if (!node) {
		node = Const1;
	}
	if (!ifLeft) {
		node->NegData();
	}
	//nextNode->Forward(node, res, graph);
}

//--------------------------------------------------------------------------

MulForward::MulForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}

MulForward::~MulForward() {
	FreeForward();
	anotherNode = nullptr;
}

void MulForward::SetAnotherNode(CTensor* ano) {
	anotherNode = ano;
}

void MulForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	CTensor* cpynode = anotherNode->Copy();
	cpynode->GetData()->ReshapeIn(pNode->GetData()->GetShape());
	CTensor* result = cpynode;
	graph->push_back(cpynode);
	if (node) {
		result = (*cpynode) * (*node);
		graph->push_back(result);
	}
	//nextNode->Forward(result, res, graph);
}

//--------------------------------------------------------------------------

DivForward::DivForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	isLeft = false;
	nextNode = nnode;
	anotherNode = nullptr;
}

DivForward::~DivForward() {
	FreeForward();
	anotherNode = nullptr;
}

void DivForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	CTensor* next;
	if (isLeft) {
		if (node) {
			next = (*node) / (*anotherNode);
		}
		else {
			next = (*Const1) / (*anotherNode);
		}
		graph->push_back(next);
	}
	else {
		denominator = pNode->Pow(-2, -1);
		graph->push_back(denominator);
		next = (*denominator) * (*anotherNode);
		graph->push_back(next);
		if (node) {
			next = (*next) * (*node);
			graph->push_back(next);
		}
	}
	//nextNode->Forward(next, res, graph);
}

void DivForward::SetAnotherNode(CTensor* ano) {
	anotherNode = ano;
}

void DivForward::IsLeft() {
	isLeft = true;
}



//--------------------------------------------------------------------------

CpyForward::CpyForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}

CpyForward::~CpyForward() {
	FreeForward();
}

void CpyForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	if (!node) {
		MatSize shape = pNode->GetData()->GetShape();
		node = new CTensor(shape.first, shape.second, 1.);
	}
	//nextNode->Forward(node, res, graph);
}

//--------------------------------------------------------------------------

PowForward::PowForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}

PowForward::~PowForward() {
	FreeForward();
}

void PowForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {
	CTensor* next;
	next = pNode->Pow(powCoe - 1, sign * powCoe);
	graph->push_back(next);
	if (node) {
		next = (*next) * (*node);
		graph->push_back(next);
	}
	//nextNode->Forward(next, res, graph);
}

void PowForward::SetFactor(Scalar sign, Scalar powCoe) {
	this->sign = sign;
	this->powCoe = powCoe;
}

//--------------------------------------------------------------------------

MmForward::MmForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}

MmForward::~MmForward() {
	FreeForward();
}

void MmForward::Forward(CTensor* node, VecTsr* res, VecTsr* graph) {

}

void MmForward::SetAnotherNode(CTensor* ano) {
	anotherNode = ano;
}

void MmForward::IsLeft() {
	isLeft = true;
}


//--------------------------------------------------------------------------

ExpForward::ExpForward(CTensor* node, CTensor* nnode) {
	pNode = node;
	nextNode = nnode;
}
