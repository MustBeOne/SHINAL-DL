#include "CLossFunction.h"
#include "tools.h"
#include "CTensor.h"
#include "Backward.h"

///-----------------Loss Function---------------------


PtrTen CMSELF::CmpLoss(PtrTen input, PtrTen target) {
	MatrData* sub = new MatrData(input->GetData()->GetData() - target->GetData()->GetData());
	MatrData* loss = new MatrData(sub->array().pow(2));
	tempLoss = new CTensor(loss->mean(), true);
	//no leaf
	tempLoss->SetLeafSta(false);
	BackwardBase* bw = new MseLossBackward();
	*sub *= (2. / Scalar(sub->size()));
	tyTensor* raw = new tyTensor(*sub);
	bw->SaveNextFuns(input, raw);
	bw->SaveNode(tempLoss);
	tempLoss->SetGradFn(bw);
	FreeObj(sub);
	FreeObj(loss);
	raw = nullptr;
	return tempLoss;
}
