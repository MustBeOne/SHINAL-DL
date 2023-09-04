#pragma once
#include "CFunction.h"

///-----------------Loss Function---------------------
class CMSELF :public CLossFunction {
public:
	CMSELF() {};
	virtual ~CMSELF() { FreeObj(tempLoss); };

	virtual PtrTen CmpLoss(PtrTen, PtrTen);
};
