#pragma once
//the NN Module Base Class-----
#include "CModuleBase.h"
//|_"CFunction.h"
#include "CFunction.h"
//   |
//   |_"CLossFunction.h"
//   |_"CNonlinActiFunction.h"
#include "CLossFunction.h"
#include "CNonlinActiFunction.h"
//|
//|_"CLinearLayer.h"
#include "CLinearLayer.h"

//the Optimizer Class
#include "COptimizerBase.h"
//the learning Parameter Manage Class
#include "CParameter.h"
//the Module Container Manage Class
#include "CSequential.h"

#include "InitBase.h"