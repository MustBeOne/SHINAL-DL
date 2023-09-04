#include "COptimizerBase.h"
#include "CTensor.h"
#include "CLinearLayer.h"
#include "tools.h"
using namespace std;
//-----------------------------------------------
COptimizerSGD::COptimizerSGD(ItParam params, Scalar lr, Scalar momentum /*= 0*/, Scalar WeightDecay /*= 0*/, Scalar dampening /*= 0*/, bool nesterov /*= false*/, bool maximize /*= false*/, bool foreach /*= false */) {
	itParams = params;
	this->lr = lr;
	this->momentum = momentum;
	this->WeightDecay = WeightDecay;
	this->dampening = dampening;
	this->nesterov = nesterov;
	this->maximize = maximize;
	this->foreach = foreach;
}

COptimizerSGD::~COptimizerSGD() {

}

void COptimizerSGD::ZeroGrad() {
	auto its = itParams.first->begin();
	auto ite = itParams.first->end();
	for (auto it = its; it != ite; ++it) {
		(*it)->ZeroGrad();
	}
}

void COptimizerSGD::Step(/*FuncClosure closure,*/ CModuleBase* pMod) {
	tyTensor* g;
	tyTensor* gradient = nullptr;
	tyTensor* weight = nullptr;
	tyTensor* momen = nullptr;
	auto its = itParams.first->begin();
	auto ite = itParams.first->end();
	for (auto it = its; it != ite; ++it) {
		gradient = new tyTensor(*(*it)->GetGrad());
		if (WeightDecay) {
			weight = (*it)->GetData();
			tyTensor temp = WeightDecay * *weight;
			*gradient += temp;
		}
		if (momentum) {
			if (ifFirst) {
				ifFirst = false;
				momen = gradient;
			}
			else {
				tyTensor temp = (1. - dampening) * *gradient;
				*momen *= momentum;
				*momen += temp;
			}
			if (nesterov) {
				tyTensor temp = momentum * *momen;
				*gradient += temp;
			}
			else {
				*gradient = *momen;
			}
		}
		tyTensor temp1 = lr * *gradient;
		//temp1 = temp1 / times;
		g = (*it)->GetData();
		if (!maximize) {
			temp1 *= -1.;
		}
		*g += temp1;
		//cout<< *(*it)->GetData();
		//cout << *(*it)->GetGrad();
#ifdef show
		(*it)->ShowGrad();
#endif
		FreeObj(gradient);
		FreeObj(weight);
		FreeObj(momen);
		g = nullptr;
	}
}

//-----------------this is L-BFGS optimizer-------------------

COptimizerLBFGS::COptimizerLBFGS(
	ItParam params,
	Scalar lr /*= 1*/,
	int max_iter /*= 20*/,
	Scalar max_eval /*= 0*/,
	Scalar tolerance_grad /*= 1e-7*/,
	Scalar tolerance_change /*= 1e-9*/,
	int history_size /*= 100*/,
	emLineSearFn line_search_fn /*= nullptr */,
	struLBFGSCtrlParam* ctrlPtr /*= nullptr */
) {
	this->itParams = params;
	this->lr = lr;
	this->maxIter = max_iter;
	this->maxEval = max_eval;
	this->toleranceGrad = tolerance_grad;
	this->toleranceChange = tolerance_change;
	this->histSize = history_size;
	this->lineSrchFn = line_search_fn;
	this->ctrlPtr = (ctrlPtr == nullptr) ? &defParam : ctrlPtr;
	nParamDim = 0;
	//
	VecParam* vecPara = itParams.first;
	auto it = vecPara->begin();
	auto ite = vecPara->end();
	for (; it != ite; ++it) {
		nParamDim += (*it)->GetData()->GetSize();
	}
	vecPara = nullptr;
	state = new vector<void*>;
}

COptimizerLBFGS::~COptimizerLBFGS() {
	FreeObj(ctrlPtr);
	if (state == nullptr) {
		CallbackConverExit();
	}
}

void COptimizerLBFGS::ZeroGrad() {
	auto its = itParams.first->begin();
	auto ite = itParams.first->end();
	for (auto it = its; it != ite; ++it) {
		(*it)->ZeroGrad();
	}
}

void COptimizerLBFGS::Step(/*FuncClosure closure /*= nullptr, */CModuleBase* pMod /*= nullptr*/) {
	Scalar origLoss, loss, aprxHessian, oriStep, step, oriGradDotDesDir, yDots, iBeta;
	Scalar* alpha;
	VectData* flatGrad = nullptr, * preFlatGrad = nullptr;
	//the descend direction,middle data
	VectData* desDir = nullptr;
	//the temp value,pay attention to free their memory
	VectData* Sk, * Yk, * Qk, * Rk, * temp, * temp1;
	Qk = nullptr;
	vector<VectData*>* histYk = nullptr, * histSk = nullptr;
	vector<Scalar>* histRho = nullptr;
	int nIter, nLsIter, i, pastSize;
	//manage the calculate intermediate data
	state->push_back(flatGrad);
	state->push_back(preFlatGrad);
	state->push_back(desDir);
	state->push_back(histYk);
	state->push_back(histSk);
	state->push_back(histRho);
	/*
		initial the memory of the history parameters
	*/
	histYk = new vector<VectData*>;
	histSk = new vector<VectData*>;
	histRho = new vector<Scalar>;
	//
	if (pMod == nullptr) {
		/*err msg*/
		return;
	}
	//the first step: initial
	//origLoss = GetLossValue(closure, pMod);
	origLoss = GetLossValue(pMod);
	FlatGrad(flatGrad);
	//cout << "oriloss: " << origLoss << " faltgrad: " << flatGrad->block(0,0,100,1).transpose() << endl;
	if (!(flatGrad->array().abs().maxCoeff() > toleranceGrad)) {
		CallbackConverExit();
		return;
	}
	MatrixOp::MatrNega<VectData>(desDir, flatGrad);
	MatrixOp::MatrCopy<VectData>(preFlatGrad, flatGrad);
	aprxHessian = 1;
	oriStep = min(1., 1. / flatGrad->array().abs().sum()) * lr; 
	//cout << flatGrad->array().abs().sum() << " asdad";
	loss = origLoss;
	step = oriStep;
	//loop to optim
	nIter = 1;
	do {
		oriGradDotDesDir = MatrixOp::VectDot(flatGrad, desDir);
		if (oriGradDotDesDir > -toleranceChange) {
			//Exit with the convergence
			CallbackConverExit();
			return;
		}
		/*
			perform the line-search
		*/
		//LineSearchByStrongWolfe(step, desDir, loss, flatGrad, oriGradDotDesDir, nLsIter, closure, pMod);
		LineSearchByStrongWolfe(step, desDir, loss, preFlatGrad, oriGradDotDesDir, nLsIter, pMod);
		/*
			free the flatGrad memory and get the searched
		*/
		FlatGrad(flatGrad);
		//cout << "loss: " << loss << " faltgrad: " << flatGrad->block(0, 0, 100, 1).transpose() << endl;
		if (abs(origLoss - loss) < toleranceChange && loss < toleranceGrad) {
			//Exit with the convergence
			CallbackConverExit();
			return;
		}
		origLoss = loss;
		if (!(flatGrad->array().abs().maxCoeff() > toleranceGrad)) {
			//Exit with the convergence
			CallbackConverExit();
			return;
		}
		if (!(desDir->array().abs().maxCoeff() > toleranceChange / step)) {
			//Exit with the convergence
			CallbackConverExit();
			return;
		}
		/*
			Store the Y_k and S_k and Rho_k history value
			and judge if the size of history is out of history_size
		*/
		Yk = MatrixOp::SubMatr<VectData>(flatGrad, preFlatGrad);
		Sk = MatrixOp::MulScalarMatr<VectData>(desDir, step);
		//cout << "Yk " << Yk->block(0, 0, 100, 1).transpose() << " Sk " << Sk->block(0, 0, 100, 1).transpose() << endl;
		yDots = MatrixOp::VectDot(Yk, Sk);
		if (yDots > 1.0e-10) {
			if (histYk->size() == histSize) {
				FreeObj((*histYk)[0]);
				histYk->erase(histYk->begin());
				FreeObj((*histSk)[0]);
				histSk->erase(histSk->begin());
				histRho->erase(histRho->begin());
			}
			histYk->push_back(Yk);
			histSk->push_back(Sk);
			histRho->push_back(1. / yDots);
			aprxHessian = yDots / MatrixOp::VectDot(Yk, Yk);
			//cout << "aprxHessian " << aprxHessian;
		}
		/*
			the Two-Loop to calculate the new descend direction
		*/
		MatrixOp::MatrNega<VectData>(Qk, flatGrad);
		pastSize = static_cast<int>(histYk->size());
		alpha = new Scalar[pastSize];
		//the first loop
		for (i = pastSize - 1; i > -1; --i) {
			alpha[i] = (*histRho)[i] * MatrixOp::VectDot((*histSk)[i], Qk);
			/*
			temp = Qk;
			temp1 = MatrixOp::MulScalarMatr((*histYk)[i], alpha[i]);
			Qk = MatrixOp::SubMatr(Qk, temp1);*/
			MatrixOp::MatrAdd(Qk, (*histYk)[i], -alpha[i]);
		}
		//the second loop,point the Rk to desDir.desDir is calculated in the second loop
		FreeObj(desDir);
		desDir = MatrixOp::MulScalarMatr<VectData>(Qk, aprxHessian);
		Rk = desDir;
		for (i = 0; i < pastSize; ++i) {/*
			iBeta = MatrixOp::VectDot((*histYk)[i], Rk);
			iBeta *= (*histRho)[i];
			MatrixOp::MatrAdd<VectData>(Rk, (*histSk)[i], alpha[i] - iBeta);*/
			iBeta = MatrixOp::VectDot((*histYk)[i], desDir);
			iBeta *= (*histRho)[i];
			MatrixOp::MatrAdd<VectData>(desDir, (*histSk)[i], alpha[i] - iBeta);
		}
		//cout << " desDir " << desDir->block(0, 0, 100, 1).transpose() << endl;
		Rk = nullptr;
		//end loop,free the temp value
		FreeObj(alpha);
		FreeObj(Qk);
		//ready for the next loop
		MatrixOp::MatrCopy(preFlatGrad, flatGrad);
		step = lr;
	} while (nIter++ < maxIter);
	CallbackConverExit();
}

/// <summary>
/// flat all grad of the parameters,to the n*1 matrix.Return by the ref of PtrTen
/// </summary>
/// <param name="flatedGrad"></param>
void COptimizerLBFGS::FlatGrad(VectData*& flatedGrad) {
	VecParam* vecPara = itParams.first;
	auto it = vecPara->begin();
	auto ite = vecPara->end();
	FreeObj(flatedGrad);
	flatedGrad = new VectData(nParamDim);
	it = vecPara->begin();
	int pos = 0, leng;
	for (; it != ite; ++it) {
		leng = (*it)->GetGrad()->GetSize();
		flatedGrad->block(pos, 0, leng, 1) << (*it)->GetGrad()->GetDataPtr()->reshaped<RowMajor>();
		pos += leng;
	}
}

Scalar COptimizerLBFGS::GetLossValue(/*FuncClosure closure,*/ CModuleBase* pMod) {
	PtrTen loss = pMod->Closure();
	Scalar lossVal = loss->GetData()->GetValue(0, 0);
	FreeObj(loss);
	return lossVal;
}

/// <summary>
/// the line-search in pytorch is based on the strong wolfe condition,
/// but there are some shortages which are optimized in this function
/// reseting the parameters is performded in the line-search in pytorch,however ,
/// it just be performed once in this function when meet the exiting condition
/// </summary>
/// <param name="oriStep">the input step,return by the ref with the new minval step</param>
/// <param name="desDir">the descend direction which is vector-data</param>
/// <param name="oriloss">the old-lossVal,return by the ref with the new lossVal in the new-params</param>
/// <param name="flatGrad">the flated grad in the $X_k$ params</param>
/// <param name="oriGradDotDesDir">the old-gtd</param>
/// <param name="nIter">the iter-times the evaluate-func be performed</param>
/// <param name="closure">the closure function</param>
void COptimizerLBFGS::LineSearchByStrongWolfe(
	Scalar& oriStep,
	VectData* desDir,
	Scalar& oriloss,
	VectData* flatGrad,
	Scalar	      oriGradDotDesDir,
	int& nIter,
	/*FuncClosure   closure,*/
	CModuleBase* pMod
)
{
	pair<Scalar, Scalar> bound;
	Scalar inc = 2.1, des = 0.5, procFac;
	Scalar c1 = ctrlPtr->wolfeC1;
	Scalar c2 = ctrlPtr->wolfeC2;
	Scalar minStep, maxStep, temp;
	Scalar prestep = 0, preloss = oriloss, preGradDotDesDir = oriGradDotDesDir;
	Scalar newstep = oriStep, newLoss, newGradDotDesDir;
	Scalar bracket[2], bracketF[2], bracketGTD[2];
	bool done = false, insufProgress = false;
	nIter = 0;
	VectData* newGrad;
	auto xInit = SavePreParam(); 
	newGrad = MatrixOp::MatrCopy(flatGrad);
	while (true && nIter <= 25) {
		++nIter;
		/*
			Evaluate in the descend direction,and the parameters of the network have been modified
			in this function,the newGrad is returned by the pointer and
			it point to gradient in the $X_{k+1}$
		*/
		newLoss = EvalInDesDir(/*closure, */pMod, newstep, desDir, newGrad);
		/*
			calculate the GradF($X_{k+1}$) ./dot $d_{k}$
		*/
		newGradDotDesDir = MatrixOp::VectDot(newGrad, desDir);

		//judge if meet the Armijo condition
		if (newLoss > oriloss + c1 * newstep * oriGradDotDesDir || (nIter > 1 && newLoss >= oriloss)) {
			//do not meet the Armijo condition
			bracket[0] = prestep, bracket[1] = newstep;
			bracketF[0] = preloss, bracketF[1] = newLoss;
			bracketGTD[0] = preGradDotDesDir, bracketGTD[1] = newGradDotDesDir;
			procFac = des;
			ResetParams(xInit);
			break;
		}
		else {
			//meet the Armijo condition
			if (ctrlPtr->backtrack == emBackTracking::LineSearchArmijo) {
				//exit with the Armijo Condition
				oriloss = newLoss;
				oriStep = newstep;
				ReleasePreParamMemo(xInit);
				return;
			}
			//judge if meet the weak Wolfe condition
			if (abs(newGradDotDesDir) <= -c2 * oriGradDotDesDir) {
				done = true;
				oriloss = newLoss;
				oriStep = newstep;
				ReleasePreParamMemo(xInit);
				cout << "ret1 step " << newstep << "  ";
				return;
			}
			else {
				if (newGradDotDesDir >= 0)
				{
					bracket[0] = prestep, bracket[1] = newstep;
					bracketF[0] = preloss, bracketF[1] = newLoss;
					bracketGTD[0] = preGradDotDesDir, bracketGTD[1] = newGradDotDesDir;
					ResetParams(xInit);
					break;
				}
			}

		}
		minStep = newstep + 0.01 * (newstep - prestep);
		maxStep = newstep * 10;
		temp = newstep;
		bound = make_pair(minStep, maxStep);
		newstep = CubicInterpolate(
			prestep, preloss, preGradDotDesDir,
			newstep, newLoss, newGradDotDesDir,
			&bound
		);
		cout << newstep << "new" << endl;
		prestep = temp;
		preloss = newLoss;
		preGradDotDesDir = newGradDotDesDir;
	}
	int	low_pos, high_pos;
	Scalar dNorm = desDir->array().abs().maxCoeff(), eps, mx, mi;
	bracketF[0] <= bracketF[1] ? low_pos = 0, high_pos = 1 : low_pos = 1, high_pos = 0;
	while (!done && nIter <= 25)
	{
		++nIter;
		if (abs(bracket[1] - bracket[0]) * dNorm < toleranceChange)
		{
			oriloss = EvalInDesDir(pMod, newstep, desDir, newGrad);
			oriStep = newstep;
			ReleasePreParamMemo(xInit);
			break;
		}
		newstep = CubicInterpolate(
			bracket[0], bracketF[0], bracketGTD[0],
			bracket[1], bracketF[1], bracketGTD[1]);
		cout << newstep << '\n';
		mx = max(bracket[0], bracket[1]), mi = min(bracket[0], bracket[1]);
		eps = 0.1 * (mx - mi);
		if (min(mx - newstep, newstep - mi) < eps)
		{
			if (insufProgress || newstep >= mx || newstep <= mi)
			{
				if (abs(newstep - mx) < abs(newstep - mi)) 
					newstep = mx - eps;
				else 
					newstep = mi + eps;
				insufProgress = false;
			}
			else insufProgress = true;
		}
		else insufProgress = false;
		newLoss = EvalInDesDir(pMod, newstep, desDir, newGrad);
		newGradDotDesDir = MatrixOp::VectDot(newGrad, desDir);

		if (newLoss > oriloss + c1 * newstep * oriGradDotDesDir || newLoss >= bracketF[low_pos])
		{
			bracket[high_pos] = newstep;
			bracketF[high_pos] = newLoss;
			bracketGTD[high_pos] = newGradDotDesDir;
			bracketF[0] <= bracketF[1] ? low_pos = 0, high_pos = 1 : low_pos = 1, high_pos = 0;
			if (nIter <= 25)ResetParams(xInit);
		}
		else {
			if (abs(newGradDotDesDir) <= -c2 * oriGradDotDesDir) 
			{
				ReleasePreParamMemo(xInit);
				oriloss = newLoss;
				oriStep = newstep;
				cout << "ret2 step " << newstep << "  ";
				return;
			}
			else if (newGradDotDesDir * (bracket[high_pos] - bracket[low_pos]) >= 0)
			{ 
				bracket[high_pos] = bracket[low_pos];
				bracketF[high_pos] = bracketF[low_pos];
				bracketGTD[high_pos] = bracketGTD[low_pos];
			}
			if (nIter <= 25)ResetParams(xInit);
			bracket[low_pos] = newstep;
			bracketF[low_pos] = newLoss;
			bracketGTD[low_pos] = newGradDotDesDir;
		}
		if (nIter > 25)
		{
			oriloss = bracketF[low_pos];
			oriStep = bracket[low_pos];/*
			oriloss = EvalInDesDir(pMod, newstep, desDir, newGrad);
			oriStep = newstep;*/
			ReleasePreParamMemo(xInit);
			cout << "ret3 step " << newstep << "  ";
			return;
		}
	}
	return;
}
//
//void COptimizerLBFGS::LineSearchByStrongWolfe(
//	Scalar& oriStep,
//	VectData* desDir,
//	Scalar& oriloss,
//	VectData* flatGrad,
//	Scalar	      oriGradDotDesDir,
//	int& nIter,
//	/*FuncClosure   closure,*/
//	CModuleBase* pMod
//) {
//	Scalar inc = 1.6, des = 0.8, procFac;
//	Scalar c1 = ctrlPtr->wolfeC1;
//	Scalar c2 = ctrlPtr->wolfeC2;
//	Scalar step = oriStep;
//	Scalar newLoss, newGradDotDesDir;
//	nIter = 0;
//	VectData* newGrad;
//	auto xInit = SavePreParam();
//	newGrad = MatrixOp::MatrCopy(flatGrad);
//	Scalar temp;
//	while (true) {
//		++nIter;
//		/*
//			Evaluate in the descend direction,and the parameters of the network have been modified
//			in this function,the newGrad is returned by the pointer and
//			it point to gradient in the $X_{k+1}$
//		*/
//		newLoss = EvalInDesDir(/*closure, */pMod, step, desDir, newGrad);
//		/*
//			calculate the GradF($X_{k+1}$) ./dot $d_{k}$
//		*/
//		newGradDotDesDir = MatrixOp::VectDot(newGrad, desDir);
//		//judge if meet the Armijo condition
//		if (newLoss > oriloss + c1 * step * oriGradDotDesDir) {
//			//do not meet the Armijo condition
//			procFac = des;
//			ResetParams(xInit);
//		}
//		else {
//			//meet the Armijo condition
//			if (ctrlPtr->backtrack == emBackTracking::LineSearchArmijo) {
//				//exit with the Armijo Condition
//				oriloss = newLoss;
//				oriStep = step;
//				return;
//			}
//			//judge if meet the weak Wolfe condition
//			if (newGradDotDesDir < c2 * oriGradDotDesDir) {
//				//do not meet the weak Wolfe condition
//				procFac = inc;
//				ResetParams(xInit);
//			}
//			else {
//				// meet the weak Wolfe condition
//				if (ctrlPtr->backtrack == emBackTracking::LineSearchWeakWolfe) {
//					//exit with weak Wolfe Condition
//					oriloss = newLoss;
//					oriStep = step;
//					return;
//				}
//				//judge if meet the strong Wolfe condition
//				if (-newGradDotDesDir < c2 * oriGradDotDesDir) {
//					//do not meet the strong Wolfe condition
//					procFac = des;
//					ResetParams(xInit);
//				}
//				else {
//					//meet the strong Wolfe condition
//					//exit with the strong Wolfe Condition
//					oriloss = newLoss;
//					oriStep = step;
//					return;
//				}
//			}
//
//		}
//		step *= procFac;
//	}
//}

Scalar COptimizerLBFGS::EvalInDesDir(/*FuncClosure closure,*/ CModuleBase* pMod, Scalar step, VectData* desDir, VectData*& grad) {
	Scalar loss;
	AddParams(step, desDir);
	loss = GetLossValue(/*closure,*/ pMod);
	FlatGrad(grad);
	return loss;
}

void COptimizerLBFGS::AddParams(Scalar coe, VectData* delta) {
	VecParam* vecPara = itParams.first;
	MatSize size;
	int pos = 0, leng;
	auto it = vecPara->begin();
	auto ite = vecPara->end();
	MatrData* temp;
	for (; it != ite; ++it) {
		size = (*it)->GetData()->GetShape();
		leng = (*it)->GetData()->GetSize();
		temp = new MatrData(delta->block(pos, 0, leng, 1).reshaped<RowMajor>(size.first, size.second));
		MatrixOp::MatrAdd((*it)->GetData()->GetDataPtr(), temp, coe);
		pos += leng;
		FreeObj(temp);
	}
	vecPara = nullptr;
}

void COptimizerLBFGS::ResetParams(VecParamLBFGS* init) {
	auto it = init->begin();
	auto ite = init->end();
	auto itp = itParams.first->begin();
	auto itep = itParams.first->end();
	int i = 0;
	CTensor* temp;
	for (; it != ite, itp != itep; ++it, ++itp) 
	{
		(*itp)->SetData(it->first);
		(*itp)->SetGrad(it->second);
	}
}

void COptimizerLBFGS::CallbackConverExit() {
	/*
		the first 3 are VectData*
		the next  2 are vector<VectData*>*
		the last  1 is  vector<Scalar*>*
	*/
	int i;
	for (i = 0; i < 3; ++i) {
		FreeObj((VectData*)((*state)[i]));
	}
	for (; i < 5; ++i) {
		ClearObjVec((vector<VectData*>*)((*state)[i]));
	}
	/*((vector<Scalar>*)((*state)[5]))->clear();
	delete ((vector<Scalar>*)((*state)[5]));
	(*state)[5] = nullptr;*/
	state->clear();
	delete state;
	state = nullptr;
}

VecParamLBFGS* COptimizerLBFGS::SavePreParam()
{
	tyTensor* data, * grad;
	VecParamLBFGS* prePara = new VecParamLBFGS;
	for (auto it : *itParams.first)
	{
		data = new tyTensor(*it->GetData());
		grad = new tyTensor(*it->GetGrad());
		prePara->push_back(make_pair(data, grad));
	}
	return prePara;
}

void COptimizerLBFGS::ReleasePreParamMemo(VecParamLBFGS* para)
{
	for (auto it : *para)
	{
		FreeObj(it.first);
		FreeObj(it.second);
	}
}


Scalar COptimizerLBFGS::CubicInterpolate(
	Scalar x1, 
	Scalar f1, 
	Scalar g1, 
	Scalar x2,
	Scalar f2, 
	Scalar g2, 
	pair<Scalar, Scalar>* bound /*= nullptr*/)
{
	Scalar xmin_bound, xmax_bound, d1, d2_square, d2, min_pos;
	if (bound)
	{
		xmin_bound = bound->first;
		xmax_bound = bound->second;
	}
	else
	{
		xmin_bound = x1 <= x2 ? x1 : x2;
		xmax_bound = x1 <= x2 ? x2 : x1;
	}
	d1 = g1 + g2 - 3. * (f1 - f2) / (x1 - x2);
	d2_square = static_cast<Scalar>(powf(d1, 2.f)) - g1 * g2;
	if (d2_square >= 0)
	{
		d2 = sqrt(d2_square);
		if (x1 <= x2)
		{
			min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2));
		}
		else {
			min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2));
		}
		return min(max(min_pos, xmin_bound), xmax_bound);
	}
	else 
	{
		return (xmin_bound + xmax_bound) / 2.;
	}
}
