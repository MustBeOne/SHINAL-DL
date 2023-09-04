#include "struMat.h"
#include "CRandom.h"


using namespace std;
struMat::struMat() {
	mat = new MatrData(0, 0);
	tyshape = emShape::NONE;
	row = 0;
	col = 0;
	size = 0;
}

struMat::struMat(Scalar data) {
	mat = new MatrData(1, 1);
	*mat << data;
	tyshape = emShape::ONE;
	row = 1;
	col = 1;
	size = 1;
}

emShape CalShape(int row, int col) {
	if (row * col == 1) {
		return emShape::ONE;
	}
	else if (col == 1) {
		return emShape::COL;
	}
	else if (row == 1) {
		return emShape::ROW;
	}
	return emShape::FULL;
}
struMat::struMat(int row, int col, Scalar data) {
	mat = new MatrData(row, col);
	mat->setConstant(row, col, data);
	tyshape = CalShape(row, col);
	this->row = row;
	this->col = col;
	this->size = row * col;
}

struMat::struMat(MatrData* data, bool ifCopy) {
	this->row = data->rows();
	this->col = data->cols();
	this->size = data->size();
	if (ifCopy)
	{
		mat = new MatrData(*data);
	}
	else
	{
		mat = data;
	}
	tyshape = CalShape(row, col);
}

struMat::struMat(MatrData& data) {
	this->row = data.rows();
	this->col = data.cols();
	this->size = data.size();
	mat = new MatrData(data);
	tyshape = CalShape(row, col);
}

struMat::struMat(MatrData&& data) {
	this->row = data.rows();
	this->col = data.cols();
	this->size = data.size();
	mat = new MatrData(data);
	tyshape = CalShape(row, col);
}

struMat::struMat(struMat* data) {
	MatSize ms = data->GetShape();
	row = ms.first;
	col = ms.second;
	size = row * col;
	mat = new MatrData(data->GetData());
	tyshape = data->GetShapetype();
}

struMat::struMat(struMat& data) {
	MatSize ms = data.GetShape();
	row = ms.first;
	col = ms.second;
	size = row * col;
	mat = new MatrData(data.GetData());
	tyshape = data.GetShapetype();
}

struMat::struMat(struMat&& data) {
	MatSize ms = data.GetShape();
	row = ms.first;
	col = ms.second;
	size = row * col;
	mat = new MatrData(data.GetData());
	tyshape = data.GetShapetype();
}

struMat::struMat(MatSize size, Scalar data) {
	row = size.first;
	col = size.second;
	this->size = row * col;
	mat = new MatrData(row, col);
	mat->setConstant(data);
	tyshape = CalShape(row, col);
}

struMat::~struMat() {
	FreeObj(mat);
}

void struMat::BroadcastIn(MatSize size) {
	row = size.first;
	col = size.second;
	Scalar temp;
	this->size = row * col;
	switch (tyshape) {
	case FULL:return;
		break;
	case NONE:PRINTERRORINFO("Can't Broadcast!"); return;
		break;
	case ROW:if (row <= 1 || col != mat->cols()) return;
		*mat = mat->row(0).replicate(row, 1);
		tyshape = FULL;
		break;
	case COL:if (col <= 1 || row != mat->rows()) return;
		*mat = mat->col(0).replicate(1, col);
		tyshape = FULL;
		break;
	case ONE:if (row <= 1 && col <= 1)
		return;
		temp = (*mat)(0, 0);
		mat->resize(row, col);
		mat->setConstant(temp);
		if (row > 1) {
			tyshape = COL;
			if (col > 1) {
				tyshape = FULL;
				break;
			}
		}
		if (col > 1) {
			tyshape = ROW;
		}
		break;
	default:
		break;
	}
	return;
}

void struMat::CompressIn(MatSize size) {
	row = size.first;
	col = size.second;
	this->size = row * col;
	Scalar t;
	MatrData temp;
	if (row != 1 && col != 1) {
		PRINTERRORINFO("Can't Broadcast!");
		return;
	}
	switch (tyshape) {
	case FULL:
		if (row * col == 1) {
			t = mat->sum();
			mat->resize(1, 1);
			(*mat)(0, 0) = t;
			tyshape = ONE;
			return;
		}
		else if (row == 1) {
			temp = mat->colwise().sum();
			FreeObj(mat);
			mat = new MatrData(temp);
			tyshape = ROW;
			return;
		}
		else if (col == 1) {
			temp = mat->rowwise().sum();
			FreeObj(mat);
			mat = new MatrData(temp);
			tyshape = COL;
			return;
		}
		break;
	case NONE:PRINTERRORINFO("Can't Compress!"); return;
		break;
	case ROW:
	case COL:if (col * row != 1) {
		PRINTERRORINFO("Can't Compress!");
		return;
	}
			t = mat->sum();
			mat->resize(1, 1);
			(*mat)(0, 0) = t;
			tyshape = ONE;
			break;
	case ONE:return;
		break;
	default:
		break;
	}
	return;
}

void struMat::SetData(MatrData* data) {
	FreeObj(mat);
	mat = new MatrData(*data);
}

MatrData struMat::GetData() {
	return *mat;
}

MatrData* struMat::GetDataPtr() {
	return mat;
}

MatSize struMat::GetShape() {
	int row = mat->rows();
	int col = mat->cols();
	MatSize shape = make_pair(row, col);
	return shape;
}

int struMat::GetSize() {
	return this->mat->size();
}

Scalar struMat::GetValue(int row, int col) {
	return (*mat)(row, col);
}

bool struMat::ifOneDim() {
	return mat->cols() == 1 || mat->rows() == 1;
}

emShape struMat::GetShapetype() {
	return tyshape;
}

void struMat::ReshapeIn(MatSize size) {
	int row, col;
	row = size.first;
	col = size.second;
	if (mat->size() > row * col) {
		CompressIn(size);
	}
	else {
		BroadcastIn(size);
	}
}

void struMat::NegIP() {
	*mat = -*mat;
}

struMat* struMat::Neg()
{
	struMat* res = new struMat(this);
	res->NegIP();
	return res;
}

struMat struMat::Transpose() {
	return struMat(mat->transpose());
}

struMat* struMat::TransposePtr() {
	struMat* t = new struMat(mat->transpose());
	return t;
}

void struMat::TransposeIP() {
	mat->transposeInPlace();
	swap(row, col);
	switch (tyshape) {
	case ROW:tyshape = COL;
		break;
	case COL:tyshape = ROW;
		break;
	default:
		break;
	}
}

struMat* struMat::Scaled(Scalar t)
{
	struMat* res = new struMat(t * *mat);
	return res;
}

void struMat::ScaledIP(Scalar t)
{
	*mat = *mat * t;
}

void struMat::AddScalarIP(Scalar t)
{
	mat->array() += t;
}

struMat* struMat::Dot(struMat* data) {
	return new struMat(*this->mat * data->GetData());
}

struMat* struMat::Pow(Scalar coePow /*= 1*/, Scalar sign /*= 1*/) {
	struMat* mat = new struMat(this->mat->array().pow(coePow) * sign);
	return mat;
}

struMat* struMat::ExpEach(Scalar coeExp, Scalar sign) {
	struMat* mat = new struMat((this->mat->array() * coeExp).exp() * sign);
	return mat;
}

Scalar struMat::Sum() {
	return mat->sum();
}


tyTensor* struMat::Cat(tyTensor* left, tyTensor* right, int dim) {
	if (dim != 0 && dim != 1) {
		cout << "The Concatenation Can Not Be Implemented For Two 2-D Matrixs!" << endl;
		return nullptr;
	}
	MatrData* res = nullptr;
	tyTensor* result = nullptr;
	MatSize lSize = left->GetShape();
	MatSize rSize = right->GetShape();
	int matchCoe1, matchCoe2;
	int cat1, cat2;
	if (!dim) {
		matchCoe1 = lSize.second;
		matchCoe2 = rSize.second;
		if (matchCoe1 != matchCoe2) {
			cout << "Sizes of tensors must match except in dimension " << dim << ".Got " << matchCoe1 << " and " << matchCoe2 << " in dimension " << !dim << "!" << endl;
			return nullptr;
		}
		cat2 = rSize.first;
		cat1 = lSize.first;
		res = new MatrData(cat1 + cat2, matchCoe1);
		*res << *left->GetDataPtr(), * right->GetDataPtr();
	}
	else {
		matchCoe1 = lSize.first;
		matchCoe2 = rSize.first;
		if (matchCoe1 != matchCoe2) {
			cout << "Sizes of tensors must match except in dimension " << dim << ".Got " << matchCoe1 << " and " << matchCoe2 << " in dimension " << !dim << "!" << endl;
			return nullptr;
		}
		cat2 = rSize.second;
		cat1 = lSize.second;
		res = new MatrData(matchCoe1, cat1 + cat2);
		*res << *left->GetDataPtr(), * right->GetDataPtr();
	}
	result = new tyTensor(res);
	FreeObj(res);
	return result;
}

tyTensor* struMat::AddMM(Scalar beta, tyTensor* mat, Scalar alpha, tyTensor* mat1, tyTensor* mat2)
{
	tyTensor* data;
	tyTensor* temp1, temp2;
	temp1 = mat1->Dot(mat2);
	temp1->ScaledIP(alpha);
	temp2 = beta * *mat;
	data = new tyTensor(*temp1 + temp2);
	FreeObj(temp1);
	return data;
}

struMat operator+(struMat& lmat, struMat& rmat) {
	emShape rsha = rmat.GetShapetype();
	emShape lsha = lmat.GetShapetype();
	MatrData rdata = rmat.GetData();
	MatrData ldata = lmat.GetData();
	if (rsha == emShape::NONE || lsha == emShape::NONE) {
		PRINTERRORINFO("Dim error!");
		return struMat();
	}
	char mark = rsha - lsha;
	switch (mark) {
	case 0:return struMat(ldata + rdata); break;
	case 2:return struMat(rdata.rowwise() + ldata.row(0)); break;
	case 3:return struMat(rdata.colwise() + ldata.col(0)); break;
	case 5:return struMat(MatrData::Constant(rdata.rows(), rdata.cols(), ldata(0, 0)) + rdata); break;
	case -2:return struMat(ldata.rowwise() + rdata.row(0)); break;
	case -3:return struMat(ldata.colwise() + rdata.col(0)); break;
	case -5:return struMat(MatrData::Constant(ldata.rows(), ldata.cols(), rdata(0, 0)) + ldata); break;
	default:PRINTERRORINFO("Dim error!"); return struMat();
		break;
	}
}
struMat operator+(Scalar ldata, struMat& rmat) {
	MatrData rdata = rmat.GetData();
	return struMat(MatrData::Constant(rdata.rows(), rdata.cols(), ldata) + rdata);
}
struMat operator+(struMat& lmat, Scalar rdata) {
	MatrData ldata = lmat.GetData();
	return struMat(MatrData::Constant(ldata.rows(), ldata.cols(), rdata) + ldata);
}
struMat operator-(struMat& lmat, struMat& rmat) {
	MatrData rdata = rmat.GetData();
	struMat newmat(-rdata);
	return lmat + newmat;
}
struMat operator-(Scalar ldata, struMat& rmat) {
	MatrData rdata = rmat.GetData();
	return struMat(MatrData::Constant(rdata.rows(), rdata.cols(), ldata) - rdata);
}
struMat operator-(struMat& lmat, Scalar rdata) {
	MatrData ldata = lmat.GetData();
	return struMat(ldata - MatrData::Constant(ldata.rows(), ldata.cols(), rdata));
}
struMat operator*(struMat& lmat, struMat& rmat) {
	emShape rsha = rmat.GetShapetype();
	emShape lsha = lmat.GetShapetype();
	ArayData rdata = rmat.GetData();
	ArayData ldata = lmat.GetData();
	if (rsha == emShape::NONE || lsha == emShape::NONE) {
		PRINTERRORINFO("Dim error!");
		return struMat();
	}
	char mark = rsha - lsha;
	switch (mark) {
	case 0:return struMat(ldata * rdata); break;
	case 2:return struMat(rdata.rowwise() * ldata.row(0)); break;
	case 3:return struMat(rdata.colwise() * ldata.col(0)); break;
	case 5:return struMat(ArayData::Constant(rdata.rows(), rdata.cols(), ldata(0, 0)) * rdata); break;
	case -2:return struMat(ldata.rowwise() * rdata.row(0)); break;
	case -3:return struMat(ldata.colwise() * rdata.col(0)); break;
	case -5:return struMat(ArayData::Constant(ldata.rows(), ldata.cols(), rdata(0, 0)) * ldata); break;
	default:PRINTERRORINFO("Dim error!"); return struMat();
		break;
	}
}
struMat operator*(Scalar ldata, struMat& rmat) {
	ArayData rdata = rmat.GetData();
	return struMat(ArayData::Constant(rdata.rows(), rdata.cols(), ldata) * rdata);
}
struMat operator*(struMat& lmat, Scalar rdata) {
	ArayData ldata = lmat.GetData();
	return struMat(ArayData::Constant(ldata.rows(), ldata.cols(), rdata) * ldata);
}
struMat operator/(Scalar ldata, struMat& rmat) {
	ArayData rdata = rmat.GetData();
	if (!(rdata != 0).all()) {
		PRINTERRORINFO("Divider has 0 error!");
		return struMat();
	}
	return struMat(ArayData::Constant(rdata.rows(), rdata.cols(), ldata) / rdata);
}
struMat operator/(struMat& lmat, struMat& rmat) {
	ArayData rdata = rmat.GetData();
	if (!(rdata != 0).all()) {
		PRINTERRORINFO("Divider has 0 error!");
		return struMat();
	}
	struMat temp(1. / rdata);
	return lmat * temp;
}
struMat operator/(struMat& lmat, Scalar rdata) {
	ArayData ldata = lmat.GetData();
	if (!rdata) {
		PRINTERRORINFO("Divider has 0 error!");
		return struMat();
	}
	return struMat(ldata / ArayData::Constant(ldata.rows(), ldata.cols(), rdata));
}

struMat operator+=(struMat& lmat, struMat& rmat) {
	lmat = lmat + rmat;
	return lmat;
}
struMat operator+=(struMat& lmat, Scalar rdata) {
	lmat = lmat + rdata;
	return lmat;
}

struMat operator*=(struMat& lmat, struMat& rmat) {
	lmat = lmat * rmat;
	return lmat;
}

struMat operator*=(struMat& lmat, Scalar rdata) {
	lmat = lmat * rdata;
	return lmat;
}

ostream& operator<<(ostream& out, struMat& data) {
	out << data.GetData();
	return out;
}

ostream& operator<<(ostream& out, struMat&& data) {
	out << data.GetData();
	return out;
}

struMat& struMat::operator<<(Scalar data) {
	if (index <= size) {
		(*mat)(index / col, index % col) = data;
		++index %= size;
	}
	else {
		PRINTERRORINFO("Insert Error!");
	}
	return *this;
}

struMat struMat::operator=(struMat& data) {
	*mat = data.GetData();
	tyshape = data.GetShapetype();
	return data;
}

struMat struMat::operator=(struMat&& data) {
	*mat = data.GetData();
	tyshape = data.GetShapetype();
	return data;
}

/*
* ============================================================
* ===================== MatrixOP APIs ========================
* ============================================================
*/
Scalar MatrixOp::VectDot(VectData* x, VectData* y) {
	return x->dot(*y);
}

MatrData* MatrixOp::NormalDistMat(int row, int col, Scalar mean /*= 0.*/, Scalar stddev /*= 1.*/) {
	CRandom<>::generator.seed(CRandom<>::generator());
	normal_distribution<Scalar> distribution(mean, stddev);
	MatrData* data = new MatrData(row, col);
	int size = row * col;
	Scalar* temp = new Scalar[size];
	int i;
	for (i = 0; i < size; ++i) {
		temp[i] = distribution(CRandom<Scalar>::generator);
	}
	memcpy(&(*data)(0, 0), temp, size * sizeof(Scalar));
	delete[] temp;
	return data;
}

MatrData* MatrixOp::UniformDistMat(int row, int col, Scalar lb /*= 0*/, Scalar ub /*= 1*/) {
	CRandom<>::generator.seed(CRandom<>::generator());
	uniform_real_distribution<> distribution(lb, ub);
	MatrData* data = new MatrData(row, col);
	int size = row * col;
	Scalar* temp = new Scalar[size];
	int i;
	for (i = 0; i < size; ++i) {
		temp[i] = distribution(CRandom<Scalar>::generator);
	}
	memcpy(&(*data)(0, 0), temp, size * sizeof(Scalar));
	delete[] temp;
	return data;
}

MatrData* MatrixOp::Sigmoid(MatrData* in)
{
	MatrData inExp = 1. + Eigen::exp(-in->array()).array();
	MatrData* out = new MatrData(1. / inExp.array());
	return out;
}

MatrData* MatrixOp::Tanh(MatrData* in)
{
	MatrData inExp = Eigen::exp(in->array());
	MatrData inExpN = Eigen::exp(-in->array());
	MatrData* out = new MatrData((inExp - inExpN).array() / (inExp + inExpN).array());
	return out;
}

void MatrixOp::ReLU(MatrData*& in)
{
	MatrData* t = new MatrData((in->array() > 0).select(*in, 0));
	FreeObj(in);
	in = t;
}