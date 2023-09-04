#pragma once
#include "tools.h"

struct struMat {
private:
	MatrData* mat;
	emShape tyshape;
	int row, col, size;
	int index = 0;
private:
	void BroadcastIn(MatSize);
	void CompressIn(MatSize);
public:
	struMat();
	struMat(MatrData*, bool ifCopy = true);
	struMat(MatrData&);
	struMat(MatrData&&);
	struMat(struMat*);
	struMat(struMat&);
	struMat(struMat&&);
	struMat(Scalar);
	struMat(int, int, Scalar data = 1);
	struMat(MatSize, Scalar);
	~struMat();

	void SetData(MatrData*);
	//void SetData(MatrData&);
	MatrData GetData();
	MatrData* GetDataPtr();
	MatSize GetShape();
	int GetSize();
	Scalar GetValue(int, int);
	bool ifOneDim();
	emShape GetShapetype();
	void ReshapeIn(MatSize);

	void NegIP();
	struMat* Neg();
	struMat Transpose();
	void TransposeIP();
	struMat* TransposePtr();
	struMat* Scaled(Scalar);
	void ScaledIP(Scalar);
	void AddScalarIP(Scalar);
	struMat* Dot(struMat*);
	struMat* Pow(Scalar coePow = 1, Scalar sign = 1);
	struMat* ExpEach(Scalar coeExp = 1, Scalar sign = 1);
	Scalar Sum();

	static tyTensor* Cat(tyTensor*, tyTensor*, int); 
	static tyTensor* AddMM(Scalar, tyTensor*, Scalar, tyTensor*, tyTensor*);

	friend struMat operator+(struMat&, struMat&);
	friend struMat operator+(Scalar, struMat&);
	friend struMat operator+(struMat&, Scalar);
	friend struMat operator-(struMat&, struMat&);
	friend struMat operator-(Scalar, struMat&);
	friend struMat operator-(struMat&, Scalar);
	friend struMat operator*(struMat&, struMat&);
	friend struMat operator*(Scalar, struMat&);
	friend struMat operator*(struMat&, Scalar);
	friend struMat operator/(struMat&, struMat&);
	friend struMat operator/(Scalar, struMat&);
	friend struMat operator/(struMat&, Scalar);
	friend struMat operator+=(struMat&, struMat&);
	friend struMat operator+=(struMat&, Scalar);
	friend struMat operator*=(struMat&, struMat&);
	friend struMat operator*=(struMat&, Scalar);

	friend std::ostream& operator<<(std::ostream& out, struMat& data);
	friend std::ostream& operator<<(std::ostream& out, struMat&& data);
	struMat& operator<<(Scalar);
	struMat operator=(struMat& data);
	struMat operator=(struMat&& data);
};


namespace MatrixOp {

	/// <summary>
	/// define simple operation of matrdata
	/// </summary>
	/// 
	/// 

	/*
		Input :
			   MatrData* matA : the pointer of the matrixA(or vector)
			   MatrData* matB : the pointer of the matrixB(or vector)
		Retval: if the A and B have the same shape
	*/
	template<class MatType>
	bool IfSameShape(MatType* matA, MatType* matB) {
		if (matA->cols() != matB->cols() || matA->rows() != matB->rows()
			|| matA == nullptr || matB == nullptr) {
			return false;
		}
		return true;
	}
	/*
		Input :
			   MatType* neg    : the pointer of the new negative matrix
			   MatType* ptrmat : the pointer of the original matrix
		Retval: the pointer of the negative of the input-matrix or vector

		Memory Warning:the memory has been allocated in the function
		attention to free the receive-pointer's oringinal data
	*/
	template<class MatType>
	void MatrNega(MatType*& neg, MatType* ptrmat) {
		FreeObj(neg);
		neg = new MatType(-1 * (*ptrmat));
	}

	/*
		Input : the pointer of the matrix
		Retval: the pointer of the copy matrix of the input-matrix

		Memory Warning:the memory has been allocated in the function
		attention to free the receive-pointer's oringinal data
	*/
	template<class MatType>
	MatType* MatrCopy(MatType* ptrmat) {
		return new MatType(*ptrmat);
	}
	/*
		Input :
			   MatType* xcopy : the pointer of the original matrix
			   MatType* x     : the pointer of the copyed matrix
		Retval: the status if this operate is successful

		Memory Warning:there are no additional memory will be allocated,the operate is
		performed in the xcopy pointer.However, must pay attention!: The x do not be
		a nullptr
	*/
	template<class MatType>
	bool MatrCopy(MatType*& xcopy, MatType* x) {
		if (x != nullptr) {
			FreeObj(xcopy);
			xcopy = new MatType(*x);
			return true;
		}
		return false;
	}
	/*
		Input:
			  MatType* xsum: the matrix pointer of the result after added the x*coe
			  MatType* x   : the matrix pointer of the base matrix
			  Scalar coe    : the coefficient to be multiply in the x
		RetVal: the status if the operate is legal or if the operate is succeessful

		Memory Warning:there are no additional memory will be allocated,the result is
		modify in the xsum pointer
	*/
	template<class MatType>
	bool MatrAdd(MatType* xsum, MatType* x, Scalar coe) {
		if (IfSameShape<MatType>(xsum, x)) {
			*xsum = *xsum + (x->array() * coe).matrix();
			return true;
		}
		return false;
	}
	/*
		Input:
			  VectData* x : the vector pointer of x
			  VectData* y : the vector pointer of y
		RetVal: the result of the dot operate between x and y
	*/
	Scalar VectDot(VectData* x, VectData* y);
	/*
		Input:
			  MatType*& mat    : the MatType pointer of mat to be allocated
			  int row, int col : the allocated matrix size
		RetVal: the status if the allocate operate is legal or successful
	*/
	template<class MatType>
	bool AllocMatr(MatType*& mat, int row, int col) {
		FreeObj(mat);
		mat = new MatType(row, col);
		return true;
	}
	/*
		Input:
			  MatType*& result     : the MatType pointer of result
			  MatType* left, right : the left and right coefficient
		RetVal: the status if the allocate operate is legal or successful
	*/
	template<class MatType>
	bool SubMatr(MatType*& result, MatType* left, MatType* right) {
		FreeObj(result);
		result = new MatType(*left - *right);
		return true;
	}
	/*
		Input:
			  MatType* left, right : the left and right coefficient
		RetVal: the MatType pointer of result

		Memory Warning:the memory has been allocated in the function
		attention to free the receive-pointer's oringinal data
	*/
	template<class MatType>
	MatType* SubMatr(MatType* left, MatType* right) {
		if (left == nullptr || right == nullptr)
			return nullptr;
		return new MatType(*left - *right);
	}
	/*
		Input:
			  MatType*& resulte : the MatType pointer of result
			  Scalar coe        : the multiply coefficient
		RetVal: the pointer of the result

		Memory Warning:the memory has been allocated in the function
		attention to free the receive-pointer's oringinal data
	*/
	template<class MatType>
	MatType* MulScalarMatr(MatType* result, Scalar coe) {
		if (result == nullptr)
			return nullptr;
		return new MatType(result->array() * coe);
	}
	/*
		Input:
			  MatType*& resulte : the MatType pointer of result
			  Scalar coe        : the multiply coefficient
		RetVal: the status if the allocate operate is legal or successful

		Memory Warning:there are no additional memory will be allocated,the result is
		modify in the xsum pointer
	*/
	template<class MatType>
	bool MulScalarIPMatr(MatType*& result, Scalar coe) {
		if (result == nullptr) {
			return false;
		}
		*result = result->array() * coe;
		return true;
	}
	template<class MatType>
	MatType* AddMM(Scalar beta, MatType* mat, Scalar alpha, MatType* mat1, MatType* mat2)
	{
		MatType* data;
		data = new MatType(beta * *mat + alpha * (*mat1 * *mat2));
		return data;
	}

	MatrData* NormalDistMat(int row, int col, Scalar mean = 0., Scalar stddev = 1.);
	MatrData* UniformDistMat(int row, int col, Scalar lb = 0, Scalar ub = 1);
	//=========the NN functions=========
	MatrData* Sigmoid(MatrData*);
	MatrData* Tanh(MatrData*);
	void ReLU(MatrData*&);
}