#if 1
#pragma once
#include "tools.h"
#include "struMat.h"
class CTensor {
private:
	/*
	* numerical member
	*/
	tyTensor* m_pData;
	tyTensor* m_pGrad;
	/*
	* autograd structure member
	*/
	BackwardBase* m_pGradFn;
	setTsr* m_stOrigin;
	bool if_grad;
	bool is_leaf;
	/**/
	VecTsr* m_vcRetainGraph;
	bool if_gradGet;
#ifdef FORWARD_AUTOGRAD_MODLE
	setFwd* m_stFwPath;
	VecFowd* m_vcForward;
#endif // FORWARD_AUTOGRAD_MODLE
	setFwd* m_stFwPath;
	VecFowd* m_vcForward;

public:
	explicit CTensor();
	explicit CTensor(int, int, Scalar data = 0., bool if_grad = false);
	explicit CTensor(Scalar, bool if_grad = false);
	explicit CTensor(tyTensor* data, bool if_grad = false, bool ifCopy = true);
	explicit CTensor(tyTensor& data, bool if_grad = false);
	explicit CTensor(tyTensor&& data, bool if_grad = false);
	explicit CTensor(MatrData* data, bool if_grad = false, bool ifCopy = true);
	explicit CTensor(MatrData& data, bool if_grad = false);
	explicit CTensor(MatrData&& data, bool if_grad = false);
	explicit CTensor(CTensor* tensor);
	explicit CTensor(CTensor& tensor);
	explicit CTensor(CTensor&& tensor);
	virtual ~CTensor();

	//the read-write APIs
	tyTensor* GetData();
	void SetData(tyTensor*);
	void SetData(MatrData*);

	tyTensor* GetGrad();
	void ShowGrad();
	void SetGrad(tyTensor*);
	void AddGrad(Scalar);
	void AddGrad(tyTensor*);

	BackwardBase* GetGradFn();
	void SetGradFn(BackwardBase*);

	bool IfGrad(void);
	void SetGradSta(bool);

	bool IsLeaf(void);
	void SetLeafSta(bool);

	bool IsGradGet(void);
	void SetGradGetSta(bool);
	setTsr* GetOrigin();
	void SetOrigin(setTsr*);
	//the storage-manage APIs
	void ClearGrad();
	void ClearData();
	void ZeroGrad();
	void DeleteInleafNode(bool ifRetainSelf = false);
	void SetRetainedGraph(VecTsr*);
	void CleanGraph();

	tyTensor* GetSameSizeData(Scalar);


#ifdef FORWARD_AUTOGRAD_MODLE
	void ClearForwawrdGraph();
	VecFowd* GetForward();
	void SetForward(ForwardBase*);
#endif
public:
	void RequireGrad(bool);
	void RetainGrad();
	/*
	* Followings are the autograd APIs in Backward-Mode.The "Backward"
	* API is to realize the cur-node's gradients to all Tensors.
	* Others are to realize the 2nd,3rd or higher order gradient.
	*/
	//these are Assistant-Functions
	bool IfSearchDir(CTensor*);
	//these are Main-APIs
	void Backward(bool retainGrad = false, bool retainFwGraph = false);
	CTensor* AutoGradBwMode(CTensor*, bool ifRetainGraph = false, bool ifCreateGraph = false);

#ifdef FORWARD_AUTOGRAD_MODLE
	/*
	* Followings are the autograd APIs in Forward-Mode. However, there
	* are still some technical-trouble to perform the dot-product of
	* Matrix. These will be solved later.
	*/
	//these are Assistant-Functions
	void Forward(CTensor*, VecTsr*, VecTsr*);
	bool SearchPath(CTensor*);
	void ClearPath();
	//these are Main-APIs
	CTensor* AutoGradFwMode(CTensor*, bool ifRetainGraph = false, bool ifCreateGraph = false);
#endif // FORWARD_AUTOGRAD_MODLE

public:
	CTensor* Transpose();
	void TransposeIP();
	void NegData();
	/*
	* The similar Function to the "Concatenation",but the
	*  concatenation tensor is Zeros.
	* "dim" indicate the dimension to concatenation
	* "leng" indicate the leng of the tensor
	*/
	CTensor* FillWith0(int dim, int leng, int pos);
	/*
	* numerical operation APIs,each has its specified backward process
	*/

	/// <summary>
	/// Get "this" tensor's sum of each element
	/// </summary>
	/// <returns>
	/// the retval is a tensor object which contain a scalar
	/// </returns>
	CTensor* Sum();
	/// <summary>
	/// the exponentiation of the tensor
	/// </summary>
	/// <param name="powCoe">
	///	the power of the exponentiation for "this" tensor object
	/// </param>
	/// <param name="sign">
	/// the multiplier of the result,equally a scaler to each elements of the tensor
	/// </param>
	/// <returns>
	/// the result of the exponentiation,a Tensor object
	/// </returns>
	CTensor* Pow(Scalar powCoe, Scalar sign = 1, Scalar alpha = 0.);
	/// <summary>
	/// the matrix's dot-product operate
	/// </summary>
	/// <param name="rTensor">
	/// the right-hand tensor to take part in the operation,equally (this .* rtensor)
	/// </param>
	/// <returns></returns>
	CTensor* Dot(CTensor* rTensor, bool lT = false, bool rT = false);
	static CTensor* Dot(CTensor* lTensor, CTensor* rTensor, bool lT = false, bool rT = false);
	/// <summary>
	/// the following two APIs aims to perform that the size-matching all-scaler matrix dot the tensor(the "scaler" is not the I-matrix * scaler!)
	/// the "dim" is the rest dofs
	/// </summary>
	static CTensor* Dot(Scalar lScaler, CTensor* rTensor, int dim = 1, bool rT = false);
	static CTensor* Dot(CTensor* lTensor, Scalar rScaler, int dim = 1, bool lT = false);
	/// <summary>
	/// Perform the out = beta * mat + alpha * (mat1 @ mat2).(the @ is the matrix-dot operation)
	/// </summary>
	/// <param name="beta">the scaler to multiply mat</param>
	/// <param name="mat">the first matrix in the equation</param>
	/// <param name="alpha">the scaler to multiply the result of mat1 @ mat2</param>
	/// <param name="mat1">the second matrix in the equation</param>
	/// <param name="mat2">the third matrix in the equation</param>
	/// <returns></returns>
	static CTensor* AddMM(Scalar beta, CTensor* mat, Scalar alpha, CTensor* mat1, CTensor* mat2);
	/// <summary>
	/// the clone of "this" tensor 
	/// </summary>
	/// <returns></returns>
	CTensor* Copy();
	/// <summary>
	/// Exponential Operation of Natural Logarithm,
	/// and each elements of "this" tnesor are the power of the Natural number
	/// </summary>
	/// <param name="coeExp">a scaler multiplier to the power,equally sign*E^(coePow*this) </param>
	/// <param name="sign">a scaler multiplier to the result</param>
	/// <returns></returns>
	CTensor* ExpEach(Scalar coeExp = 1, Scalar sign = 1);
	/// <summary>
	/// compute the reciprocal of "this" tensor which can be multiplied a 
	/// scaler.Equally the Pow(-1,dynamic_data).
	/// </summary>
	/// <param name="scaler">
	/// the alternative scaler to the reciprocal of "this" tensor
	/// Default: 1
	/// </param>
	/// <returns></returns>
	CTensor* Reciprocal(Scalar scaler = 1);
	/// <summary>
	/// Perform the Quadratic Polynomials to this tensor ,and the return tensor is
	/// retval = f(x) = a * x^2 + b * x +c
	/// </summary>
	/// <param name="a">the polynomials coefficient</param>
	/// <param name="b">the polynomials coefficient</param>
	/// <param name="c">the polynomials coefficient</param>
	/// <returns></returns>
	CTensor* QuadraPoly(Scalar a = 0., Scalar b = 1., Scalar c = 0.);
	/// <summary>
	/// Do the concatenation on the left and right tensors.The default dimension
	/// which decide the concatenation direction is 0, equally the first dimension.
	/// </summary>
	/// <param name="left">the left tensor to participate in the operation</param>
	/// <param name="right">the right tensor to participate in the operation</param>
	/// <param name="dim">the indicated dimension to concatenation</param>
	/// <returns></returns>
	static CTensor* Concatenation(CTensor* left, CTensor* right, int dim = 0);
	/// <summary>
	/// Calculate the mean of the tensor,the retval is a one-element tensor
	/// </summary>
	/// <param name="pTen">the tensor to be calculate mean</param>
	/// <returns>the result of the mean of the input tensor</returns>
	static CTensor* Mean(CTensor* pTen);

	static CTensor* Split(CTensor* input) {};
	/// <summary>
	/// Do scaling transformation on this tensor in place
	/// </summary>
	/// <param name="scaler">the proportion to this tensor</param>
	/// <returns></returns>
	void ScaleIP(Scalar scaler);
	/// <summary>
	/// Do add up scalar on this tensor in place
	/// </summary>
	/// <param name="scaler">the proportion to this tensor</param>
	/// <returns></returns>
	void AddScalarIP(Scalar scaler);
	/// <summary>
	/// Multiple tensors' add operation
	/// </summary>
	/// <param name="tsArr">the vector object-pointer containing multiple tensors</param>
	/// <returns>the accumulated result tensor</returns>
	static CTensor* Accumulate(VecTsr* tsArr);
	/*
	* For the standard operator overloading of tensor, the same mode as that of pythoch is adopted.
	* The default is array operation, and the broadcast mechanism and automatic derivation
	* are implemented
	*/
	friend CTensor* operator+(Scalar, CTensor&);
	friend CTensor* operator+(CTensor&, Scalar);
	friend CTensor* operator+(CTensor&, CTensor&);
	friend CTensor* operator-(Scalar, CTensor&);
	friend CTensor* operator-(CTensor&, Scalar);
	friend CTensor* operator-(CTensor&, CTensor&);
	CTensor* operator-();
	friend CTensor* operator*(CTensor&, CTensor&);
	friend CTensor* operator*(CTensor&, Scalar);
	friend CTensor* operator*(Scalar, CTensor&);
	friend CTensor* operator/(CTensor&, CTensor&);
	friend CTensor* operator/(CTensor&, Scalar);
	friend CTensor* operator/(Scalar, CTensor&);
	//
	CTensor& operator=(CTensor&);
	CTensor& operator=(CTensor&&);
	bool operator<(CTensor&);
	friend std::ostream& operator<<(std::ostream&, CTensor&);
	friend std::ostream& operator<<(std::ostream&, CTensor&&);
	CTensor& operator<<(Scalar);
};

#endif