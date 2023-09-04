#pragma once
#include <map>
#include <set>
#include <vector>
#include <Eigen>
#include <time.h>

using namespace Eigen;

using std::vector;
using std::map;
using std::string;
using std::pair;

#define PI 3.1415926535897932384626433832795

#define PRINTLOCATION printf("%s%d",__FILE__,__LINE__)
#define PRINTERRORINFO(msg) cout<<__FILE__<<__LINE__<<" : "<<msg<<endl

#define iterObjTypeMap(T1,T2)  typedef map<T1, T2>::iterator
#define ObjTypeMap(T1,T2)  typedef map<T1, T2>
#define iterObjTypeVector(T)  typedef vector<T>::iterator
#define ObjTypeVector(T)  typedef vector<T>

#define ITER(container) container::iterator

//Attention:input the pointer！
#define VectExp(D) Eigen::exp(D->array())

#ifndef _ERRORMSG_
#define _ERRORMSG_

static string NeuLayNumSetERROR = "Create Neural Net Failed: Neron Layer Error! The number of layer is lower than the mininum.";
static string LossFuncNumSetERROR = "Create Neural Net Failed: LossFunction Input Error!";
static string TrainNetIndexERROR = "Find Neural Net Failed:the Net **name:[%s]** you input is not exist!";
static string NueLayIndexERROR = "Find Neural Layer Failed:the Layer **name:[%s]** you input is not exist!";
static string LossFuncIndexERROR = "Find Neural Layer Failed:the Loss Function you input is not exist!";
static string LayInsertERROR = "Insert Neuron Layer Error:Too many input or output layer!";

#endif

///-----------------TYPEDEF PART----------------------
/*
* redefine the datatype for all headers
*/

/*
* the Bottom-level Structure based on "Eigen"
*/
using VectData = Eigen::VectorXd;
using RowVectData = Eigen::RowVectorXd;
using MatrData = Eigen::MatrixXd;
using ArayData = Eigen::ArrayXXd;

/*
* the container of some class or data-structure based on STL
*/
using VecVect = vector<VectData>;
using VecMatr = vector<MatrData>;
using MapMatr = map<string, MatrData*>;


/*
* the PreDeclaration of the class to typedef
*/
//------------the BottomLevel DataStructure-----------------
/// <summary>
/// the repack of eigen datastructure to perform broadcast
/// </summary>
struct struMat;
/// <summary>
/// the BaseClass of sequence-container
/// </summary>
struct struSeqBase;
/// <summary>
/// the dataset which should be paceked in batches
/// </summary>
struct struBatchedData;
/// <summary>
/// the rawdataset hasn't been processed which read from the datasetFiles
/// </summary>
struct struRawData;

/// <summary>
/// the MAIN CLASS in the framework.This is the base to realize the 
/// Autograd
/// Neural Network Computation
/// Data I/O
/// Computation Tree Node
/// </summary>
class CTensor;
/// <summary>
/// Both are the Autograd Base.
/// The BackwardBase is the BaseClass Node in the Backward Mode.
/// The ForwardBase  is the BaseClass Node in the Forward  Mode.
/// </summary>
class BackwardBase;
class ForwardBase;
//------------the NNConstructure DataStructure-----------------
/// <summary>
/// the NN Optimizer BaseClass and also the Required Part of a Neural Network
/// </summary>
class COptimizerBase;
/// <summary>
/// the manager of the parameter of a neural network.Also the input of the Optimizer
/// </summary>
class CParameter;
/// <summary>
/// the BaseClass of 
/// LayerBase
/// LossFunctionBase
/// ActivationFunctionBase
/// CustomNN
/// </summary>
class CModuleBase;
class CLinearLayerBase;
class CActivationsFunction;
/// <summary>
/// the Loader of the Dataset Class to process the datainput and target for the neural network
/// </summary>
class CDataLoader;
//------------the Typedef Part For all Headers-----------------
using Scalar = double;

using tyTensor = struMat; 
using PtrMat = struMat*; 
using PtrSeq = struSeqBase*; 
using PtrRawData = struRawData*; 
using PtrTen = CTensor*; 
using PtrPara = CParameter*; 
using PtrLossFunc = CModuleBase*; 
using PtrOptm = COptimizerBase*; 

using VecTsr = vector<CTensor*>; 
using VecBcwd = vector<BackwardBase*>; 
using VecFowd = vector<ForwardBase*>; 
using VecModule = vector<CModuleBase*>; 
using VecParam = vector<PtrTen>; 
using VecParamLBFGS = vector<pair<tyTensor*, tyTensor*>>; 

typedef std::set<CTensor*> setTsr;
typedef std::set<ForwardBase*> setFwd;

using prDepend = pair<CTensor*, tyTensor*>;
//           last node-ptr   raw grad of node
using MpFunDepend = vector<prDepend>;

using ItParam = pair<VecParam*, VecModule*>;
using LoadData = pair<vector<PtrMat>, vector<PtrMat>>;
using MatSize = pair<int, int>;
//
//

typedef bool (*FuncEval)(MatrData*, MatrData*);

//Specific for L-BFGS
//typedef PtrTen(CModuleBase::* FuncClosure)();
typedef PtrTen(*FuncClosure)();


enum emShape :char {
	FULL = 7, ROW = 5, COL = 4, ONE = 2, NONE = 0
};


enum class DistriType {
	Normal,
	Poisson
};
enum class emNeuronType :int {
	SIGMOID,
	SOFTMAX,
	TANH,
	NONE
};
enum class emInitType :int {
	Normal
};
enum class emLayerType :int {
	INPUT,
	HIDDEN,
	OUTPUT,
};

enum class EmFileType :unsigned char {
	NONE,
	MAT_FILE
};

enum class emNeuralNetType :int {
	BP
};

enum class emNetDataType :int {
	MNIST_IMAGE
};


enum class emLossFuncType :int {
	MSE,
	CROSSENTROPY,
	LOGARITHMLIKELIHOOD
};

enum class EmDatasetType : unsigned char {
	NONE,
	MNIST,
	PINNS_BURGERSEQUA
};

enum class EmBackwardType :uint64_t {
	NONE,
	ACCUMULATE,
	NEG,
	COPY,
	ADD,
	SUB,
	MUL,
	DIV,
	SUM,
	MM,
	ADDMM,
	POW,
	EXP,
	ACC,
	MEAN,
	CAT,
	SLICE,
	SIGMOID,
	TANH,
	MSE
};