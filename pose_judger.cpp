#include <iostream>
#include "Python.h"
#include "svm/svm.h"
#include "pose_judger.h"

using namespace std;

#define SVM_PARAMS_NUM		5

RelocalizationJudger * RelocalizationJudger::mInstance = nullptr;
mutex RelocalizationJudger::mInstanceMutex;

RelocalizationJudger::RelocalizationJudger()
{
	mPyFilePathCfg = "sys.path.append('./')";
	mPyFileNameCfg = "analysis_module";
	mPredictDataFileNameCfg = "PredictData.csv";
	mAnalysisResultFileNameCfg = "AnalysisResult.txt";
	mSetPyPathFunCfg = "SetWorkPath";
	mRunPyModuleFunCfg = "Run";
	mGetEigenNamesFunCfg = "GetEigenNames";
	mGetEigenSpaceFunCfg = "GetEigenSpace";
	mGetLableFunCfg = "GetLable";
	mGetTrainEigenFunCfg = "GetTrainEigenSpace";
	mGetTrainEigenNormFunCfg = "GetTrainEigenSpaceNormalized";
	mGetTrainLableFunCfg = "GetTrainLable";
	mGetTestEigenFunCfg = "GetTestEigenSpace";
	mGetTestEigenNormFunCfg = "GetTestEigenSpaceNormalized";
	mGetTestLableFunCfg = "GetTestLable";
	mGetMeanAndStdFunCfg = "GetTrainMeanAndStdInNormalization";
	mGetRatioFunCfg = "GetRatioInNormalization";
	mGetSVCParamFunCfg = "GetSVCParams";

	mNumOfEigenElem = 0;
	mRawEigenSpaceLen = 0;
	mTrainEigenSpaceNormLen = 0;
	mTestEigenSpaceNormLen = 0;
	mTrainEigenSpaceLen = 0;
	mTestEigenSpaceLen = 0;

	mRawLabel = nullptr;
	mRawEigenSpace = nullptr;
	mTrainEigenSpace = nullptr;
	mTestEigenSpace = nullptr;
}

RelocalizationJudger::~RelocalizationJudger()
{
	DestoryRawData();
	DestoryTrainSVMProb();
	DestoryTestSVMProb();
	svm_destroy_param(&mSVMParam);
	svm_free_and_destroy_model(&mJudgerModel.svmModel);
}

RelocalizationJudger * RelocalizationJudger::Instance()
{
	if (mInstance == nullptr) {
		mInstanceMutex.lock();
		if (mInstance == nullptr) {
			mInstance = new RelocalizationJudger();
		}
		mInstanceMutex.unlock();
	}
	return mInstance;
}

void RelocalizationJudger::ReleaseInstance()
{
	if (mInstance) {
		mInstanceMutex.lock();
		if (mInstance) {
			delete mInstance;
			mInstance = nullptr;
		}
		mInstanceMutex.unlock();
	}
}

void RelocalizationJudger::DestoryRawData()
{
	if (mRawLabel) {
		delete[] mRawLabel;
	}
	
	if (mRawEigenSpace) {
		for (size_t i = 0; i < mRawEigenSpaceLen; i++) {
			delete[] mRawEigenSpace[i];
		}
		delete[] mRawEigenSpace;
	}

	if (mTrainEigenSpace) {
		for (size_t i = 0; i < mTrainEigenSpaceLen; i++) {
			delete[] mTrainEigenSpace[i];
		}
		delete[] mTrainEigenSpace;
	}

	if (mTestEigenSpace) {
		for (size_t i = 0; i < mTestEigenSpaceLen; i++) {
			delete[] mTestEigenSpace[i];
		}
		delete[] mTestEigenSpace;
	}
}

void RelocalizationJudger::DestoryTrainSVMProb()
{
	if (mTrainSVMProb.y) {
		delete[] mTrainSVMProb.y;
	}
	
	if (mTrainSVMProb.x) {
		for (size_t i = 0; i < mTrainEigenSpaceNormLen; i++) {
			delete[] mTrainSVMProb.x[i];
		}
		delete[] mTrainSVMProb.x;
	}
}

void RelocalizationJudger::DestoryTestSVMProb()
{
	if (mTestSVMProb.y) {
		delete[] mTestSVMProb.y;
	}
	
	if (mTestSVMProb.x) {
		for (size_t i = 0; i < mTestEigenSpaceNormLen; i++) {
			delete[] mTestSVMProb.x[i];
		}
		delete[] mTestSVMProb.x;
	}
}

void RelocalizationJudger::LoadPythonModule()
{
	// Import python module and get global storage containers
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(mPyFilePathCfg); 
	mPyModule = PyImport_ImportModule(mPyFileNameCfg);

	if (!mPyModule) {
		cout << "mPyModule is null, please check .py file path and syntax" << endl;
		system("pause");
		return;
	}
}

void RelocalizationJudger::SetPythonWorkPath(const string path)
{
	// Set global work path for python module
	PyObject *pFunSetPath;
	pFunSetPath = PyObject_GetAttrString(mPyModule, mSetPyPathFunCfg);
	PyObject *pArgs = PyTuple_New(1);
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", path.c_str()));
	PyObject_CallObject(pFunSetPath, pArgs);
}

void RelocalizationJudger::PythonTrainAndOptimize()
{
	// Split dataset and run sklearn to train and optimize model parameters
	PyObject *pFunRunPyModule;
	pFunRunPyModule = PyObject_GetAttrString(mPyModule, mRunPyModuleFunCfg);
	PyObject_CallObject(pFunRunPyModule, NULL);
}

void RelocalizationJudger::GetRawDataFromPython()
{
	PyObject *pFuncGetEigenNames, *pFuncGetEigenSpace, *pFuncGetLabel;
	PyObject *pEigenNames, *pEigenSpace, *pLabel;

	pFuncGetEigenNames = PyObject_GetAttrString(mPyModule, mGetEigenNamesFunCfg);
	pEigenNames = PyObject_CallObject(pFuncGetEigenNames, NULL);

	pFuncGetEigenSpace = PyObject_GetAttrString(mPyModule, mGetEigenSpaceFunCfg);
	pEigenSpace = PyObject_CallObject(pFuncGetEigenSpace, NULL);

	pFuncGetLabel = PyObject_GetAttrString(mPyModule, mGetLableFunCfg);
	pLabel = PyObject_CallObject(pFuncGetLabel, NULL);

	// Check length of data tuple
	size_t sizeOfListEigenSpace = PyList_Size(pEigenSpace);
	size_t sizeOfListLabel = PyList_Size(pLabel);

	if (sizeOfListEigenSpace != sizeOfListLabel) {
		cout << "sizeOfListEigenSpace != sizeOfListLabel" << endl;
		system("pause");
		return;
	}

	// Malloc and fill the names of eigen elements
	mNumOfEigenElem = PyList_Size(pEigenNames);

	for (size_t i = 0; i < mNumOfEigenElem; i++) {
		PyObject *pItemEigenName = PyList_GetItem(pEigenNames, i);
		mJudgerModel.eigenNames.push_back(PyUnicode_AsUTF8(pItemEigenName));
	}

	// Malloc and fill the eigen space and label
	mRawEigenSpaceLen = sizeOfListEigenSpace;
	mRawLabel = new double[mRawEigenSpaceLen];
	mRawEigenSpace = new svm_node *[mRawEigenSpaceLen];

	for (size_t i = 0; i < mRawEigenSpaceLen; i++) {
		mRawEigenSpace[i] = new svm_node[mNumOfEigenElem];
		PyObject *pListEigenVector = PyList_GetItem(pEigenSpace, i);
		for (size_t j = 0; j < mNumOfEigenElem; j++) {
			PyObject *pItemEigen = PyList_GetItem(pListEigenVector, j);
			mRawEigenSpace[i][j].index = j;
			mRawEigenSpace[i][j].value = PyFloat_AsDouble(pItemEigen);
		}
		PyObject *pItemLabel = PyList_GetItem(pLabel, i);
		mRawLabel[i] = PyFloat_AsDouble(pItemLabel);
	}
}

void RelocalizationJudger::GetTrainDataFromPython()
{
	PyObject *pFuncGetXtrain, *pFuncGetXtrainNorm, *pFuncGetYtrain;
	PyObject *pXtrain, *pXtrainNorm, *pYtrain;

	pFuncGetXtrain = PyObject_GetAttrString(mPyModule, mGetTrainEigenFunCfg);
	pXtrain = PyObject_CallObject(pFuncGetXtrain, NULL);

	pFuncGetXtrainNorm = PyObject_GetAttrString(mPyModule, mGetTrainEigenNormFunCfg);
	pXtrainNorm = PyObject_CallObject(pFuncGetXtrainNorm, NULL);

	pFuncGetYtrain = PyObject_GetAttrString(mPyModule, mGetTrainLableFunCfg);
	pYtrain = PyObject_CallObject(pFuncGetYtrain, NULL);

	// Check length of data tuple
	size_t sizeOfListXtrain = PyList_Size(pXtrain);
	size_t sizeOfListXtrainNorm = PyList_Size(pXtrainNorm);
	size_t sizeOfListYtrain = PyList_Size(pYtrain);

	if (sizeOfListXtrain != sizeOfListXtrainNorm
		|| sizeOfListXtrain != sizeOfListYtrain
		|| sizeOfListXtrainNorm != sizeOfListYtrain) {
		cout << "sizeOfListXtrain, sizeOfListXtrainNorm and sizeOfListYtrain is different!" << endl;
		system("pause");
		return;
	}

	// Check number of eigen elements
	PyObject *pListItemXtrainNorm = PyList_GetItem(pXtrainNorm, 0);
	size_t numOfItemXtrainNorm = PyList_Size(pListItemXtrainNorm);

	if (numOfItemXtrainNorm != mNumOfEigenElem) {
		cout << "The number of train data eigen elements is wrong!" << endl;
		system("pause");
		return;
	}

	// Malloc and fill the train eigen space
	mTrainEigenSpaceLen = sizeOfListXtrain;
	mTrainEigenSpace = new svm_node *[mTrainEigenSpaceLen];

	for (size_t i = 0; i < mTrainEigenSpaceLen; i++) {
		mTrainEigenSpace[i] = new svm_node[mNumOfEigenElem];
		PyObject *pListItemXtrain = PyList_GetItem(pXtrain, i);
		for (size_t j = 0; j < mNumOfEigenElem; j++) {
			PyObject *pItemXtrain = PyList_GetItem(pListItemXtrain, j);
			mTrainEigenSpace[i][j].index = j;
			mTrainEigenSpace[i][j].value = PyFloat_AsDouble(pItemXtrain);
		}
	}

	// Malloc and fill the train svm problem
	mTrainEigenSpaceNormLen = sizeOfListXtrainNorm;
	mTrainSVMProb.l = mTrainEigenSpaceNormLen;
	mTrainSVMProb.y = new double[mTrainEigenSpaceNormLen];
	mTrainSVMProb.x = new svm_node *[mTrainEigenSpaceNormLen];

	for (size_t i = 0; i < mTrainEigenSpaceNormLen; i++) {
		mTrainSVMProb.x[i] = new svm_node[mNumOfEigenElem + 1];
		PyObject *pListItemXtrainNorm = PyList_GetItem(pXtrainNorm, i);
		size_t j = 0;
		for (; j < mNumOfEigenElem; j++) {
			PyObject *pItemXtrainNorm = PyList_GetItem(pListItemXtrainNorm, j);
			mTrainSVMProb.x[i][j].index = j;
			mTrainSVMProb.x[i][j].value = PyFloat_AsDouble(pItemXtrainNorm);
		}
		mTrainSVMProb.x[i][j].index = -1;	// Separator in libsvm
		mTrainSVMProb.x[i][j].value = 0;

		PyObject *pItemYtrain = PyList_GetItem(pYtrain, i);
		mTrainSVMProb.y[i] = PyFloat_AsDouble(pItemYtrain);
	}
}

void RelocalizationJudger::GetTestDataFromPython()
{
	PyObject *pFuncGetXtest, *pFuncGetXtestNorm, *pFuncGetYtest;
	PyObject *pXtest, *pXtestNorm, *pYtest;

	pFuncGetXtest = PyObject_GetAttrString(mPyModule, mGetTestEigenFunCfg);
	pXtest = PyObject_CallObject(pFuncGetXtest, NULL);

	pFuncGetXtestNorm = PyObject_GetAttrString(mPyModule, mGetTestEigenNormFunCfg);
	pXtestNorm = PyObject_CallObject(pFuncGetXtestNorm, NULL);

	pFuncGetYtest = PyObject_GetAttrString(mPyModule, mGetTestLableFunCfg);
	pYtest = PyObject_CallObject(pFuncGetYtest, NULL);

	// Check length of data tuple
	size_t sizeOfListXtest = PyList_Size(pXtest);
	size_t sizeOfListXtestNorm = PyList_Size(pXtestNorm);
	size_t sizeOfListYtest = PyList_Size(pYtest);

	if (sizeOfListXtest != sizeOfListXtestNorm
		|| sizeOfListXtest != sizeOfListYtest
		|| sizeOfListXtestNorm != sizeOfListYtest) {
		cout << "sizeOfListXtest, sizeOfListXtestNorm and sizeOfListYtest is different!" << endl;
		system("pause");
		return;
	}

	// Check number of eigen elements
	PyObject *pListItemXtestNorm = PyList_GetItem(pXtestNorm, 0);
	size_t numOfItemXtestNorm = PyList_Size(pListItemXtestNorm);

	if (numOfItemXtestNorm != mNumOfEigenElem) {
		cout << "The number of train data eigen elements is wrong!" << endl;
		system("pause");
		return;
	}

	// Malloc and fill the test eigen space
	mTestEigenSpaceLen = sizeOfListXtest;
	mTestEigenSpace = new svm_node *[mTestEigenSpaceLen];

	for (size_t i = 0; i < mTestEigenSpaceLen; i++) {
		mTestEigenSpace[i] = new svm_node[mNumOfEigenElem];
		PyObject *pListItemXtest = PyList_GetItem(pXtest, i);
		for (size_t j = 0; j < mNumOfEigenElem; j++) {
			PyObject *pItemXtest = PyList_GetItem(pListItemXtest, j);
			mTestEigenSpace[i][j].index = j;
			mTestEigenSpace[i][j].value = PyFloat_AsDouble(pItemXtest);
		}
	}

	// Malloc and fill the test svm problem
	mTestEigenSpaceNormLen = sizeOfListXtestNorm;
	mTestSVMProb.l = mTestEigenSpaceNormLen;
	mTestSVMProb.y = new double[mTestEigenSpaceNormLen];
	mTestSVMProb.x = new svm_node *[mTestEigenSpaceNormLen];

	for (size_t i = 0; i < mTestEigenSpaceNormLen; i++) {
		mTestSVMProb.x[i] = new svm_node[mNumOfEigenElem + 1];
		PyObject *pListItemXtestNorm = PyList_GetItem(pXtestNorm, i);
		size_t j = 0;
		for (; j < mNumOfEigenElem; j++) {
			PyObject *pItemXtestNorm = PyList_GetItem(pListItemXtestNorm, j);
			mTestSVMProb.x[i][j].index = j;
			mTestSVMProb.x[i][j].value = PyFloat_AsDouble(pItemXtestNorm);
		}
		mTestSVMProb.x[i][j].index = -1;	// Separator in libsvm
		mTestSVMProb.x[i][j].value = 0;

		PyObject *pItemYtest = PyList_GetItem(pYtest, i);
		mTestSVMProb.y[i] = PyFloat_AsDouble(pItemYtest);
	}
}

void RelocalizationJudger::GetDataFromPython()
{
	GetRawDataFromPython();
	GetTrainDataFromPython();
	GetTestDataFromPython();
}

void RelocalizationJudger::GetMeanAndStdFromPython()
{
	PyObject *pFuncGetMeanAndStd;
	PyObject *pMeanAndStd;

	// Get mean and std from train dataset
	pFuncGetMeanAndStd = PyObject_GetAttrString(mPyModule, mGetMeanAndStdFunCfg);
	pMeanAndStd = PyObject_CallObject(pFuncGetMeanAndStd, NULL);

	PyObject *pListItemMean = PyList_GetItem(pMeanAndStd, 0);
	for (size_t index = 0; index < mNumOfEigenElem; index++) {
		PyObject *pItemMean = PyList_GetItem(pListItemMean, index);
		mJudgerModel.eigenMeans.push_back(PyFloat_AsDouble(pItemMean));
	}

	PyObject *pListItemStd = PyList_GetItem(pMeanAndStd, 1);
	for (size_t index = 0; index < mNumOfEigenElem; index++) {
		PyObject *pItemStd = PyList_GetItem(pListItemStd, index);
		mJudgerModel.eigenStds.push_back(PyFloat_AsDouble(pItemStd));
	}
}

void RelocalizationJudger::GetRatioFromPython()
{
	PyObject *pFuncGetRatio;
	PyObject *pRatio;

	// Get mean and std from train dataset
	pFuncGetRatio = PyObject_GetAttrString(mPyModule, mGetRatioFunCfg);
	pRatio = PyObject_CallObject(pFuncGetRatio, NULL);
	mJudgerModel.eigenRatio = PyLong_AsLong(pRatio);
}

void RelocalizationJudger::GetParamsFromPython()
{
	PyObject *pFuncGetParam;
	PyObject *pParam;

	pFuncGetParam = PyObject_GetAttrString(mPyModule, mGetSVCParamFunCfg);
	pParam = PyObject_CallObject(pFuncGetParam, NULL);

	size_t numOfItemParams = PyList_Size(pParam);
	if (numOfItemParams == SVM_PARAMS_NUM) {
		// Params: C cache_size degree gamma eps
		mSVMParam.svm_type = C_SVC;
		mSVMParam.C = PyFloat_AsDouble(PyList_GetItem(pParam, 0));
		mSVMParam.cache_size = PyFloat_AsDouble(PyList_GetItem(pParam, 1));
		mSVMParam.degree = PyFloat_AsDouble(PyList_GetItem(pParam, 2));
		mSVMParam.gamma = PyFloat_AsDouble(PyList_GetItem(pParam, 3));
		mSVMParam.eps = PyFloat_AsDouble(PyList_GetItem(pParam, 4));
		mSVMParam.kernel_type = RBF;
		mSVMParam.coef0 = 0.0;
		mSVMParam.shrinking = 1;
		mSVMParam.probability = 1;
		mSVMParam.nr_weight = 0;
		mSVMParam.weight_label = NULL;
		mSVMParam.nu = 0.0;
		mSVMParam.weight = NULL;
		mSVMParam.p = 0;
	} else {
		cout << "GetParamsFromPython(): the num of params error!" << endl;
		system("pause");
		return;
	}
}

void RelocalizationJudger::RunPythonModule(const string workPath)
{
	Py_Initialize();

	if (!Py_IsInitialized()) {
		cout << "Py_IsInitialized failed!" << endl;
		system("pause");
		return;
	}

	LoadPythonModule();
	SetPythonWorkPath(workPath);
	PythonTrainAndOptimize();
	GetDataFromPython();
	GetParamsFromPython();
	GetMeanAndStdFromPython();
	GetRatioFromPython();

	Py_Finalize();
}

void RelocalizationJudger::RunSVMModule()
{
	const char * errorMsg = svm_check_parameter(&mTrainSVMProb, &mSVMParam);
	if (errorMsg) {
		cout << "RunSVMModule(): check svm parameter error for " << errorMsg << endl;
		system("pause");
		return;
	}

	mJudgerModel.svmModel = svm_train(&mTrainSVMProb, &mSVMParam);
}

void RelocalizationJudger::SaveJudgerModel(const string path)
{
	FILE *fp;
	fopen_s(&fp, path.c_str(), "w");
	if (fp == nullptr) 		{
		cout << "SaveJudgerModel(): can not open file!" << endl;
		system("pause");
		return;
	}

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	const svm_parameter& param = mJudgerModel.svmModel->param;

	fprintf(fp, "#pragma once\n\n");
	fprintf(fp, "#define SVM_TYPE %d\n", param.svm_type);
	fprintf(fp, "#define KERNEL_TYPE %d\n", param.kernel_type);
	fprintf(fp, "#define GAMMA %g\n", param.gamma);

	fprintf(fp, "#define NR_CLASS %d\n", mJudgerModel.svmModel->nr_class);
	fprintf(fp, "#define TOTAL_SV %d\n", mJudgerModel.svmModel->l);
	fprintf(fp, "#define RHO %g\n", mJudgerModel.svmModel->rho[0]);

	if (mJudgerModel.svmModel->probA) // regression has probA only
		fprintf(fp, "#define PROBA %g\n", mJudgerModel.svmModel->probA[0]);
	if (mJudgerModel.svmModel->probB)
		fprintf(fp, "#define PROBB %g\n", mJudgerModel.svmModel->probB[0]);

	fprintf(fp, "#define EIGEN_ELEM_NUM %d\n", mNumOfEigenElem);
	fprintf(fp, "#define SVM_NORMALIZATION_RATIO %d\n", mJudgerModel.eigenRatio);

	if (mJudgerModel.svmModel->label) {
		fprintf(fp, "int gLabel[2] = { ");
		fprintf(fp, "%d,", mJudgerModel.svmModel->label[0]);
		fprintf(fp, "%d };\n", mJudgerModel.svmModel->label[1]);
	}

	if (mJudgerModel.svmModel->nSV) {
		fprintf(fp, "int gNrSv[2] = { ");
		fprintf(fp, "%d,", mJudgerModel.svmModel->nSV[0]);
		fprintf(fp, "%d };\n", mJudgerModel.svmModel->nSV[1]);
	}

	if (mJudgerModel.eigenMeans.size() != mJudgerModel.eigenStds.size()) {
		cout << "SaveJudgerModel(): mJudgerModel.eigenMeans.size() != mJudgerModel.eigenStds.size()!" << endl;
		system("pause");
		return;
	}

	if (mNumOfEigenElem) {
		fprintf(fp, "const char * gEigenNames[%d] = { ", mNumOfEigenElem);
		for (int i = 0; i < mNumOfEigenElem; i++) {
			if (i == (mNumOfEigenElem - 1))
				fprintf(fp, "\"%s\" };\n", mJudgerModel.eigenNames[i].c_str());
			else
				fprintf(fp, "\"%s\",", mJudgerModel.eigenNames[i].c_str());
		}
		
		fprintf(fp, "double gEigenMeans[%d] = { ", mNumOfEigenElem);
		for (int i = 0; i < mNumOfEigenElem; i++) {
			if (i == (mNumOfEigenElem - 1))
				fprintf(fp, "%.16g };\n", mJudgerModel.eigenMeans[i]);
			else
				fprintf(fp, "%.16g,", mJudgerModel.eigenMeans[i]);
		}

		fprintf(fp, "double gEigenStds[%d] = { ", mNumOfEigenElem);
		for (int i = 0; i < mNumOfEigenElem; i++) {
			if (i == (mNumOfEigenElem - 1))
				fprintf(fp, "%.16g };\n", mJudgerModel.eigenStds[i]);
			else
				fprintf(fp, "%.16g,", mJudgerModel.eigenStds[i]);
		}
	}

	fprintf(fp, "double gSV[%d][%d] = {\n", mJudgerModel.svmModel->l, mNumOfEigenElem + 1);
	const double * const *sv_coef = mJudgerModel.svmModel->sv_coef;
	const svm_node * const *SV = mJudgerModel.svmModel->SV;

	for (int i = 0; i < mJudgerModel.svmModel->l; i++) {
		for (int j = 0; j < mJudgerModel.svmModel->nr_class - 1; j++)
			fprintf(fp, "{ %.16g,", sv_coef[j][i]);

		const svm_node *p = SV[i];
		for (int j = 0; j < mNumOfEigenElem; j++) {
			if ((i == mJudgerModel.svmModel->l - 1) && (j == mNumOfEigenElem - 1))
				fprintf(fp, "%.8g }\n", p->value);
			else if (j == mNumOfEigenElem - 1)
				fprintf(fp, "%.8g },\n", p->value);
			else
				fprintf(fp, "%.8g,", p->value);
			p++;
		}
	}
	fprintf(fp, "};\n");

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) {
		cout << "SaveJudgerModel(): file write error!" << endl;
		system("pause");
		return;
	}
}

double RelocalizationJudger::OldJudger(const svm_node * eigenVec)
{
	double predict_label = 0;
	double valid_data_num = 0;
	double ave_avg_conf = 0;
	double avg_hit_conf = 0;
	double max_acc_conf = 0;
	double max_ser_conf = 0;
	double max_non_conf = 0;

	valid_data_num = eigenVec[0].value;
	max_non_conf = eigenVec[1].value;
	ave_avg_conf = eigenVec[2].value;
	max_ser_conf = eigenVec[3].value;
	avg_hit_conf = eigenVec[4].value;
	max_acc_conf = eigenVec[5].value;

	if ((max_ser_conf > 290) ||
		(max_ser_conf > 200 && max_acc_conf > 55 && ave_avg_conf > 50 && max_non_conf < 150) ||
		(max_ser_conf > 150 && max_acc_conf > 45 && max_non_conf < 130) ||
		(avg_hit_conf > 80 && max_acc_conf > 50 && max_non_conf < 100) ||
		(avg_hit_conf > 90 && valid_data_num > 50))
		predict_label = 1;
	else
		predict_label = 0;

	return predict_label;
}

double RelocalizationJudger::NewJudger(const svm_node * eigenVec)
{
	return svm_predict(mJudgerModel.svmModel, eigenVec);
}

void RelocalizationJudger::PredictAndAnalysis(const string path)
{
	string predictDataPath = path + mPredictDataFileNameCfg;
	string analysisResultPath = path + mAnalysisResultFileNameCfg;

	FILE * predictDataFile;
	fopen_s(&predictDataFile, predictDataPath.c_str(), "w");
	if (predictDataFile == nullptr) {
		cout << "PredictAndAnalysis(): can not open predictDataFile!" << endl;
		system("pause");
		return;
	}

	FILE * analysisResultFile;
	fopen_s(&analysisResultFile, analysisResultPath.c_str(), "w");
	if (analysisResultFile == nullptr) {
		cout << "PredictAndAnalysis(): can not open analysisResultFile!" << endl;
		system("pause");
		return;
	}

	size_t total = 0;
	size_t oldCorrect = 0, newCorrect = 0;
	vector<int> oldTP, oldFP, oldFN, oldTN;
	vector<int> newTP, newFP, newFN, newTN;

	for (size_t i = 0; i < mNumOfEigenElem; i++) {
		fprintf(predictDataFile, mJudgerModel.eigenNames[i].c_str());
		fprintf(predictDataFile, ",");
	}
	fprintf(predictDataFile, "real_value,old_predict_value,old_outliers,real_value,new_predict_value,new_outliers\n");
	
	for (size_t i = 0; i < mTestEigenSpaceLen; i++) {
		// Output eigen vector
		for (size_t j = 0; j < mNumOfEigenElem; j++) {
			fprintf(predictDataFile, "%g,", mTestEigenSpace[i][j].value);
		}

		// Predict
		double oldPredict = 0, newPredict = 0;
		oldPredict = OldJudger(mTestEigenSpace[i]);
		newPredict = NewJudger(mTestSVMProb.x[i]);

		// Analysis and output results
		// Old judger
		fprintf(predictDataFile, "%g,", mTestSVMProb.y[i]);
		fprintf(predictDataFile, "%g,", oldPredict);
		if (mTestSVMProb.y[i] == oldPredict) {
			oldCorrect++;
			fprintf(predictDataFile, ",");
		} else {
			fprintf(predictDataFile, "*,");		// Outliers flag
		}

		if ((oldPredict == 1) && (mTestSVMProb.y[i] == 1))
			oldTP.push_back(i);

		if ((oldPredict == 1) && (mTestSVMProb.y[i] == 0))
			oldFP.push_back(i);

		if ((oldPredict == 0) && (mTestSVMProb.y[i] == 1))
			oldFN.push_back(i);

		if ((oldPredict == 0) && (mTestSVMProb.y[i] == 0))
			oldTN.push_back(i);

		// New judger
		fprintf(predictDataFile, "%g,", mTestSVMProb.y[i]);
		fprintf(predictDataFile, "%g,", newPredict);
		if (mTestSVMProb.y[i] == newPredict) {
			newCorrect++;
			fprintf(predictDataFile, "\n");
		} else {
			fprintf(predictDataFile, "*\n");		// Outliers flag
		}

		if ((newPredict == 1) && (mTestSVMProb.y[i] == 1))
			newTP.push_back(i);

		if ((newPredict == 1) && (mTestSVMProb.y[i] == 0))
			newFP.push_back(i);

		if ((newPredict == 0) && (mTestSVMProb.y[i] == 1))
			newFN.push_back(i);

		if ((newPredict == 0) && (mTestSVMProb.y[i] == 0))
			newTN.push_back(i);

		total++;
	}

	// Output analysis result
	fprintf(analysisResultFile, "**************** old judger predict result ****************\n");
	fprintf(analysisResultFile, "TP = %d\n", oldTP.size());
	fprintf(analysisResultFile, "FP = %d\n", oldFP.size());
	fprintf(analysisResultFile, "FN = %d\n", oldFN.size());
	fprintf(analysisResultFile, "TN = %d\n", oldTN.size());
	fprintf(analysisResultFile, "Accuracy = %g%s\n", (double)oldCorrect / total * 100, "%");

	fprintf(analysisResultFile, "\n");
	fprintf(analysisResultFile, "**************** new judger predict result ****************\n");
	fprintf(analysisResultFile, "TP = %d\n", newTP.size());
	fprintf(analysisResultFile, "FP = %d\n", newFP.size());
	fprintf(analysisResultFile, "FN = %d\n", newFN.size());
	fprintf(analysisResultFile, "TN = %d\n", newTN.size());
	fprintf(analysisResultFile, "Accuracy = %g%s\n", (double)newCorrect / total * 100, "%");

	fclose(predictDataFile);
	predictDataFile = nullptr;

	fclose(analysisResultFile);
	analysisResultFile = nullptr;
}
