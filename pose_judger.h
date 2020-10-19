#pragma once

#include "svm/svm.h"
#include "Python.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <mutex>

using namespace std;

#define RELOCALIZATIONJUDGER (RelocalizationJudger::Instance())

struct JudgerModel {
	svm_model *svmModel;
	vector<string> eigenNames;
	vector<double> eigenMeans;		// 训练集特征均值，用于对特征值归一化
	vector<double> eigenStds;		// 训练集特征标准差，用于对特征值归一化
	int32_t eigenRatio;				// 训练集特征归一化倍率，用于对特征值归一化
};

class RelocalizationJudger {
	/* Singleton */
private:
	static RelocalizationJudger * mInstance;
	static mutex mInstanceMutex;
	RelocalizationJudger();
	~RelocalizationJudger();

public:
	static RelocalizationJudger * Instance();
	void ReleaseInstance();

	/* Free and destory */
	void DestoryRawData();
	void DestoryTrainSVMProb();
	void DestoryTestSVMProb();

	/* Configure parameters */
private:
	const char * mPyFilePathCfg;
	const char * mPyFileNameCfg;
	const char * mPredictDataFileNameCfg;
	const char * mAnalysisResultFileNameCfg;
	const char * mSetPyPathFunCfg;
	const char * mRunPyModuleFunCfg;
	const char * mGetEigenNamesFunCfg;
	const char * mGetEigenSpaceFunCfg;
	const char * mGetLableFunCfg;
	const char * mGetTrainEigenFunCfg;
	const char * mGetTrainEigenNormFunCfg;
	const char * mGetTrainLableFunCfg;
	const char * mGetTestEigenFunCfg;
	const char * mGetTestEigenNormFunCfg;
	const char * mGetTestLableFunCfg;
	const char * mGetOriginalTestEigenFunCfg;
	const char * mGetMeanAndStdFunCfg;
	const char * mGetRatioFunCfg;
	const char * mGetSVCParamFunCfg;

	/* Python operate */
private:
	PyObject * mPyModule;				// Python脚本模块

	void LoadPythonModule();
	void SetPythonWorkPath(const string path);
	void PythonTrainAndOptimize();
	void GetRawDataFromPython();
	void GetTrainDataFromPython();
	void GetTestDataFromPython();
	void GetDataFromPython();
	void GetMeanAndStdFromPython();
	void GetRatioFromPython();
	void GetParamsFromPython();

public:
	void RunPythonModule(const string workPath);

	/* Raw data */
private:
	size_t mNumOfEigenElem;
	double *mRawLabel;
	struct svm_node **mRawEigenSpace;
	size_t mRawEigenSpaceLen;
	struct svm_node **mTrainEigenSpace;
	size_t mTrainEigenSpaceLen;
	struct svm_node **mTestEigenSpace;
	size_t mTestEigenSpaceLen;

	/* SVM operate */
private:
	svm_parameter mSVMParam;		// SVM参数
	svm_problem mTrainSVMProb;		// 训练集
	svm_problem mTestSVMProb;		// 测试集
	size_t mTrainEigenSpaceNormLen;
	size_t mTestEigenSpaceNormLen;

public:
	void RunSVMModule();

	/* Judger model */
private:
	JudgerModel mJudgerModel;		// 判断模型

public:
	void SaveJudgerModel(const string path);

	/* Predict and analysis */
private:
	double OldJudger(const svm_node * eigenVec);
	double NewJudger(const svm_node * eigenVec);

public:
	void PredictAndAnalysis(const string path);
};
