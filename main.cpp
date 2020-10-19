#include <iostream>
#include "Python.h"
#include "svm/svm.h"
#include "pose_judger.h"

using namespace std;

int main(int argc, char **argv)
{
	string workPath;
	if (argc != 2) {
		cout << "Invalid parameter! argc must be 2!" << endl;
		workPath = "../TestAndAnalysis_test/";
		//system("pause");
		//return 0;
	}
	else {
		workPath = argv[1];
	}
	
	string judgerModelPath = workPath + "RelocalizationAnalysis/judger_model.h";
	string relocalizationAnalysisPath = workPath + "RelocalizationAnalysis/";
	
	// Train, optimize and get the parameters
	RELOCALIZATIONJUDGER->RunPythonModule(workPath);
	
	// Train and get the model
	RELOCALIZATIONJUDGER->RunSVMModule();
	
	// Save judger model as .h file
	RELOCALIZATIONJUDGER->SaveJudgerModel(judgerModelPath);
	
	// Analysis and export result
	RELOCALIZATIONJUDGER->PredictAndAnalysis(relocalizationAnalysisPath);

	RELOCALIZATIONJUDGER->ReleaseInstance();
	return 0;
}
