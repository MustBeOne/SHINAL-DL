#include "tools.h"
#include <windows.h>
#include <fstream>
#include <iomanip>

using namespace std;
CTools::CTools() {

}

CTools::~CTools() {

}
int* CTools::AllocI(int n) {
	int* val;
	val = new int[n];
	memset(val, 0, n * sizeof(int));
	return val;
}
Scalar* CTools::Alloc(int n) {
	Scalar* val;
	val = new Scalar[n];
	memset(val, 0, n * sizeof(Scalar));
	return val;
}
Scalar** CTools::Alloc(int n1, int n2) {
	Scalar** val;
	int i;
	val = new Scalar* [n1];
	for (i = 0; i < n1; i++) {
		val[i] = new Scalar[n2];
		memset(val[i], 0, n2 * sizeof(Scalar));
	}
	return val;
}


void CTools::FreeI(int*& val) {
	if (!val) return;
	delete[] val;
	val = nullptr;
}
void CTools::Free(Scalar*& val) {
	if (!val) return;
	delete[] val;
	val = nullptr;
}
void CTools::Free(Scalar**& val, int n1) {
	if (!val) return;
	if (n1 == 1) {
		delete[] val;
		val = nullptr;
		return;
	}
	int i;
	for (i = 0; i < n1; i++) {
		delete[]val[i];
		val[i] = nullptr;
	}
	delete[] val;
	val = nullptr;
}
int CTools::Round(Scalar val) {
	int nt;
	nt = (int)val;
	if ((val - nt) >= 0.5) nt = nt + 1;
	return nt;
}
bool CTools::IsEqual(Scalar v1, Scalar v2) {
	if (fabs(v1 - v2) < 1e-5) return true;
	return false;
}
bool CTools::IsZero(Scalar v) {
	if (fabs(v) < 1e-7) return true;
	return false;
}
void CTools::SaveDataFile(string fileName, int n1, int n2, Scalar** val) {
	int i, j;
	ofstream file(fileName.c_str());
	if (file.fail()) {
		return;
	}
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n2; j++) {
			file << val[i][j] << " ";
		}
		file << endl;
	}
	file.close();
}
void CTools::Save1DataFile(string fileName, int n1, Scalar* val) {
	int i;
	ofstream file(fileName.c_str());
	if (file.fail()) {
		return;
	}
	for (i = 0; i < n1; i++) {

		file << val[i] << " ";
		file << endl;
	}
	file.close();
}
void CTools::ProcLineStrTwoParam(string& str, string& unit1, string& unit2) {
	int np;
	np = (int)str.find('=');
	if (np > 0) {
		unit1 = str.substr(0, np);
		unit2 = str.substr(np + 1, str.size() - np);
	}
	else {
		unit1 = str;
	}
}

void CTools::PrintProcessInWin(int epoch, int percen) {
	char info[512];
	COORD band;
	band.X = 0;
	band.Y = epoch * 3;
	HANDLE ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleCursorPosition(ConsoleHandle, band);
	sprintf_s(info, 512, "Training Epoch %d complete : ", epoch);
	cout << info;
	band.X = strlen(info);
	SetConsoleCursorPosition(ConsoleHandle, band);
	cout << setw(percen) << setfill('*') << "";
	cout << "  " << percen << '%';
}

void CTools::PrintTestProcessInWin(int epoch, int percen) {
	char info[512];
	COORD band;
	band.X = 0;
	band.Y = epoch * 3 + 1;
	HANDLE ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleCursorPosition(ConsoleHandle, band);
	sprintf_s(info, 512, "Testing Epoch %d complete : ", epoch);
	cout << info;
	band.X = strlen(info);
	SetConsoleCursorPosition(ConsoleHandle, band);
	cout << setw(percen) << setfill('*') << "";
	cout << "  " << percen << '%';
}
