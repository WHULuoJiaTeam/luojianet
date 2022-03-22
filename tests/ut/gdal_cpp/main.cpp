#include "large_process.h"
#include <iostream>

int main(int argc, char** argv) {
	const char* pszSrcFile = argv[1];
	const char* pszDstFile = argv[2];
	const char* pszFormat = "GTiff";
	int nBlockSize = 10000;
	int FLAG = BlockProcess(pszSrcFile, pszDstFile, pszFormat, nBlockSize);
	
	std::cout<<"FLAG: "<<FLAG<<std::endl;
	
	return 1;
}