#include "CFileWriter.h"

string GetCPUInfo()
{
	int cpuInfo[4] = { -1 };
	char cpu_manufacture[32] = { 0 };
	char cpu_type[32] = { 0 };
	char cpu_freq[32] = { 0 };

	__cpuid(cpuInfo, 0x80000002);
	memcpy(cpu_manufacture, cpuInfo, sizeof(cpuInfo));

	__cpuid(cpuInfo, 0x80000003);
	memcpy(cpu_type, cpuInfo, sizeof(cpuInfo));

	__cpuid(cpuInfo, 0x80000004);
	memcpy(cpu_freq, cpuInfo, sizeof(cpuInfo));
	string s1, s2, s3;
	s1 = cpu_manufacture;
	s2 = cpu_type;
	s3 = cpu_freq;
	return s1 + s2 + s3;
}

string GetOsInfo()
{
	string osTy;
#ifdef _WIN32
	osTy = "Windows";
#elif __linux__
	osTy = "Linux";
#elif __APPLE__
	osTy = "Apple";
#endif
	int bits = sizeof(void*) * 8;
	return osTy + std::to_string(bits);
}

