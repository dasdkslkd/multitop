#include "fem.h"
#include "mmacontext.h"
#ifdef WIN32 //Windows
#include <direct.h>
#include <io.h>
#else  // Linux
#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

int main(int argc, char* argv[])
{
	int nelx = 6;
	int nely = 3;
	int nelz = 3;
	double volfrac = 0.4;
	Femproblem fem(nelx, nely, nelz, volfrac, 1);
	vector<int> cons(3 * (nely + 1) * (nelz + 1));
	for (int i = 0; i < cons.size(); ++i)
		cons[i] = i;
	fem.setconstrain(move(cons));
	Eigen::VectorXd force=Eigen::VectorXd::Constant(fem.ndof,0);
	for (int i = 0; i <= nely; ++i)
	{
		force[(nelx * (nely + 1) * (nelz + 1) + i) * 3 + 2] = -1;
	}
	fem.setforce(force);
	mmacontext mma(fem);

	string sampleid = "";
	if (argc == 2)
		sampleid = string(argv[1]) + "\\";
	string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\" + sampleid;
	if (access(outpath.c_str(), 0) == -1) { //判断该文件夹是否存在
#ifdef WIN32
		int flag = mkdir(outpath.c_str());  //Windows创建文件夹
#else
		int flag = mkdir(dir.c_str(), S_IRWXU);  //Linux创建文件夹
#endif
	}
	mma.solve_gpu(outpath);
	//mma.solve();
	return 0;
}