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
	Femproblem fem(nelx, nely, nelz, volfrac, 0);

	vector<int> cons(3 * (nely + 1) * (nelz + 1));
	for (int i = 0; i < cons.size(); ++i)
		cons[i] = i;
	fem.setconstrain(move(cons));
	Eigen::VectorXd force = Eigen::VectorXd::Constant(fem.ndof, 0);
	for (int i = 0; i <= nely; ++i)
	{
		force[(nelx * (nely + 1) * (nelz + 1) + i) * 3 + 2] = -1;
	}
	//force[nelx * (nely + 1) * (nelz + 1) + static_cast<int>(nelz / 2) * (nely + 1) + static_cast<int>(nely / 2)] = -1;
	fem.setforce(force);

	// vector<int> cons(6*(nely+1));
	// for(int i=0;i<3*(nely+1);++i)
	// {
	// 	cons[i]=i;
	// 	cons[i+3*(nely+1)]=3*nelx*(nely+1)*(nelz+1)+i;
	// }
	// fem.setconstrain(move(cons));
	// Eigen::VectorXd force=Eigen::VectorXd::Constant(fem.ndof,0);
	// for(int i=0;i<nely+1;++i)
	// {
	// 	for(int j=0;j<nelx+1;++j)
	// 	{
	// 		force[3*(j*(nelz+1)*(nely+1)+nelz*(nely+1)+i)+2]=-1;
	// 	}
	// }
	// fem.setforce(force);

	// vector<int> cons(3 * (nely + 1) * (nelz + 1));
	// for (int i = 0; i < cons.size(); ++i)
	// 	cons[i] = i;
	// fem.setconstrain(move(cons));
	// Eigen::VectorXd force=Eigen::VectorXd::Constant(fem.ndof,0);
	// for (int i = 0; i <= nely; ++i)
	// {
	// 	force[(nelx * (nely + 1) * (nelz + 1) + nelz*(nely+1)+i) * 3 + 2] = -1;
	// }
	// fem.setforce(force);

	mmacontext mma(fem);
	string sampleid = "";
	if (argc == 2)
		sampleid = string(argv[1]) + "\\";
	string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\" + sampleid;
	if (access(outpath.c_str(), 0) == -1) { //判断该文件夹是否存在
#ifdef WIN32
		int flag = mkdir(outpath.c_str());  //Windows创建文件夹
#else
		int flag = mkdir(outpath.c_str(), S_IRWXU);  //Linux创建文件夹
#endif
	}
	mma.solve_gpu(outpath);
	// Femproblem fem2(nelx, nely, nelz, volfrac, 1);
	// fem2.setconstrain(move(cons));
	// fem2.setforce(force);
	// mmacontext mma2(fem2);
	// mma2.solve();
	return 0;
}