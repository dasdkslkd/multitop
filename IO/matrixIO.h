#pragma once
#include<Eigen/Core>
#include<fstream>
#include<string>
using namespace std;

template<class T>
void savemat(string fileName, T matrix)
{
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

template<class T>
void savevec(string fileName, vector<T> matrix)
{
	ofstream file(fileName);
	if (file.is_open())
	{
		for (auto i : matrix)
			file << i << ',' << endl;
		file.close();
	}
}

template<class T>
void savearr(string filename, T* arr, size_t len)
{
	ofstream file(filename);
	if (file.is_open())
	{
		for (int i = 0; i < len; ++i)
			file << arr[i] << ',' << endl;
		file.close();
	}
}

inline Eigen::MatrixXd readmat(string filename)
{

	// the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
	// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format



	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//	  d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
	vector<double> matrixEntries;

	// in this object we store the data from the matrix
	ifstream matrixDataFile(filename);

	// this variable is used to store the row of the matrix that contains commas 
	string matrixRowString;

	// this variable is used to store the matrix entry;
	string matrixEntry;

	// this variable is used to track the number of rows
	int matrixRowNumber = 0;


	while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

		while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
		matrixRowNumber++; //update the column numbers
	}

	// here we convet the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

inline vector<double> readmatrix(string filename)
{
	vector<double> matrixEntries;
	ifstream matrixDataFile(filename);
	string matrixRowString;
	string matrixEntry;
	int matrixRowNumber = 0;
	while (getline(matrixDataFile, matrixRowString))
	{
		stringstream matrixRowStringStream(matrixRowString);
		while (getline(matrixRowStringStream, matrixEntry, ','))
			matrixEntries.push_back(stod(matrixEntry));
		matrixRowNumber++;
	}
	return matrixEntries;
}