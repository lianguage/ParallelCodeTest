#include <iostream>
#include <fstream>

#include <cstdlib>
#include <ctime>

#define filesize 42424242

using namespace std;


int main () {

	ofstream myfile;
	myfile.open ("rand");
	std::srand(std::time(0));
	myfile << filesize << endl;
	for( int i = 0 ; i < filesize ; i++ ){
		myfile << rand() << endl;
	}

	myfile.close();

	return 0;
}