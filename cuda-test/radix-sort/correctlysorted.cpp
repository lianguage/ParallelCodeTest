#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char * argv[]){
	if(argc !=  2){
		cout <<   "usage: " << argv[0] << "<filename>" << endl;
	}
	else{
		string line;
		ifstream myfile(argv[1]);
		int size = 0;
		int filesize = 0;
		int lines = 0;
		unsigned int * h_numbers;
		unsigned int * h_position;
		if( myfile.is_open()){
			getline(myfile,line);
			size = atoi(line.c_str());//first line of file is assumed to show number of elements in file.
			filesize = sizeof(int)*size;
			h_numbers = (unsigned int *)malloc(filesize);
			h_position = (unsigned int *)malloc(filesize);
			int i = 0;
			//cout << "moo:" << size << endl;
			while(getline(myfile, line)){
				 //cout << line << endl;
				 h_numbers[i] = atoi(line.c_str());
				 h_position[i] = i;
				 lines++;
				 i++;
			}
			myfile.close();
		}
		else {
			cout << "Sorry mate, can't load file" << endl;
			return 0;
		}

		for( int i = 0, prev = 0 ; i < size ; i++ ){
			if ( prev > h_numbers[i] ){
				cout << "nope: i = " << i <<endl;
			}
		}

		free(h_numbers);
		free(h_position);

	}

}