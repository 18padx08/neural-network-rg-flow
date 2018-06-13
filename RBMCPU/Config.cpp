#include "Config.h"



REGISTERED_TESTS Config::enumFromString(string str)
{
	if (str == "plotRGFlowLamNeq0") {
		return REGISTERED_TESTS::plotRGFlowLamNeq0;
	}
	else if (str == "plotNonZeroLamTests") {
		return REGISTERED_TESTS::plotNonZeroLamTests;
	}
	return REGISTERED_TESTS::None;
}

function<void()> Config::getFunction(REGISTERED_TESTS currentTest, map<string, double> num_vars, map<string, string> str_vars, map<string,vector<double>> list_vars)
{
	function<void()> f;
	switch (currentTest) {
	case REGISTERED_TESTS::plotRGFlowLamNeq0:
		f = [=] 
		{
			if (list_vars.size() <= 0) {
				if (num_vars.find("kappa") == num_vars.end()) return;
				if (num_vars.find("lambda") == num_vars.end()) return;
				if (num_vars.find("batchsize") == num_vars.end()) return;

				RGFlowTest test;
				double kappa = num_vars.at("kappa");
				double lambda = num_vars.at("lambda");
				int batch_size = num_vars.at("batchsize");
				std::cout << "Start running plotRGFlowLamNeq0 with" << std::endl;
				std::cout << "kappa = " << kappa << std::endl;
				std::cout << "lambda = " << lambda << std::endl;
				std::cout << "batchsize = " << batch_size << std::endl;
				test.plotRGFlowLamNeq0(kappa, lambda, batch_size);
			}
			else {
				if (list_vars.find("kappa") == list_vars.end()) return;
				if (list_vars.find("lambda") == list_vars.end()) return;
				if (list_vars.find("batchsize") == list_vars.end()) return;

				vector<double> kappas = list_vars.at("kappa");
				vector<double> lambdas = list_vars.at("lambda");
				vector<int> batchsizes;
				for (auto t : list_vars.at("batchsize")) {
					batchsizes.push_back((int)t);
				}
				for (auto bs : batchsizes) {
					for (auto k : kappas) {
						for (auto l : lambdas) {
							std::cout << "Start plotRGFlowLamNeq0 with parameters" << std::endl;
							std::cout << "kappa: " << k <<std::endl;
							std::cout << "lambda: " << l << std::endl;
							std::cout << "batchsize: " << bs << std::endl << std::endl;;
							RGFlowTest test;
							test.plotRGFlowLamNeq0(k, l, bs);
						}
					}
				}
			}
		};
		break;
	case REGISTERED_TESTS::plotNonZeroLamTests:
		f = [=] {
			if (num_vars.find("kappa") == num_vars.end()) return;
			if (num_vars.find("lambda") == num_vars.end()) return;
			double kappa = num_vars.at("kappa");
			double lambda = num_vars.at("lambda");
			ErrorAnalysis test;
			test.plotNonZeroLamTests(kappa, lambda);
		};
		break;
	}
	return f;
}

Config::Config()
{
	this->config_file = ifstream("config_file_" + to_string(time(NULL)));
}

Config::Config(string config_file)
{
	this->config_file = ifstream(config_file);

	sectionNames = regex("([A-z0-9]+):{1}");
	number = regex("\\d+(?:\\.?\\d*)");
	list = regex("\\{(?=(?:(?:\\w+),*)*\\})((?:(?:(?:\\w+))+?,?)*)");
	listReg = "\\{(?=(?:(?:.+),*)*\\})(?:(?:(?:(?:.+))+?,?)*)";
	varName = regex("^[A-z]+.*");
	varNameReg = "[A-z]+\\w*";
	numberReg = "\\d+(?:\\.?\\d*)";
	end = regex(".*;;.*");
}


Config::~Config()
{
}

void Config::load()
{
	vector<function<void()>> runChain;
	if (this->config_file.is_open()) {
		string line;
		map<string, double> num_vars;
		map<string, vector<double>> list_vars;
		map<string, string> str_vars;
		REGISTERED_TESTS currentTest = REGISTERED_TESTS::None;
		while (getline(this->config_file, line)) {
			//parse line
			if (level == -1 && currentTest == REGISTERED_TESTS::None) {
				//we expect a valid test name with a colon
				smatch matches;
				regex_search(line, matches, sectionNames);
				string testName;
				//the first submatch should be the testmname
				if (matches.size() < 2) continue;
				testName = (*(matches.begin() + 1)).str();

				
				currentTest = enumFromString(testName);
				if (currentTest != REGISTERED_TESTS::None) {
					level = 0;
				}
				continue;
			}
			if (level >= 0) {
				//we are actually in test section
				//check if we reached the end
				if (regex_match(line,end)) {
					runChain.push_back(getFunction(currentTest, num_vars, str_vars, list_vars));
					num_vars.clear();
					str_vars.clear();
					continue;
				}
				
				regex setParameterNum("\\s*(" + varNameReg + ")\\s*=\\s*(" + numberReg + ")\\s*");
				regex setParameterList("\\s*(" + varNameReg + ")\\s*=\\s*(" + listReg  + ")\\s*");
				smatch matches;
				if (regex_match(line, matches, setParameterNum)) {
					if (matches.size() < 3) continue;
					string name = (*(matches.begin() + 1)).str();
					double value = stod((*(matches.begin() + 2)).str());
					num_vars[name] = value;
				}
				else if (regex_match(line, matches, setParameterList)) {
					smatch numbers;
					string name = (*(matches.begin() + 1)).str();
					string theList = (*(matches.begin()+2)).str();
					vector<double> tmp;
					while (regex_search(theList, numbers, number)) {
						auto theNumberString = (*numbers.begin()).str();
						tmp.push_back(stod(theNumberString));
						theList = numbers.suffix().str();
					}
					list_vars[name] = tmp;
				}
			}
		}
		
	}
	else {
		std::cout << "[DEBUG] config_file is not open" << std::endl;
	}
	functions = runChain;
}

void Config::run()
{
	std::cout << "Start running with config" << std::endl;
	std::cout << "We've got " << functions.size() << " functions to run" << std::endl <<std::endl;
	for (auto el  : functions) {
		el();
		std::cout << std::endl << std::endl;
	}
	std::cout << "We're finished running!" << std::endl;
}
