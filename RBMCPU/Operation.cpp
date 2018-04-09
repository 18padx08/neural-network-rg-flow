#include "Operation.h"


namespace ct {
	Operation::Operation()
	{
	}


	Operation::~Operation()
	{
		inputs.clear();
		consumers.clear();
		output.reset();
	}

	string Operation::type()
	{
		return "operation";
	}
	
}
