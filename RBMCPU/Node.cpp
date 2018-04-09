#include "Node.h"

namespace ct {
	Node::Node()
	{
	}


	Node::~Node()
	{
		inputs.clear();
		output.reset();
		consumers.clear();
	}
	

}
