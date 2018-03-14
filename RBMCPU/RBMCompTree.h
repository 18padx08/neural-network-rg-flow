#pragma once
#include "OptPlaceholder.h"
#include "Sigmoid.h"
#include "Variable.h"
#include "ProbPooling.h"
#include "RGLayer.h"
#include "Graph.h"
namespace ct {
	class RBMCompTree
	{
	public:
		RBMCompTree();
		~RBMCompTree();
		static shared_ptr<Graph> getRBMGraph();
	};
}

