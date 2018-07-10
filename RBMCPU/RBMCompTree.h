#pragma once
#include "OptPlaceholder.h"
#include "Sigmoid.h"
#include "Variable.h"
#include "ProbPooling.h"
#include "RGLayer.h"
#include "Graph.h"
#include "RGFlowCont.h"
#include "RGFlowCont2D.h"
namespace ct {
	class RBMCompTree
	{
	public:
		RBMCompTree();
		~RBMCompTree();
		static shared_ptr<Graph> getRBMGraph();
		static shared_ptr<Graph> getRBM2DGraph();
	};
}

