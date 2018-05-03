#include "RBMCompTree.h"

namespace ct {

	RBMCompTree::RBMCompTree()
	{
	}


	RBMCompTree::~RBMCompTree()
	{
	}

	shared_ptr<Graph> ct::RBMCompTree::getRBMGraph()
	{
		auto optPl = make_shared<OptPlaceholder>(OptPlaceholder("x"));
		auto storeVisible = make_shared<Storage>(Storage(optPl, "visibles_pooled"));
		auto coupling = make_shared<Variable>(Variable());
		coupling->value = make_shared<Tensor>(Tensor({ 1 }, { -1.0}));
		coupling->name = "A";
		auto scaling = make_shared<Variable>(Variable());
		scaling->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		scaling->name = "s";
		auto positive = make_shared<RGFlowCont>(RGFlowCont(storeVisible, coupling, scaling, false));

		auto storeHidden_raw = make_shared<Storage>(Storage(positive, "hiddens_raw"));

		auto negative = make_shared<RGFlowCont>(RGFlowCont(storeHidden_raw, coupling, scaling, true));

		auto storeVisible_raw= make_shared<Storage>(Storage(negative, "visibles_raw"));
		auto visible = make_shared<ProbPooling>(ProbPooling(storeVisible_raw));
		auto graph = shared_ptr<Graph>(new Graph(visible));
		return graph;
	}
}
