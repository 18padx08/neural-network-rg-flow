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
		auto kappa = make_shared<Variable>(Variable());
		kappa->value = make_shared<Tensor>(Tensor({ 1 }, { 0.2}));
		kappa->name = "kappa";
		auto Av = make_shared<Variable>(Variable());
		Av->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		Av->name = "Av";
		auto Ah = make_shared<Variable>(Variable());
		Ah->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		Ah->name = "Ah";
		auto lambda = make_shared<Variable>(Variable());
		lambda->value = make_shared<Tensor>(Tensor({ 1 }, { 0 }));
		lambda->name = "lambda";
		auto positive = make_shared<RGFlowCont>(RGFlowCont(storeVisible, kappa, Av, Ah, lambda, false));

		auto storeHidden_raw = make_shared<Storage>(Storage(positive, "hiddens_raw"));

		auto negative = make_shared<RGFlowCont>(RGFlowCont(storeHidden_raw, kappa, Av, Ah, lambda, true));

		auto storeVisible_raw= make_shared<Storage>(Storage(negative, "visibles_raw"));
		auto visible = make_shared<ProbPooling>(ProbPooling(storeVisible_raw));
		auto graph = shared_ptr<Graph>(new Graph(storeVisible_raw));
		return graph;
	}
}
