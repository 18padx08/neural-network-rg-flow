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
		auto coupling = make_shared<Variable>(Variable());
		coupling->value = make_shared<Tensor>(Tensor({ 1 }, { -1 }));
		auto positive = make_shared<RGLayer>(RGLayer(optPl, coupling, false));
		auto sigmoid1 = make_shared<Sigmoid>(Sigmoid(positive));
		auto hiddens = make_shared<ProbPooling>(ProbPooling(sigmoid1));
		auto storeHidden = make_shared<Storage>(Storage(hiddens, "hiddens"));
		auto negative = make_shared<RGLayer>(RGLayer(storeHidden, coupling, true));
		auto sigmoid2 = make_shared<Sigmoid>(Sigmoid(negative));
		auto visible = make_shared<ProbPooling>(ProbPooling(sigmoid2));
		auto storeVisible = make_shared<Storage>(Storage(visible, "visibles"));
		auto graph = make_shared<Graph>(Graph(storeVisible));
		return graph;
	}
}
