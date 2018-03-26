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
		auto positive = make_shared<RGLayer>(RGLayer(storeVisible, coupling, false));
		//sigmoid is in RGlayer implemented
		//auto sigmoid1 = make_shared<Sigmoid>(Sigmoid(positive));
		auto storeHidden_raw = make_shared<Storage>(Storage(positive, "hiddens_raw"));
		auto hiddens = make_shared<ProbPooling>(ProbPooling(storeHidden_raw));
		auto storeHidden = make_shared<Storage>(Storage(hiddens, "hiddens_pooled"));
		auto negative = make_shared<RGLayer>(RGLayer(storeHidden, coupling, true));
		//sigmoid is in RGlayer implemented
		//auto sigmoid2 = make_shared<Sigmoid>(Sigmoid(negative));
		auto storeVisible_raw= make_shared<Storage>(Storage(negative, "visibles_raw"));
		auto visible = make_shared<ProbPooling>(ProbPooling(storeVisible_raw));
		auto graph = make_shared<Graph>(Graph(visible));
		return graph;
	}
}
