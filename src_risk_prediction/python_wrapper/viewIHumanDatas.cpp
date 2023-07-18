#include "viewIHumanDatas.h"

//define function to view human states
hde::interfaces::IHumanState* viewIHumanState(yarp::dev::PolyDriver& poly) {
	hde::interfaces::IHumanState* humanState;
	if (poly.view<hde::interfaces::IHumanState>(humanState)) {
		return humanState;
	}
	else {
		return nullptr;
	}
}

// define function to view human dynamics
hde::interfaces::IHumanDynamics* viewIHumanDynamics(yarp::dev::PolyDriver& poly) {
	hde::interfaces::IHumanDynamics* humanDynamics;
	if (poly.view<hde::interfaces::IHumanDynamics>(humanDynamics)) {
		return humanDynamics;
	}
	else {
		return nullptr;
	}
}

// define function to view human wrenches
hde::interfaces::IHumanWrench* viewIHumanWrench(yarp::dev::PolyDriver& poly) {
	hde::interfaces::IHumanWrench* IHumanWrench;
	if (poly.view<hde::interfaces::IHumanWrench>(IHumanWrench)) {
		return IHumanWrench;
	}
	else {
		return nullptr;
	}
}