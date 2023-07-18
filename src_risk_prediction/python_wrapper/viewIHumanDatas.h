#include<yarp/dev/all.h>
#include<hde/interfaces/IHumanState.h>
#include<hde/interfaces/IHumanDynamics.h>
#include<hde/interfaces/IHumanWrench.h>

hde::interfaces::IHumanState* viewIHumanState(yarp::dev::PolyDriver& poly);
hde::interfaces::IHumanDynamics* viewIHumanDynamics(yarp::dev::PolyDriver& poly);
hde::interfaces::IHumanWrench* viewIHumanWrench(yarp::dev::PolyDriver& poly);
