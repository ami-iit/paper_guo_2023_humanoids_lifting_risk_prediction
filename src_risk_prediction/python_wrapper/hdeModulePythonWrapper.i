%module hdeModulePythonWrapper

%include "std_array.i"
%import "yarp.i"

%{
#include <yarp/dev/all.h>
#include <hde/interfaces/IHumanState.h>
#include <hde/interfaces/IHumanDynamics.h>
#include <hde/interfaces/IHumanWrench.h>
#include <viewIHumanDatas.h>
%}

%include<hde/interfaces/IHumanState.h>
%include<hde/interfaces/IHumanDynamics.h>
%include<hde/interfaces/IHumanWrench.h>
%include<viewIHumanDatas.h>

%template(DArrayThree) std::array<double, 3>;
%template(DArrayFour) std::array<double, 4>;
%template(DArraySix) std::array<double, 6>;
