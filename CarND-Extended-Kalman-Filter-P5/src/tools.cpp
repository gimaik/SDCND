#include "tools.h"
#include <iostream>
#include <math.h> 
#include <assert.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
								const vector<VectorXd> &ground_truth) {

	assert(estimations.size() > 0);
	assert(estimations.size() == ground_truth.size()); 

	VectorXd rmse = VectorXd(4);
	rmse <<  0, 0, 0, 0;
	
	for(int i=0; i < estimations.size(); ++i){
		VectorXd diff = estimations[i] - ground_truth[i]; 
		diff = diff.array() * diff.array();
		rmse += diff;
	}

	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse; 
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	assert(x_state.size() == 4);
	
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	double sqsum   = pow(px, 2) + pow(py, 2);
	double sqsqsum = sqrt(sqsum);
	double sqcube = pow(sqsum, 3.0 / 2.0);
	double diff1 = vx*py - vy*px;
	double diff2 = vy*px - vx*py;

	if ( fabs(sqsum) < 0.00001 ) {
		sqsum = 0.00001;
	}

	MatrixXd jacobian = MatrixXd(3, 4);
	jacobian << 	px / sqsqsum, py / sqsqsum, 0, 0,
					-py / sqsum, px / sqsum, 0,  0,
					py * diff1 / sqcube, px * diff2 / sqcube, px / sqsqsum,  py / sqsqsum;
	return jacobian;
}


VectorXd Tools::ConvertCartesianToPolar(const VectorXd& x) {
	assert(x.size() == 4);
	VectorXd polar_output = VectorXd(3);

	double radius = sqrt(pow(x[0], 2) + pow(x[1], 2));
	double angle = atan2(x[1], x[0]);
	double angular_velocity = (x[0]*x[2] + x[1]*x[3])/ radius;
  
	polar_output << 	radius,
						angle,
						angular_velocity;
	return polar_output;  
}


float Tools::WrapAnglePi(const float angle)  {
	// scale module 2pi to  wrap the angle between [-pi, pi]
	float angle_result = angle;
	if (angle > 0) { 
		angle_result = fmod(angle + M_PI, 2 * M_PI) - M_PI;
	} else {
		angle_result = fmod(angle - M_PI, 2 * M_PI) + M_PI;
	}
	return angle_result;
}