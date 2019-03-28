#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
 public:
	Tools();

	virtual ~Tools();

	Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
								const std::vector<Eigen::VectorXd> &ground_truth);

	Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

	Eigen::VectorXd ConvertCartesianToPolar(const Eigen::VectorXd& x);

	float WrapAnglePi(const float angle);
};

#endif  // TOOLS_H_
