#include "kalman_filter.h"
#include "tools.h"
#include <iostream>
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateParameters(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;
  VectorXd y = z - tools.ConvertCartesianToPolar(x_);
  y[1] = tools.WrapAnglePi(y[1]);
  UpdateParameters(y);
}


void KalmanFilter::UpdateParameters(const VectorXd &y) {
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse(); 
  MatrixXd I = MatrixXd::Identity(4,4);

  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}


